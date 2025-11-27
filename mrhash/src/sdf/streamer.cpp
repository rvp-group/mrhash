#include "streamer.cuh"
#include "utils/point_cloud_serializer.h"
#include <unordered_set>

namespace cupanutils {
  namespace cugeoutils {

    template <typename T>
    void
    Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::create(const Eigen::Vector3f& voxel_extents,
                                                                      const uint max_num_sdf_block_integrate_from_global_hash,
                                                                      unsigned int initial_chunk_list_size) {
      voxel_extents_             = voxel_extents;
      const float max_chunk_ext_ = std::max(std::max(voxel_extents_.x(), voxel_extents_.y()), voxel_extents_.z());
      chunk_radius_              = 0.5f * max_chunk_ext_ * sqrt(3.f);
      initial_chunk_list_size_   = initial_chunk_list_size;
      max_num_sdf_block_integrate_from_global_hash_ = max_num_sdf_block_integrate_from_global_hash;
      stream_in_done_                               = true;

      // propagate data to hash structure
      container_->voxel_extents_ = Eig2CUDA(voxel_extents_);

      // use cuda host allocation, allocate page-lock memory on host,
      // this is required for asynchronous stream, parallel copy GPU-CPU

      calculateMemoryUsage();

      // desc hash entry, other voxel
      CUDA_CHECK(cudaMallocHost(&h_SDFBlockDescOutput_, sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_));
      CUDA_CHECK(
        cudaMallocHost(&h_SDFBlockOutput_, sizeof(T) * total_sdf_block_size * max_num_sdf_block_integrate_from_global_hash_));

      CUDA_CHECK(cudaMalloc(&d_SDFBlockDescOutput_, sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_));
      CUDA_CHECK(
        cudaMalloc(&d_SDFBlockOutput_, sizeof(T) * total_sdf_block_size * max_num_sdf_block_integrate_from_global_hash_));

      CUDA_CHECK(cudaMalloc(&d_SDFBlockDescInput_, sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_));
      CUDA_CHECK(cudaMalloc(&d_SDFBlockInput_, sizeof(T) * total_sdf_block_size * max_num_sdf_block_integrate_from_global_hash_));

      CUDA_CHECK(cudaMalloc(&d_SDF_block_counter_, sizeof(uint)));

      container_->updateFieldsDevice(); // update this gets called after constructors

      CUDA_CHECK(cudaEventCreate(&start_event_));
      CUDA_CHECK(cudaEventCreate(&stop_event_));
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::clearGrid() {
      for (auto& [chunk_pos, chunk_ptr] : grid_) {
        chunk_ptr->clear();
        chunk_ptr.reset();
      }
      grid_.clear();
    }
    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::destroy() {
      // Cleanup CUDA events
      if (start_event_) {
        CUDA_CHECK(cudaEventDestroy(start_event_));
        start_event_ = cudaEvent_t{};
      }
      if (stop_event_) {
        CUDA_CHECK(cudaEventDestroy(stop_event_));
        stop_event_ = cudaEvent_t{};
      }

      clearGrid();

      // Cleanup host memory
      if (h_SDFBlockDescOutput_) {
        CUDA_CHECK(cudaFreeHost(h_SDFBlockDescOutput_));
        h_SDFBlockDescOutput_ = nullptr;
      }
      if (h_SDFBlockOutput_) {
        CUDA_CHECK(cudaFreeHost(h_SDFBlockOutput_));
        h_SDFBlockOutput_ = nullptr;
      }

      // Cleanup device memory
      if (d_SDFBlockDescOutput_) {
        CUDA_CHECK(cudaFree(d_SDFBlockDescOutput_));
        d_SDFBlockDescOutput_ = nullptr;
      }
      if (d_SDFBlockOutput_) {
        CUDA_CHECK(cudaFree(d_SDFBlockOutput_));
        d_SDFBlockOutput_ = nullptr;
      }
      if (d_SDFBlockDescInput_) {
        CUDA_CHECK(cudaFree(d_SDFBlockDescInput_));
        d_SDFBlockDescInput_ = nullptr;
      }
      if (d_SDFBlockInput_) {
        CUDA_CHECK(cudaFree(d_SDFBlockInput_));
        d_SDFBlockInput_ = nullptr;
      }
      if (d_SDF_block_counter_) {
        CUDA_CHECK(cudaFree(d_SDF_block_counter_));
        d_SDF_block_counter_ = nullptr;
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::serializeData(const std::string& filename_hash,
                                                                                  const std::string& filename_voxel) const {
      utils::PointCloud hash_points;
      utils::PointCloud voxel_points;
      for (const auto& [chunk_pos, chunk_ptr] : grid_) {
        const data::vector<SDFBlockDesc>& descs = chunk_ptr->getSDFBlockDescs();
        const data::vector<SDFBlock<T>>& blocks = chunk_ptr->getSDFBlocks();
        for (uint k = 0; k < descs.size(); ++k) {
          const Eigen::Vector3i pos      = CUDA2Eig(descs[k].pos);
          const Eigen::Vector3f block_pw = (pos * sdf_block_size).cast<float>() * container_->virtual_voxel_size_;
          Eigen::Vector4f block_color    = Eigen::Vector4f::Zero();
          uint valid_voxels              = 0;
          float block_weight_sum         = 0.f;
          const int scale                = 1 << (finest_block_log2_dim - descs[k].resolution);
          const int num_voxels           = scale * scale * scale;
          const int scaling_factor       = 1 << descs[k].resolution;
          for (uint l = 0; l < num_voxels; ++l) {
            if (blocks[k].data[l].weight > 0) {
              // voxel world coordinates
              const Eigen::Vector3i dl =
                CUDA2Eig(SDFBlock<T>::delinearizeVoxelIndex(l, sdf_block_size / scaling_factor)) * scaling_factor;
              const Eigen::Vector3f voxel_pw = block_pw + dl.cast<float>() * container_->virtual_voxel_size_;

              // voxel color
              Eigen::Vector4f voxel_color;
              if (descs[k].resolution == 0) {
                voxel_color(0) = 1.f;
                voxel_color(1) = 0.f;
                voxel_color(2) = 0.f;
              } else if (descs[k].resolution == 1) {
                voxel_color(0) = 0.f;
                voxel_color(1) = 1.f;
                voxel_color(2) = 0.f;
              }
              voxel_color(3)           = 0;
              const float voxel_weight = static_cast<float>(blocks[k].data[l].weight);
              const float voxel_sdf    = blocks[k].data[l].sdf;
              voxel_points.push_back(voxel_pw, voxel_color, voxel_weight, voxel_sdf);
              // interpolate block color among valid voxels
              block_color += voxel_color;
              block_weight_sum += voxel_weight;
              valid_voxels++;
            }
          }
          if (valid_voxels > 0) {
            const Eigen::Vector4f avg_color = block_color / valid_voxels;
            const float avg_weight          = block_weight_sum / static_cast<float>(valid_voxels);
            hash_points.push_back(block_pw, avg_color, avg_weight);
          }
        }
      }

      utils::PointCloudSerializer::saveToFile(filename_hash, hash_points);
      utils::PointCloudSerializer::saveToFile(filename_voxel, voxel_points);
      std::cout << "Streamer::serializeData | written " << hash_points.size() << " hash points and " << voxel_points.size()
                << " voxels to " << filename_hash << " and " << filename_voxel << std::endl;
    }

    /**
     *
     * STREAM-OUT TO HOST
     *
     **/

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::streamOutToHostPass0(const float3& camera_position,
                                                                                         const float radius) {
      container_->resetHashBucketMutex();
      clearSDFBlockCounter();

      //-------------------------------------------------------
      // Pass 1: Find all SDFBlocks that have to be transfered
      //-------------------------------------------------------

      integrateFromGlobalHashPass1(radius, camera_position);

      const uint n_sdf_block_descs = getSDFBlockCounter();
      if (n_sdf_block_descs >= max_num_sdf_block_integrate_from_global_hash_) {
        std::cerr << "streamOutToHostPass0 | sdf block to stream out: " << n_sdf_block_descs
                  << " | current capacity: " << max_num_sdf_block_integrate_from_global_hash_ << std::endl;
        throw std::runtime_error("not enough memory allocated for intermediate GPU buffer (wants to stream out more block than "
                                 "increase max_num_sdf_block_integrate_from_global_hash");
      }

      if (n_sdf_block_descs > 0) {
        // std::cerr << "SDFBlocks streamed out: " << n_sdf_block_descs << std::endl;

        //-------------------------------------------------------
        // Pass 2: Copy SDFBlocks to output buffer
        //-------------------------------------------------------
        integrateFromGlobalHashPass2(n_sdf_block_descs);

        const int stream_size = n_sdf_block_descs;
        CUDA_CHECK(cudaMemcpy(
          &h_SDFBlockDescOutput_[0], &d_SDFBlockDescOutput_[0], sizeof(SDFBlockDesc) * stream_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_SDFBlockOutput_[0],
                              &d_SDFBlockOutput_[0],
                              sizeof(T) * container_->voxel_block_volume_ * stream_size,
                              cudaMemcpyDeviceToHost));
      }

      curr_stream_out_blocks_ = n_sdf_block_descs;
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::streamOutToCPUPass1CPU() {
      if (curr_stream_out_blocks_ > 0) {
        integrateInChunkGrid(h_SDFBlockDescOutput_, h_SDFBlockOutput_);
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateInChunkGrid(const SDFBlockDesc* h_SDFBlockDescOutput,
                                                                                         const T* h_SDFBlockOutput) {
      int start_idx = 0;
      for (uint i = 0; i < curr_stream_out_blocks_; ++i) {
        const SDFBlockDesc& desc = h_SDFBlockDescOutput[i];
        const int scale          = 1 << (finest_block_log2_dim - desc.resolution);
        const int num_voxels     = scale * scale * scale;
        SDFBlock<T> block(num_voxels);

        for (int j = 0; j < num_voxels; j++) {
          const auto& voxel = h_SDFBlockOutput[start_idx + j];
          block.data.push_back(voxel);
        }

        Eigen::Vector3i pos       = CUDA2Eig(h_SDFBlockDescOutput[i].pos);
        Eigen::Vector3f pw        = pos.cast<float>() * sdf_block_size * container_->virtual_voxel_size_;
        Eigen::Vector3i chunk_pos = worldToChunks(pw);

        const float r               = 0.f;
        const float g               = float(i) / float(curr_stream_out_blocks_);
        const float b               = 1.f - float(i) / float(curr_stream_out_blocks_);
        const Eigen::Vector4f color = Eigen::Vector4f(r, g, b, 1);

        grid_.try_emplace(chunk_pos, std::make_unique<ChunkDesc<T>>(initial_chunk_list_size_));

        // add element to host list
        // if this element is in frustum cannot be accessed by the gpu anyway
        // it lives in host from now
        grid_.at(chunk_pos)->addSDFBlock(desc, block);
        start_idx += num_voxels;
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::streamAllOut() {
      const uint num_pass = (container_->total_size_ + max_num_sdf_block_integrate_from_global_hash_ - 1) /
                            max_num_sdf_block_integrate_from_global_hash_;
      uint streamed_out_blocks = 0;
      for (int pass = 0; pass < num_pass; ++pass) {
        container_->resetHashBucketMutex();
        clearSDFBlockCounter();
        CUDA_CHECK(cudaEventRecord(start_event_, 0));
        integrateFromGlobalHashPass1(pass);
        curr_stream_out_blocks_ = getSDFBlockCounter();

        if (curr_stream_out_blocks_ > 0) {
          //-------------------------------------------------------
          // Pass 2: Copy SDFBlocks to output buffer
          //-------------------------------------------------------
          integrateFromGlobalHashPass2(curr_stream_out_blocks_, pass);

          const int stream_size = curr_stream_out_blocks_;
          CUDA_CHECK(cudaMemcpy(
            &h_SDFBlockDescOutput_[0], &d_SDFBlockDescOutput_[0], sizeof(SDFBlockDesc) * stream_size, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(&h_SDFBlockOutput_[0],
                                &d_SDFBlockOutput_[0],
                                sizeof(T) * container_->voxel_block_volume_ * stream_size,
                                cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaEventRecord(stop_event_, 0));
          CUDA_CHECK(cudaEventSynchronize(stop_event_));
          CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event_, stop_event_));
          integrateInChunkGrid(h_SDFBlockDescOutput_, h_SDFBlockOutput_);
          streamed_out_blocks += curr_stream_out_blocks_;
        }
      }
    }

    /**
     *
     * stream-IN TO DEVICE
     *
     * */

    template <typename T>
    uint Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateInHash(const Eigen::Vector3f& camera_pose,
                                                                                    float radius) {
      Eigen::Vector3i camera_chunk = worldToChunks(camera_pose);
      Eigen::Vector3i chunk_radius = meterToNumberOfChunksCeil(radius);
      Eigen::Vector3i start_chunk  = camera_chunk - chunk_radius;
      Eigen::Vector3i end_chunk    = camera_chunk + chunk_radius;

      uint num_SDF_blocks = 0;
      uint copied_voxels  = 0;

      for (auto& [chunk_pos, chunk_ptr] : grid_) {
        if (!isChunkInSphere(chunk_pos, camera_pose, radius))
          continue;

        const uint num_blocks = chunk_ptr->getNElements();

        if (num_blocks + num_SDF_blocks >= max_num_sdf_block_integrate_from_global_hash_) {
          stream_in_done_ = false;
          return num_SDF_blocks;
        } else {
          for (int i = 0; i < num_blocks; ++i) {
            const SDFBlockDesc& desc = chunk_ptr->getSDFBlockDesc(i);
            const int resolution     = desc.resolution;
            const int scale          = 1 << (finest_block_log2_dim - resolution);
            const int num_voxels     = scale * scale * scale;

            CUDA_CHECK(
              cudaMemcpy(d_SDFBlockDescInput_ + num_SDF_blocks + i, &desc, sizeof(SDFBlockDesc), cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemcpy(d_SDFBlockInput_ + copied_voxels,
                                  chunk_ptr->getSDFBlock(i).data.data(),
                                  sizeof(T) * num_voxels,
                                  cudaMemcpyHostToDevice));
            copied_voxels += num_voxels;
          }
        }

        // Host cleanup
        chunk_ptr->clear();
        num_SDF_blocks += num_blocks;
      }

      stream_in_done_ = true;
      return num_SDF_blocks;
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::stream(const Eigen::Vector3f& camera_position,
                                                                           const float radius) {
      {
        CUDAProfiler::CUDAEvent event(streaming_profiler_);
        CUDA_CHECK(cudaEventRecord(start_event_, 0));
        // stream - out in RAM
        streamOutToHostPass0(Eig2CUDA(camera_position), radius);
        // blocks host until event terminate, here we are
        // copy host middle layer structure to main grid
        CUDA_CHECK(cudaEventRecord(stop_event_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event_, stop_event_));
        streamOutToCPUPass1CPU();

        // stream - in in GPU
        streamInToGPU(camera_position, radius);
      }
      streaming_profiler_.write(curr_stream_in_blocks_ + curr_stream_out_blocks_);
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::streamInToGPU(const Eigen::Vector3f& camera_position,
                                                                                  const float radius) {
      do {
        const uint n_sdf_block_descs = integrateInHash(camera_position, radius);
        curr_stream_in_blocks_       = n_sdf_block_descs;
        if (curr_stream_in_blocks_ > 0) {
          // ! alloc memory for chunks
          uint heap_count_prev; // ptr to the first free block

          CUDA_CHECK(cudaMemcpy(&heap_count_prev, container_->d_heapCounterHigh_, sizeof(uint), cudaMemcpyDeviceToHost));

          uint* d_blocks_ptr;
          CUDA_CHECK(cudaMalloc((void**) &d_blocks_ptr, sizeof(uint) * curr_stream_in_blocks_));

          chunkToGlobalHashPass1(curr_stream_in_blocks_, heap_count_prev, d_SDFBlockDescInput_, d_blocks_ptr);
          chunkToGlobalHashPass2(
            curr_stream_in_blocks_, heap_count_prev, d_SDFBlockDescInput_, (T*) d_SDFBlockInput_, d_blocks_ptr);
          CUDA_CHECK(cudaFree(d_blocks_ptr));
        }
      } while (!stream_in_done_);
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::streamInToGPUChunkNeighborhood(const Eigen::Vector3i& chunk,
                                                                                                   const int radius) {
      const Eigen::Vector3i start_chunk = chunk - Eigen::Vector3i(radius);
      const Eigen::Vector3i end_chunk   = chunk + Eigen::Vector3i(radius);

      for (int x = start_chunk.x(); x < end_chunk.x(); ++x) {
        for (int y = start_chunk.y(); y < end_chunk.y(); ++y) {
          for (int z = start_chunk.z(); z < end_chunk.z(); ++z) {
            const Eigen::Vector3i curr_chunk = Eigen::Vector3i(x, y, z);
            streamInToGPU(chunkToWorld(chunk), 1.1f * getChunkRadiusInMeter());
          }
        }
      }
    }

    /////////////////////////////////////////////////////////////
    ////////////////////////// DEBUG ////////////////////////////
    /////////////////////////////////////////////////////////////

    template <typename T>
    double Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::debugCheckForDuplicates() const {
      uint duplicates_count = 0;
      uint total_grid_count = 0;
      std::unordered_set<SDFBlockDesc, SDFBlockDesc::HashSDFBlockDesc> desc_hash;
      std::cerr << "debugCheckForDuplicates | HashTable\n";
      HashEntry* h_hashTable = new HashEntry[container_->total_size_];
      CUDA_CHECK(
        cudaMemcpy(h_hashTable, container_->d_hashTable_, sizeof(HashEntry) * container_->total_size_, cudaMemcpyDeviceToHost));
      for (uint i = 0; i < container_->total_size_; ++i) {
        if (h_hashTable[i].ptr != FREE_ENTRY) {
          total_grid_count++;
          SDFBlockDesc curr(h_hashTable[i]);
          if (desc_hash.find(curr) == desc_hash.end())
            desc_hash.insert(curr);
          else {
            duplicates_count++;
            // std::cerr << "debugCheckForDuplicates | ptr: " << curr.ptr << " pos: " << curr.pos.x << " " << curr.pos.y << " "
            // << curr.pos.z << std::endl; throw std::runtime_error("debugCheckForDuplicates | duplicate found in streaming hash
            // data (in hash)");
          }
        }
      }

      delete[] h_hashTable;

      std::cerr << "debugCheckForDuplicates | Grid\n";
      for (const auto& [chunk_pos, chunk_ptr] : grid_) {
        const data::vector<SDFBlockDesc>& descs_copy = chunk_ptr->getSDFBlockDescs();
        for (unsigned int k = 0; k < descs_copy.size(); ++k) {
          total_grid_count++;
          if (desc_hash.find(descs_copy[k]) == desc_hash.end())
            desc_hash.insert(descs_copy[k]);
          else {
            duplicates_count++;
            // std::cerr << "debugCheckForDuplicates | ptr: " << descs_copy[k].ptr << " pos: " << descs_copy[k].pos.x << " " <<
            // descs_copy[k].pos.y << " " << descs_copy[k].pos.z << std::endl; throw std::runtime_error("debugCheckForDuplicates
            // | duplicate found in streaming hash data (in grid)");
          }
        }
      }

      // stream
      const double duplicates_ratio = (double) duplicates_count / (double) total_grid_count * 100;
      std::cerr << "debugCheckForDuplicates | duplicates ratio: " << duplicates_ratio << std::endl;
      return duplicates_ratio;
    }

    /// @brief DEBUG calculate and output total memory usage by the Streamer class
    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::calculateMemoryUsage() const {
      double toMB = 1e-6;

      // host buffers
      const uint64_t size_h_SDFBlockDescOutput = sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_;
      const uint64_t size_h_SDFBlockOutput     = sizeof(T) * total_sdf_block_size * max_num_sdf_block_integrate_from_global_hash_;

      // get grid size
      const uint64_t size_h_grid = sizeof(std::vector<ChunkDesc<T>*>) * grid_.size() *
                                   (sizeof(std::vector<SDFBlock<T>>) + initial_chunk_list_size_ * sizeof(SDFBlock<T>) +
                                    sizeof(std::vector<SDFBlockDesc>) + initial_chunk_list_size_ * sizeof(SDFBlockDesc));

      std::ofstream out_file(memory_allocation_filepath_);
      if (!out_file.is_open()) {
        std::cerr << "Streamer::calculateMemoryUsage | Failed to open file for writing voxel memory usage." << std::endl;
        return;
      }
      out_file << "Streamer | size_h_SDFBlockDescOutput : " << size_h_SDFBlockDescOutput << std::endl;
      out_file << "Streamer | size_h_SDFBlockOutput : " << size_h_SDFBlockOutput << std::endl;
      out_file << "Streamer | size_h_grid : " << size_h_grid << std::endl;
      const uint64_t tot_size_host = size_h_SDFBlockDescOutput + size_h_SDFBlockOutput + size_h_grid;
      out_file << "Streamer | total h_size: " << tot_size_host << " B || " << (double) tot_size_host * toMB << " MiB"
               << std::endl;
      out_file << "=========================================================" << std::endl;

      // device buffers
      const uint64_t size_d_SDFBlockDescOutput = sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_;
      const uint64_t size_d_SDFBlockOutput     = sizeof(T) * total_sdf_block_size * max_num_sdf_block_integrate_from_global_hash_;
      const uint64_t size_d_SDFBlockDescInput  = sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_;
      const uint64_t size_d_SDFBlockInput      = sizeof(T) * total_sdf_block_size * max_num_sdf_block_integrate_from_global_hash_;

      out_file << "Streamer | size_d_SDFBlockDescOutput : " << size_d_SDFBlockDescOutput << std::endl;
      out_file << "Streamer | size_d_SDFBlockOutput : " << size_d_SDFBlockOutput << std::endl;
      out_file << "Streamer | size_d_SDFBlockDescInput : " << size_d_SDFBlockDescInput << std::endl;
      out_file << "Streamer | size_d_SDFBlockInput : " << size_d_SDFBlockInput << std::endl;

      const uint64_t tot_size_device =
        size_d_SDFBlockDescOutput + size_d_SDFBlockOutput + size_d_SDFBlockDescInput + size_d_SDFBlockInput;
      out_file << "Streamer | total d_size: " << tot_size_device << " B || " << (double) tot_size_device * toMB << " MiB"
               << std::endl;
      out_file << "=========================================================" << std::endl;
    }

  } // namespace cugeoutils
} // namespace cupanutils
