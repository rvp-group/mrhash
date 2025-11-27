#include "streamer.cuh"

namespace cupanutils {
  namespace cugeoutils {

    ///////////////////////////////////////////////////////////////////////////
    // streaming from device to host: copies an entire useless stuff in host //
    ///////////////////////////////////////////////////////////////////////////

    template <typename T>
    __global__ void integrateFromGlobalHashPass1Kernel(const float radius,
                                                       const float3 camera_position,
                                                       const int start_idx,
                                                       uint* d_outputCounter,
                                                       SDFBlockDesc* d_output,
                                                       VoxelContainer<T>* container) {
      const uint bucket_id = start_idx + blockIdx.x * blockDim.x + threadIdx.x;

      if (bucket_id >= container->total_size_)
        return;

      HashEntry& entry = container->d_hashTable_[bucket_id];

      float3 pw = SDFBlockToWorldPoint(container->virtual_voxel_size_, entry.pos);
      float d   = length(pw, camera_position);

      // if distance between point and camera is big enough we stream
      if (entry.ptr != FREE_ENTRY && d >= radius) {
        SDFBlockDesc desc(entry);

#ifndef RESOLVE_COLLISION
        uint addr      = atomicAdd(&d_outputCounter[0], 1);
        d_output[addr] = desc;

        if (entry.resolution == 0)
          container->appendHeapHigh(entry.ptr / container->getNumVoxels(entry));
        if (entry.resolution == 1)
          container->appendHeapLow(entry.ptr / container->getNumVoxels(entry));
        deleteHashEntry(bucket_id);
#endif
#ifdef RESOLVE_COLLISION
        // if there is an offset or hash doesn't belong to the bucket (linked list)
        if (entry.offset != 0 || container->calculateHash(entry.pos) != bucket_id / container->hash_bucket_size_) {
          if (container->deleteHashEntryElement(entry.pos)) {
            uint addr      = atomicAdd(&d_outputCounter[0], 1);
            d_output[addr] = desc;
          }
        } else {
          uint addr      = atomicAdd(&d_outputCounter[0], 1);
          d_output[addr] = desc;
          if (entry.resolution == 0)
            container->appendHeapHigh(entry.ptr / container->getNumVoxels(entry));
          else if (entry.resolution == 1)
            container->appendHeapLow(entry.ptr / container->getNumVoxels(entry));
          deleteHashEntry(entry);
        }
#endif
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateFromGlobalHashPass1(const float radius,
                                                                                                 const float3& camera_position) {
      const dim3 threads_per_block((n_threads * n_threads), 1);
      int stream_size = container_->total_size_;
      const dim3 n_blocks(stream_size / threads_per_block.x, 1);
      int start_idx = 0;

      if (stream_size > 0) {
        integrateFromGlobalHashPass1Kernel<<<n_blocks, threads_per_block>>>(
          radius, camera_position, start_idx, d_SDF_block_counter_, d_SDFBlockDescOutput_, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __global__ void integrateFromGlobalHashPass1Kernel(const int start_idx,
                                                       uint* d_outputCounter,
                                                       SDFBlockDesc* d_output,
                                                       VoxelContainer<T>* container) {
      const uint bucket_id = start_idx + blockIdx.x * blockDim.x + threadIdx.x;

      if (bucket_id >= container->total_size_)
        return;

      HashEntry& entry = container->d_hashTable_[bucket_id];

      // if distance between point and camera is big enough we stream
      if (entry.ptr != FREE_ENTRY) {
        SDFBlockDesc desc(entry);

#ifndef RESOLVE_COLLISION
        uint addr      = atomicAdd(&d_outputCounter[0], 1);
        d_output[addr] = desc;

        if (entry.resolution == 0)
          container->appendHeapHigh(entry.ptr / container->getNumVoxels(entry));
        if (entry.resolution == 1)
          container->appendHeapLow(entry.ptr / container->getNumVoxels(entry));
        deleteHashEntry(bucket_id);
#endif
#ifdef RESOLVE_COLLISION
        // if there is an offset or hash doesn't belong to the bucket (linked list)
        if (entry.offset != 0 || container->calculateHash(entry.pos) != bucket_id / container->hash_bucket_size_) {
          if (container->deleteHashEntryElement(entry.pos)) {
            if (entry.resolution == 0)
              container->appendHeapHigh(entry.ptr / container->getNumVoxels(entry));
            else if (entry.resolution == 1)
              container->appendHeapLow(entry.ptr / container->getNumVoxels(entry));
            uint addr      = atomicAdd(&d_outputCounter[0], 1);
            d_output[addr] = desc;
          }
        } else {
          uint addr      = atomicAdd(&d_outputCounter[0], 1);
          d_output[addr] = desc;
          if (entry.resolution == 0)
            container->appendHeapHigh(entry.ptr / container->getNumVoxels(entry));
          else if (entry.resolution == 1)
            container->appendHeapLow(entry.ptr / container->getNumVoxels(entry));
          deleteHashEntry(entry);
        }
#endif
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateFromGlobalHashPass1(const int num_pass) {
      const dim3 threads_per_block((n_threads * n_threads), 1);
      int stream_size = max_num_sdf_block_integrate_from_global_hash_;
      const dim3 n_blocks((stream_size + threads_per_block.x - 1) / threads_per_block.x, 1);
      int start_idx = num_pass * max_num_sdf_block_integrate_from_global_hash_;

      if (stream_size > 0) {
        integrateFromGlobalHashPass1Kernel<<<n_blocks, threads_per_block>>>(
          start_idx, d_SDF_block_counter_, d_SDFBlockDescOutput_, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __global__ void prepareIntegrateFromGlobalHashPass2Kernel(uint start_idx,
                                                              const uint num_SDF_block_desc,
                                                              const SDFBlockDesc* d_SDFBlockDescs,
                                                              uint* num_voxels_vec,
                                                              VoxelContainer<T>* container) {
      const uint idx_block = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
      if (idx_block >= num_SDF_block_desc)
        return;

      const SDFBlockDesc& sdf_block_desc = d_SDFBlockDescs[idx_block];
      const int scale                    = 1 << (finest_block_log2_dim - sdf_block_desc.resolution);
      const int num_voxels               = scale * scale * scale;

      num_voxels_vec[start_idx + idx_block] = (uint) num_voxels;
    }

    template <typename T>
    __global__ void integrateFromGlobalHashPass2Kernel(const int start_idx,
                                                       const uint num_SDF_block_desc,
                                                       const SDFBlockDesc* d_SDFBlockDescs,
                                                       const uint* num_voxels_vec,
                                                       T* d_output,
                                                       VoxelContainer<T>* container) {
      const uint idx_block = start_idx + blockIdx.x;

      if (idx_block >= num_SDF_block_desc)
        return;

      const SDFBlockDesc& desc = d_SDFBlockDescs[idx_block];
      const int scale          = 1 << (finest_block_log2_dim - desc.resolution);
      const int num_voxels     = scale * scale * scale;

      if (threadIdx.x >= num_voxels)
        return;

      uint start_idx_local = 0;
      for (uint i = start_idx; i < blockIdx.x; i++) {
        start_idx_local += num_voxels_vec[i];
      }

      // Copy SDF block to CPU
      d_output[start_idx + start_idx_local + threadIdx.x] = container->d_SDFBlocks_[desc.ptr + threadIdx.x];
      const auto& voxel                                   = d_output[start_idx + start_idx_local + threadIdx.x];

      // reset SDF block
      deleteVoxel<T>(container->d_SDFBlocks_[desc.ptr + threadIdx.x]);
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateFromGlobalHashPass2(const uint num_SDF_block_desc) {
      const dim3 n_blocks_prepare((num_SDF_block_desc + n_threads - 1) / n_threads, 1);
      const dim3 threads_per_block(container_->voxel_block_volume_, 1);
      int stream_size = container_->total_size_;
      const dim3 n_blocks(stream_size, 1);
      int start_idx = 0;

      if (stream_size > 0) {
        uint* num_voxels_vec;
        CUDA_CHECK(cudaMalloc((void**) &num_voxels_vec, sizeof(uint) * num_SDF_block_desc));
        prepareIntegrateFromGlobalHashPass2Kernel<<<n_blocks_prepare, n_threads>>>(
          start_idx, num_SDF_block_desc, d_SDFBlockDescOutput_, num_voxels_vec, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
        integrateFromGlobalHashPass2Kernel<<<n_blocks, threads_per_block>>>(
          start_idx, num_SDF_block_desc, d_SDFBlockDescOutput_, num_voxels_vec, (T*) d_SDFBlockOutput_, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(num_voxels_vec));
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateFromGlobalHashPass2(const uint num_SDF_block_desc,
                                                                                                 const int num_pass) {
      const dim3 n_blocks_prepare((num_SDF_block_desc + n_threads - 1) / n_threads, 1);
      const dim3 threads_per_block(container_->voxel_block_volume_, 1);
      int stream_size = num_SDF_block_desc;
      const dim3 n_blocks(stream_size, 1);
      int start_idx = 0;

      if (stream_size > 0) {
        uint* num_voxels_vec;
        CUDA_CHECK(cudaMalloc((void**) &num_voxels_vec, sizeof(uint) * num_SDF_block_desc));
        prepareIntegrateFromGlobalHashPass2Kernel<<<n_blocks_prepare, threads_per_block>>>(
          start_idx, num_SDF_block_desc, d_SDFBlockDescOutput_, num_voxels_vec, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
        integrateFromGlobalHashPass2Kernel<<<n_blocks, threads_per_block>>>(
          start_idx, num_SDF_block_desc, d_SDFBlockDescOutput_, num_voxels_vec, (T*) d_SDFBlockOutput_, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(num_voxels_vec));
      }
    }

    //////////////////////////////////////////////////////////////////////////
    // streaming from host to device: copies an entire chunk back to device //
    //////////////////////////////////////////////////////////////////////////

    template <typename T>
    __global__ void
    prepareChunkToGlobalHashPass1Kernel(const SDFBlockDesc* d_SDFBlockDescs, uint* num_blocks_vec, VoxelContainer<T>* container) {
      const SDFBlockDesc& sdf_block_desc = d_SDFBlockDescs[blockIdx.x];
      const int scale                    = 1 << (finest_block_log2_dim - sdf_block_desc.resolution);
      const int num_blocks               = scale * scale * scale;
      num_blocks_vec[blockIdx.x]         = (uint) num_blocks;
    }

    //-------------------------------------------------------
    // Pass 1: Allocate memory
    //-------------------------------------------------------

    template <typename T>
    __global__ void chunkToGlobalHashPass1Kernel(const uint num_sdf_blocks_descs,
                                                 const uint heap_count_prev,
                                                 const SDFBlockDesc* d_SDFBlockDescs,
                                                 uint* d_blocks_ptr,
                                                 VoxelContainer<T>* container) {
      const unsigned int bucket_id = blockIdx.x * blockDim.x + threadIdx.x;

      if (bucket_id >= num_sdf_blocks_descs)
        return;

      HashEntry entry;
      entry.pos            = d_SDFBlockDescs[bucket_id].pos;
      entry.offset         = 0;
      entry.resolution     = d_SDFBlockDescs[bucket_id].resolution;
      const int scale      = 1 << (finest_block_log2_dim - entry.resolution);
      const int num_voxels = scale * scale * scale;
      int ptr              = -1;
      if (entry.resolution == 0)
        ptr = container->consumeHeapHigh() * num_voxels;
      else if (entry.resolution == 1)
        ptr = container->consumeHeapLow() * num_voxels;
      d_blocks_ptr[bucket_id] = ptr;
      entry.ptr               = ptr;

      // next kernel will randomly fill memory)
      bool is_inserted = container->insertHashEntry(entry);
      if (!is_inserted && entry.pos.x != 0 && entry.pos.y != 0 && entry.pos.z != 0)
        printf("WARNING entry [ %d %d %d ] not inserted, possible memory leak!\n", entry.pos.x, entry.pos.y, entry.pos.z);
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::chunkToGlobalHashPass1(const uint num_sdf_blocks_descs,
                                                                                           const uint heap_count_prev,
                                                                                           const SDFBlockDesc* d_SDFBlockDescs,
                                                                                           uint* d_blocks_ptr) {
      const dim3 threads_per_block((n_threads * n_threads), 1);
      const dim3 n_blocks((num_sdf_blocks_descs + threads_per_block.x - 1) / threads_per_block.x, 1);

      if (num_sdf_blocks_descs > 0) {
        chunkToGlobalHashPass1Kernel<<<n_blocks, threads_per_block>>>(
          num_sdf_blocks_descs, heap_count_prev, d_SDFBlockDescs, d_blocks_ptr, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    //-------------------------------------------------------
    // Pass 2: Copy input to SDFBlocks
    //-------------------------------------------------------

    template <typename T>
    __global__ void
    prepareChunkToGlobalHashPass2Kernel(const SDFBlockDesc* d_SDFBlockDescs, uint* num_voxels_vec, VoxelContainer<T>* container) {
      const SDFBlockDesc& sdf_block_desc = d_SDFBlockDescs[blockIdx.x];
      const int scale                    = 1 << (finest_block_log2_dim - sdf_block_desc.resolution);
      const int num_voxels               = scale * scale * scale;
      num_voxels_vec[blockIdx.x]         = (uint) num_voxels;
    }

    template <typename T>
    __global__ void chunkToGlobalHashPass2Kernel(const uint heap_count_prev,
                                                 const SDFBlockDesc* d_SDFBlockDescs,
                                                 const uint* num_voxels_vec,
                                                 const T* d_SDFBlocks,
                                                 const uint* d_blocks_ptr,
                                                 VoxelContainer<T>* container) {
      const SDFBlockDesc& sdf_block_desc = d_SDFBlockDescs[blockIdx.x];
      const int scale                    = 1 << (finest_block_log2_dim - sdf_block_desc.resolution);
      const int num_voxels               = scale * scale * scale;

      if (threadIdx.x >= num_voxels)
        return;

      uint start_idx = 0;
      for (uint i = 0; i < blockIdx.x; i++) {
        start_idx += num_voxels_vec[i];
      }
      const uint ptr                             = d_blocks_ptr[blockIdx.x];
      container->d_SDFBlocks_[ptr + threadIdx.x] = d_SDFBlocks[start_idx + threadIdx.x];
      const auto& voxel                          = container->d_SDFBlocks_[ptr + threadIdx.x];
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::chunkToGlobalHashPass2(const uint num_sdf_blocks_descs,
                                                                                           const uint heap_count_prev,
                                                                                           const SDFBlockDesc* d_SDFBlockDescs,
                                                                                           const T* d_SDFBlocks,
                                                                                           uint* d_heap_ptr) {
      const dim3 n_blocks(num_sdf_blocks_descs, 1);
      const dim3 threads_per_block(container_->voxel_block_volume_, 1);

      if (num_sdf_blocks_descs > 0) {
        uint* num_voxels_vec;
        CUDA_CHECK(cudaMalloc((void**) &num_voxels_vec, sizeof(uint) * num_sdf_blocks_descs));
        prepareChunkToGlobalHashPass2Kernel<<<n_blocks, 1>>>(d_SDFBlockDescs, num_voxels_vec, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
        chunkToGlobalHashPass2Kernel<<<n_blocks, threads_per_block>>>(
          heap_count_prev, d_SDFBlockDescs, num_voxels_vec, d_SDFBlocks, d_heap_ptr, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(num_voxels_vec));
      }
    }

  } // namespace cugeoutils
} // namespace cupanutils
