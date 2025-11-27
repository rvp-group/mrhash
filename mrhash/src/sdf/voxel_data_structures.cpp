#include "voxel_data_structures.cuh"
#include <fstream>

namespace cupanutils {
  namespace cugeoutils {

    template <typename T>

    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::calculateMemoryUsage() {
      double toMB = 1e-6;

      std::ofstream out_file(memory_allocation_filepath_);
      if (!out_file.is_open()) {
        std::cerr << "VoxelContainer::calculateMemoryUsage | Failed to open file for writing voxel memory usage." << std::endl;
        return;
      }

      out_file << "VoxelContainer | running with following parameters:";
      out_file << "\nnum_sdf_blocks: " << num_sdf_blocks_ << "\nhash_num_buckets: " << hash_num_buckets_
               << "\nhash_bucket_size: " << hash_bucket_size_ << "\nmax_integration_distance: " << max_integration_distance_
               << "\nsdf_truncation: " << sdf_truncation_ << "\nsdf_truncation_scale: " << sdf_truncation_scale_
               << "\nlinked_list_size: " << linked_list_size_
               << "\nintegration_weight_sample: " << static_cast<unsigned>(integration_weight_sample_)
               << "\nintegration_weight_max: " << static_cast<unsigned>(integration_weight_max_)
               << "\ntotal_size: " << total_size_ << "\nvoxel_block_volume: " << voxel_block_volume_ << std::endl;

      out_file << "=========================================================" << std::endl;

      const int scaling_factor               = 64;
      const uint size_d_heap                 = sizeof(uint) * num_sdf_blocks_ * scaling_factor * 2;
      const uint size_d_hashDecision         = sizeof(int) * total_size_;
      const uint size_d_hashTableBucketMutex = sizeof(int) * hash_num_buckets_;
      const uint64_t size_d_hashTable        = sizeof(HashEntry) * total_size_;
      const uint64_t size_d_compactHashTable = sizeof(HashEntry) * total_size_;
      const uint64_t size_d_SDFBlocks        = sizeof(T) * num_sdf_blocks_ * voxel_block_volume_;

      out_file << "VoxelContainer | structs - size of Voxel: " << sizeof(T) << " | size of HashEntry: " << sizeof(HashEntry)
               << " | size of Triangle: " << sizeof(Triangle) << std::endl;

      out_file << "VoxelContainer | size_d_heap : " << (double) size_d_heap * toMB << " MB" << std::endl;
      out_file << "VoxelContainer | size_d_hashTable : " << (double) size_d_hashTable * toMB << " MB" << std::endl;
      out_file << "VoxelContainer | size_d_compactHashTable : " << (double) size_d_compactHashTable * toMB << " MB" << std::endl;
      out_file << "VoxelContainer | size_d_hashDecision : " << (double) size_d_hashDecision * toMB << " MB" << std::endl;
      out_file << "VoxelContainer | size_d_hashTableBucketMutex : " << (double) size_d_hashTableBucketMutex * toMB << " MB"
               << std::endl;
      out_file << "VoxelContainer | size_d_SDFBlocks : " << (double) size_d_SDFBlocks * toMB << " MB" << std::endl;

      const uint64_t tot_size_device = size_d_heap + size_d_hashTable + size_d_compactHashTable + size_d_hashDecision;
      out_file << "VoxelContainer | total d_size: " << tot_size_device << " B || " << (double) tot_size_device * toMB << " MB"
               << std::endl;

      out_file << "=========================================================" << std::endl;

      out_file.close(); // explicitly close the file (optional, done automatically on destruction)
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::resetBuffers() {
      int scaling_factor = 8;
      uint* h_heap_high  = new uint[num_sdf_blocks_];
      uint* h_heap_low   = new uint[num_sdf_blocks_ * scaling_factor];
      for (uint i = 0; i < num_sdf_blocks_; i++) {
        for (uint j = 0; j < scaling_factor; j++) {
          h_heap_low[i * scaling_factor + j] = num_sdf_blocks_ * scaling_factor; // initialize to an invalid value
        }
        h_heap_high[i] = num_sdf_blocks_ - 1 - i;
      }
      CUDA_CHECK(cudaMemcpy(d_heap_high_, h_heap_high, sizeof(uint) * num_sdf_blocks_, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_heap_low_, h_heap_low, sizeof(uint) * num_sdf_blocks_ * scaling_factor, cudaMemcpyHostToDevice));
      delete[] h_heap_high;
      delete[] h_heap_low;
      T* h_SDFBlocks = new T[num_sdf_blocks_ * voxel_block_volume_];
      CUDA_CHECK(
        cudaMemcpy(d_SDFBlocks_, h_SDFBlocks, sizeof(T) * num_sdf_blocks_ * voxel_block_volume_, cudaMemcpyHostToDevice));
      delete[] h_SDFBlocks;
      HashEntry* h_hashTable = new HashEntry[total_size_];
      CUDA_CHECK(cudaMemcpy(d_hashTable_, h_hashTable, sizeof(HashEntry) * total_size_, cudaMemcpyHostToDevice));
      delete[] h_hashTable;
      HashEntry* h_compactHashTable = new HashEntry[total_size_];
      CUDA_CHECK(cudaMemcpy(d_compactHashTable_, h_compactHashTable, sizeof(HashEntry) * total_size_, cudaMemcpyHostToDevice));
      delete[] h_compactHashTable;
      int* h_hashTableBucketMutex = new int[hash_num_buckets_];
      std::fill(h_hashTableBucketMutex, h_hashTableBucketMutex + hash_num_buckets_, FREE_ENTRY);
      CUDA_CHECK(
        cudaMemcpy(d_hashTableBucketMutex_, h_hashTableBucketMutex, sizeof(int) * hash_num_buckets_, cudaMemcpyHostToDevice));
      delete[] h_hashTableBucketMutex;
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrate(const CUDAMatrixf3& point_cloud_img,
                                                                                    const CUDAMatrixuc3& rgb_img,
                                                                                    const Camera& camera,
                                                                                    const int max_num_frames) {
      {
        CUDAProfiler::CUDAEvent event(integration_profiler_);
        allocBlocks(point_cloud_img, camera);
        flatAndReduceHashTable(camera);
        integrateDepthMap(point_cloud_img, rgb_img, camera);
        if (sdf_var_threshold_ > 0.f && num_integrated_frames_ > 0) {
          checkVarSDF();
          reallocBlocks();
          flatAndReduceHashTable(camera);
          reintegrateDepthMap(point_cloud_img, rgb_img, camera);
        }
        if (max_num_frames > 0)
          garbageCollect(camera, max_num_frames);
        num_integrated_frames_++;
      }
      integration_profiler_.write(current_occupied_blocks_);
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrate(const CUDAVectorf3& point_cloud,
                                                                                    const CUDAVectorf3& normals,
                                                                                    const CUDAVectorf& weights,
                                                                                    const Camera& camera,
                                                                                    const int max_num_frames) {
      {
        CUDAProfiler::CUDAEvent event(integration_profiler_);
        allocBlocks3D(point_cloud, normals, weights, camera);
        flatAndReduceHashTable();
        integrate3D(point_cloud, normals, weights, camera);
        if (sdf_var_threshold_ > 0.f && num_integrated_frames_ > 0) {
          checkVarSDF();
          reallocBlocks();
          flatAndReduceHashTable();
          reintegrate3D(point_cloud, normals, weights, camera);
        }
        if (max_num_frames > 0)
          garbageCollect(camera, max_num_frames);
        num_integrated_frames_++;
      }
      integration_profiler_.write(current_occupied_blocks_);
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::garbageCollect(const Camera& camera,
                                                                                         const int max_num_frames) {
      if (num_integrated_frames_ > 0 && num_integrated_frames_ % max_num_frames == 0) {
        starveVoxels(camera);
      }
      garbageCollectIdentify(camera);
      resetHashBucketMutex();
      garbageCollectFree();
    }

    template <typename T>
    int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getHeapHighFreeCount() {
      uint count;
      CUDA_CHECK(cudaMemcpy(&count, d_heapCounterHigh_, sizeof(uint), cudaMemcpyDeviceToHost));
      return count + 1; // there is one more free than the address suggests (0
                        // would be also a valid address)
    }

    template <typename T>
    int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getHeapLowFreeCount() {
      int count;
      CUDA_CHECK(cudaMemcpy(&count, d_heapCounterLow_, sizeof(int), cudaMemcpyDeviceToHost));
      return count + 1; // there is one more free than the address suggests (0
                        // would be also a valid address)
    }

  } // namespace cugeoutils
} // namespace cupanutils
