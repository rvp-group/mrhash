#include "voxel_data_structures.cuh"

#include <cfloat>

namespace cupanutils {
  namespace cugeoutils {

    template <typename T>
    __global__ void resetCompactHashTableKernel(const VoxelContainer<T>* container) {
      const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= container->total_size_)
        return;
      deleteHashEntry(container->d_compactHashTable_[idx]);
    }

    template <typename T>
    __global__ void resetHashBucketMutexKernel(const VoxelContainer<T>* container) {
      const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= container->hash_num_buckets_)
        return;
      container->d_hashTableBucketMutex_[idx] = FREE_ENTRY;
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::resetHashBucketMutex() {
      const dim3 threads_per_block((n_threads * n_threads), 1);
      const dim3 n_blocks((hash_num_buckets_ + threads_per_block.x - 1) / threads_per_block.x, 1);
      resetHashBucketMutexKernel<<<n_blocks, threads_per_block>>>(d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::consumeHeapHigh() {
      int addr = atomicSub(&d_heapCounterHigh_[0], 1);

      if (addr < 0)
        return -1;

      return d_heap_high_[addr];
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::consumeHeapLow() {
      int addr = atomicSub(&d_heapCounterLow_[0], 1);

      if (addr < 0)
        return -1;

      return d_heap_low_[addr];
    }

    template <typename T>
    __device__ void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::appendHeapHigh(const uint ptr) {
      int addr               = atomicAdd(&d_heapCounterHigh_[0], 1);
      d_heap_high_[addr + 1] = ptr;
    }

    template <typename T>
    __device__ void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::appendHeapLow(const uint ptr) {
      int addr              = atomicAdd(&d_heapCounterLow_[0], 1);
      d_heap_low_[addr + 1] = ptr;
    }

    template <typename T>
    // ! fast-approx check to see if sdf block is in camera frustum
    __device__ bool
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::isSDFBlockInCameraFrustumApprox(const Camera* camera,
                                                                                                     const int3& sdf_block) {
      for (int i = 0; i < vertex_offset_camera; i++) {
        int3 vertex            = vert_offset[i];
        int3 virtual_voxel_pos = SDFBlockToVirtualVoxelPos(sdf_block) + vertex;
        float3 world_point     = virtualVoxelPosToWorld(virtual_voxel_size_, virtual_voxel_pos);
        if (camera->isInCameraFrustumApprox(world_point))
          return true;
      }
      return false;
    }

    template <typename T>
    __device__ HashEntry
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getHashEntry(const int3& sdf_block) const {
      HashEntry entry;
      entry.pos    = sdf_block;
      entry.ptr    = FREE_ENTRY;
      entry.offset = 0;

      uint64_t h = calculateHash(sdf_block);

      for (uint i = 0; i < hash_bucket_size_; ++i) {
        const uint hash_idx = h * hash_bucket_size_ + i;
        HashEntry curr      = d_hashTable_[hash_idx];
        if (curr.pos.x == sdf_block.x && curr.pos.y == sdf_block.y && curr.pos.z == sdf_block.z && curr.ptr != FREE_ENTRY) {
          return curr;
        }
      }

#ifdef RESOLVE_COLLISION
      const uint idx_last_entry_in_bucket = (h + 1) * hash_bucket_size_ - 1; // get last index of bucket
      uint i                              = idx_last_entry_in_bucket;        // start with the last entry of the current bucket

      HashEntry curr;
      curr.offset = 0;

      int max_iter            = 0;
      int max_loop_iter_count = linked_list_size_;
#pragma unroll 1
      while (max_iter < max_loop_iter_count) { // traverse list until end

        curr = d_hashTable_[i];

        if (curr.pos.x == sdf_block.x && curr.pos.y == sdf_block.y && curr.pos.z == sdf_block.z && curr.ptr != FREE_ENTRY) {
          return curr;
        }

        if (curr.offset == 0) {
          break;
        }

        i = idx_last_entry_in_bucket + curr.offset;
        i %= (hash_bucket_size_ * hash_num_buckets_);

        max_iter++;
      }
#endif

      return entry;
    }

    template <typename T>
    __device__ HashEntry
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getHashEntryReintegrate(const int3& sdf_block) const {
      HashEntry entry;
      entry.pos    = sdf_block;
      entry.ptr    = FREE_ENTRY;
      entry.offset = 0;

      uint64_t h = calculateHash(sdf_block);

      for (uint i = 0; i < d_num_reintegrate_[0]; ++i) {
        const uint hash_idx = d_reintegrate_[i];
        HashEntry curr      = d_hashTable_[hash_idx];
        if (curr.pos.x == sdf_block.x && curr.pos.y == sdf_block.y && curr.pos.z == sdf_block.z && curr.ptr != FREE_ENTRY) {
          return curr;
        }
      }

      return entry;
    }

    template <typename T>
    __device__ uint64_t
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::calculateHash(const int3& virtual_voxel_pos) const {
      unsigned int x = (unsigned int) virtual_voxel_pos.x;
      unsigned int y = (unsigned int) virtual_voxel_pos.y;
      unsigned int z = (unsigned int) virtual_voxel_pos.z;
      int res        = ((x * (unsigned int) p0) ^ (y * (unsigned int) p1) ^ (z * (unsigned int) p2)) % hash_num_buckets_;
      if (res < 0)
        res += hash_num_buckets_;
      return res;
    }

    template <typename T>
    __device__ T VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxel(const int3& virtual_voxel_pos) const {
      T v;
      const HashEntry& entry = getHashEntry(virtualVoxelPosToSDFBlock(virtual_voxel_pos, virtual_voxel_size_, voxel_extents_));
      if (entry.ptr == FREE_ENTRY) {
        deleteVoxel<T>(v);
        return v;
      } else {
        const int scaling_factor = 1 << entry.resolution;

        v = d_SDFBlocks_[entry.ptr + virtualVoxelPosToSDFBlockIndex(virtual_voxel_pos, sdf_block_size / scaling_factor)];
        return v;
      }
    }
    template <typename T>
    __device__ T VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxel(const int3& virtual_voxel_pos,
                                                                                           int& block_res) const {
      T v;
      const HashEntry& entry = getHashEntry(virtualVoxelPosToSDFBlock(virtual_voxel_pos, virtual_voxel_size_, voxel_extents_));
      if (entry.ptr == FREE_ENTRY) {
        deleteVoxel<T>(v);
        return v;
      } else {
        const int scaling_factor = 1 << entry.resolution;
        block_res                = entry.resolution;

        uint voxel_index                   = virtualVoxelPosToSDFBlockIndex(virtual_voxel_pos, sdf_block_size / scaling_factor);
        v                                  = d_SDFBlocks_[entry.ptr + voxel_index];
        uint3 delinearized_local_voxel_pos = delinearizeVoxelPos(voxel_index, sdf_block_size / scaling_factor);
        int3 delinearized_voxel_pos        = SDFBlockToVirtualVoxelPos(entry.pos) + scaling_factor * delinearized_local_voxel_pos;
        return v;
      }
    }

    template <typename T>
    __device__ T VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxel(const float3& pos) const {
      return getVoxel(worldPointToVirtualVoxelPos(virtual_voxel_size_, pos));
    }

    template <typename T>
    __device__ T VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxel(const float3& pos,
                                                                                           int& block_res) const {
      return getVoxel(worldPointToVirtualVoxelPos(virtual_voxel_size_, pos), block_res);
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getNumVoxels(const HashEntry& entry) const {
      const int scale = 1 << (finest_block_log2_dim - entry.resolution);
      return scale * scale * scale;
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getNumVoxels(const int3& pos) const {
      const HashEntry& entry = getHashEntry(pos);
      return getNumVoxels(entry);
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getNumVoxels(const float3& pos) const {
      const HashEntry& entry = getHashEntry(worldPointToSDFBlock(virtual_voxel_size_, voxel_extents_, pos));
      return getNumVoxels(entry);
    }

    template <typename T>
    __device__ float VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxelSize(const HashEntry& entry) const {
      return virtual_voxel_size_ * (1 << entry.resolution);
    }

    template <typename T>
    __device__ float VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxelSize(const int3& pos) const {
      const HashEntry& entry = getHashEntry(pos);
      return getVoxelSize(entry);
    }

    template <typename T>
    __device__ float VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxelSize(const float3& pos) const {
      const HashEntry& entry = getHashEntry(worldPointToSDFBlock(virtual_voxel_size_, voxel_extents_, pos));
      return getVoxelSize(entry);
    }

    template <typename T>
    __global__ void getVoxelWeightKernel(const float3 pw, uchar* weight, VoxelContainer<T>* container) {
      const T& voxel = container->getVoxel(pw);
      if (voxel.weight > 0) {
        weight[0] = voxel.weight;
      }
    }

    template <typename T>
    uchar VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxelWeight(const Eigen::Vector3f& pw) {
      getVoxelWeightKernel<<<1, 1>>>(Eig2CUDA(pw), d_weight_, d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());
      uchar weight = 0;
      CUDA_CHECK(cudaMemcpy(&weight, &d_weight_[0], sizeof(uchar), cudaMemcpyDeviceToHost));
      return weight;
    }

    template <typename T>
    __device__ bool VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::trilinearInterpolation(const float3& pos,
                                                                                                            float& dist) const {
      float voxel_size       = getVoxelSize(pos);
      const float3 pos_dual  = pos - make_float3(voxel_size * 0.5f);
      const HashEntry& entry = getHashEntry(worldPointToSDFBlock(voxel_size, voxel_extents_, pos));

      const int base_resolution = entry.resolution;
      dist                      = 0.f;
      const float pos_sdf       = getVoxel(pos_dual).sdf;

      const float x0 = pos_dual.x;
      const float y0 = pos_dual.y;
      const float z0 = pos_dual.z;
      float x1       = x0;
      float y1       = y0;
      float z1       = z0;

      int resolution = 0;

      float sdf[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

      for (int i = 0; i < 8; ++i) {
        int dx = i & 1;
        int dy = (i >> 1) & 1;
        int dz = (i >> 2) & 1;

        const float3 voxel_pos = pos_dual + make_float3(dx, dy, dz) * voxel_size;

        const T& v = getVoxel(voxel_pos, resolution);

        if (!v.weight)
          return false;

        if (resolution > base_resolution) {
          const float new_voxel_size = voxel_size * 2;

          const float3 new_voxel_pos = pos - make_float3(new_voxel_size * 0.5f) + make_float3(dx, dy, dz) * new_voxel_size;
          // interpolate sdf at voxel_pos between pos_dual and new_voxel_pos based on their coordinates
          const float new_voxel_pos_sdf = getVoxel(new_voxel_pos).sdf;
          float alpha                   = 0.5f;
          const float dist_inside       = (1 - alpha) * pos_sdf + alpha * new_voxel_pos_sdf;

          sdf[i] = dist_inside;
        }

        else {
          sdf[i] = v.sdf;
        }

        if (voxel_pos.x > x1)
          x1 = voxel_pos.x;
        if (voxel_pos.y > y1)
          y1 = voxel_pos.y;
        if (voxel_pos.z > z1)
          z1 = voxel_pos.z;

        resolution = 0;
      }

      // Avoid division by zero in delta calculation
      const float dx = (x1 - x0) > 1e-6f ? (pos.x - x0) / (x1 - x0) : 0.5f;
      const float dy = (y1 - y0) > 1e-6f ? (pos.y - y0) / (y1 - y0) : 0.5f;
      const float dz = (z1 - z0) > 1e-6f ? (pos.z - z0) / (z1 - z0) : 0.5f;
      float3 delta   = {dx, dy, dz};

      float c[8] = {sdf[0],
                    (sdf[1] - sdf[0]),
                    (sdf[2] - sdf[0]),
                    (sdf[4] - sdf[0]),
                    (sdf[3] - sdf[2] - sdf[1] + sdf[0]),
                    (sdf[6] - sdf[4] - sdf[2] + sdf[0]),
                    (sdf[5] - sdf[4] - sdf[1] + sdf[0]),
                    (sdf[7] - sdf[6] - sdf[5] - sdf[3] + sdf[1] + sdf[4] + sdf[2] - sdf[0])};

      dist = c[0] + c[1] * delta.x + c[2] * delta.y + c[3] * delta.z + c[4] * delta.x * delta.y + c[5] * delta.y * delta.z +
             c[6] * delta.x * delta.z + c[7] * delta.x * delta.y * delta.z;

      return true;
    }

    template <typename T>
    __device__ float VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::findIntersectionLinear(float t_near,
                                                                                                             float t_far,
                                                                                                             float d_near,
                                                                                                             float d_far) const {
      return t_near + (d_near / (d_near - d_far)) * (t_far - t_near);
    }

    template <typename T>
    __device__ bool
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::findIntersectionBisection(const Camera* camera,
                                                                                               const float3& world_cam_pos,
                                                                                               const float3& world_dir,
                                                                                               float d0,
                                                                                               float r0,
                                                                                               float d1,
                                                                                               float r1,
                                                                                               float& alpha) const {
      float a      = r0;
      float a_dist = d0;
      float b      = r1;
      float b_dist = d1;
      float c      = 0.f;

#pragma unroll 1
      for (uint i = 0; i < n_iteration_bisection; ++i) {
        c = findIntersectionLinear(a, b, a_dist, b_dist);
        float c_dist;
        if (!trilinearInterpolation(world_cam_pos + c * world_dir, c_dist)) {
          return false;
        }
        if (a_dist * c_dist > 0) {
          a      = c;
          a_dist = c_dist;
        } else {
          b      = c;
          b_dist = c_dist;
        }
      }

      alpha = c;

      return true;
    }

    template <typename T>
    __device__ uint8_t VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::checkVoxelNeighbors(const float3& pw) {
      const int3& pi             = worldPointToVirtualVoxelPos(virtual_voxel_size_, pw);
      const int3& block_pos      = virtualVoxelPosToSDFBlock(pi, virtual_voxel_size_, voxel_extents_);
      const int voxel_resolution = getHashEntry(block_pos).resolution;
      const int scaling_factor   = 1 << voxel_resolution;

      uint8_t bit_mask = 0;
      for (int i = 0; i < neighbor_voxels; ++i) {
        const int3& voxel_neighbor_pos = pi + scaling_factor * neighbor_offsets[i];
        const int3& block_neighbor_pos = virtualVoxelPosToSDFBlock(voxel_neighbor_pos, virtual_voxel_size_, voxel_extents_);
        const auto neighbor_entry      = getHashEntry(block_neighbor_pos);

        if (neighbor_entry.ptr != FREE_ENTRY && neighbor_entry.resolution != voxel_resolution) {
          bit_mask |= (1 << i);
        }
      }
      return bit_mask;
    }

    template <typename T>
    __global__ void flatAndReduceHashTableKernel(const Camera* camera, VoxelContainer<T>* container) {
      const int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx >= container->total_size_)
        return;

      __shared__ int local_counter;
      if (threadIdx.x == 0)
        local_counter = 0;
      __syncthreads();

      int local_addr         = -1;
      const HashEntry& entry = container->d_hashTable_[idx];
      if (entry.ptr != FREE_ENTRY && container->isSDFBlockInCameraFrustumApprox(camera, entry.pos)) {
        local_addr = atomicAdd(&local_counter, 1);
      }

      __syncthreads();

      __shared__ int global_addr;
      if (threadIdx.x == 0 && local_counter > 0) {
        global_addr = atomicAdd(&container->d_compactHashCounter_[0], local_counter);
      }
      __syncthreads();

      if (local_addr != -1) {
        const uint addr                      = global_addr + local_addr;
        container->d_compactHashTable_[addr] = entry;
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::flatAndReduceHashTable(const Camera& camera) {
      const dim3 threads_per_block(n_threads_reduce_hashtable, 1);
      const dim3 n_blocks((total_size_ + threads_per_block.x - 1) / threads_per_block.x, 1);

      resetCompactHashTableKernel<<<n_blocks, threads_per_block>>>(d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemset(d_compactHashCounter_, 0, sizeof(int)));

      flatAndReduceHashTableKernel<<<n_blocks, threads_per_block>>>(camera.deviceInstance(), d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(&current_occupied_blocks_, &d_compactHashCounter_[0], sizeof(uint), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(VoxelContainer), cudaMemcpyHostToDevice));
    }

    template <typename T>
    __global__ void flatAndReduceHashTableKernel(VoxelContainer<T>* container) {
      const int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx >= container->total_size_)
        return;

      __shared__ int local_counter;
      if (threadIdx.x == 0)
        local_counter = 0;
      __syncthreads();

      // local address within block
      int local_addr         = -1;
      const HashEntry& entry = container->d_hashTable_[idx];
      if (entry.ptr != FREE_ENTRY) {
        local_addr = atomicAdd(&local_counter, 1);
      }

      __syncthreads();

      // update global count of occupied blocks
      __shared__ int global_addr;
      if (threadIdx.x == 0 && local_counter > 0) {
        global_addr = atomicAdd(&container->d_compactHashCounter_[0], local_counter);
      }
      __syncthreads();

      // assign local address and copy
      if (local_addr != -1) {
        const uint addr                      = global_addr + local_addr;
        container->d_compactHashTable_[addr] = entry;
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::flatAndReduceHashTable() {
      const dim3 threads_per_block(n_threads_reduce_hashtable, 1);
      const dim3 n_blocks((total_size_ + threads_per_block.x - 1) / threads_per_block.x, 1);

      resetCompactHashTableKernel<<<n_blocks, threads_per_block>>>(d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemset(d_compactHashCounter_, 0, sizeof(int)));

      flatAndReduceHashTableKernel<<<n_blocks, threads_per_block>>>(d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(&current_occupied_blocks_, &d_compactHashCounter_[0], sizeof(uint), cudaMemcpyDeviceToHost));
      // copy current ptr to gpu, make sure current_occupied_blocks_ is updated in gpu
      CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(VoxelContainer), cudaMemcpyHostToDevice));
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::allocBlock(const int3& pos,
                                                                                               const int resolution) {
      uint h        = calculateHash(pos);    // hash bucket
      const uint hp = h * hash_bucket_size_; // hash position

      int first_empty = -1;
      for (uint j = 0; j < hash_bucket_size_; ++j) {
        uint i                = hp + j;
        const HashEntry& curr = d_hashTable_[i];
        // in that case the SDF-block is already allocated and corresponds to the current position exit thread
        if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
          return -1;
        }

        // store the first FREE_ENTRY hash entry
        if (first_empty == -1 && curr.ptr == FREE_ENTRY) {
          first_empty = i;
        }
      }

#ifdef RESOLVE_COLLISION

      // handling collisions
      // updated variables as after the loop
      const uint idx_last_entry_in_bucket = (h + 1) * hash_bucket_size_ - 1; // get last index of bucket
      uint i                              = idx_last_entry_in_bucket;        // start with the last entry of the current bucket

      HashEntry curr;
      curr.offset = 0;
      // traverse list until end: memorize idx at list end save offset from last element of
      // bucket to list end int k = 0;

      uint max_iter            = 0;
      uint max_loop_iter_count = linked_list_size_;
#pragma unroll 1
      while (max_iter < max_loop_iter_count) {
        curr = d_hashTable_[i];
        if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
          return -1; // Block already allocated
        }
        if (curr.offset == 0) { // we have found the end of the list
          break;
        }
        i = idx_last_entry_in_bucket + curr.offset;   // go to next element in the list
        i %= (hash_bucket_size_ * hash_num_buckets_); // check for overflow

        max_iter++;
      }

#endif

      // if there is an empty entry and we haven't allocated the current entry before
      if (first_empty != -1) {
        int prev_val = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
        if (prev_val != LOCK_ENTRY) { // only proceed if the bucket has been locked
          HashEntry& entry = d_hashTable_[first_empty];
          entry.pos        = pos;
          entry.offset     = NO_OFFSET;
          entry.resolution = resolution;
          int ptr_idx      = -1;
          if (entry.resolution == 0)
            ptr_idx = consumeHeapHigh();
          else if (entry.resolution == 1)
            ptr_idx = consumeHeapLow();
          if (ptr_idx < 0) {
            printf("allocBlock |  mem size exceed, not inserting hash entry!\n");
            return -1;
          }
          const int voxel_block_volume = getNumVoxels(entry);
          entry.ptr                    = ptr_idx * voxel_block_volume;
        }
        return first_empty;
      }

#ifdef RESOLVE_COLLISION
      // handling collisions
      int offset = 0;
      // linear search for free entry
      max_iter = 0;
#pragma unroll 1
      while (max_iter < max_loop_iter_count) {
        offset++;
        // go to next hash element
        i = (idx_last_entry_in_bucket + offset) % (total_size_);
        if ((offset % hash_bucket_size_) == 0)
          continue; // cannot insert into a last bucket element (would conflict with other linked
                    // lists)
        curr = d_hashTable_[i];
        if (curr.ptr == FREE_ENTRY) { // this is the first free entry
          int prev_value = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
          if (prev_value != LOCK_ENTRY) {
            HashEntry last_entry_in_bucket = d_hashTable_[idx_last_entry_in_bucket];
            h                              = i / hash_bucket_size_;
            prev_value                     = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
            if (prev_value != LOCK_ENTRY) { // only proceed if the bucket has been locked
              HashEntry& entry = d_hashTable_[i];
              entry.pos        = pos;
              entry.offset     = last_entry_in_bucket.offset;
              entry.resolution = resolution;
              int ptr_idx      = -1;
              if (entry.resolution == 0)
                ptr_idx = consumeHeapHigh();
              else if (entry.resolution == 1)
                ptr_idx = consumeHeapLow();
              if (ptr_idx < 0) {
                printf("allocBlock |  mem size exceed, not inserting hash entry!\n");
                return -1;
              }
              const int voxel_block_volume = getNumVoxels(entry);
              entry.ptr                    = ptr_idx * voxel_block_volume;

              last_entry_in_bucket.offset            = offset;
              d_hashTable_[idx_last_entry_in_bucket] = last_entry_in_bucket;
            }
          }
          return -1; // bucket was already locked
        }

        max_iter++;
      }
#endif
      return -1;
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::reallocBlock(const int3& pos,
                                                                                                 const int resolution) {
      uint h        = calculateHash(pos);    // hash bucket
      const uint hp = h * hash_bucket_size_; // hash position

      int first_empty = -1;
      for (uint j = 0; j < hash_bucket_size_; ++j) {
        uint i                = hp + j;
        const HashEntry& curr = d_hashTable_[i];
        // in that case the SDF-block is already allocated and corresponds to the current position exit thread
        if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
          return -1;
        }

        // store the first FREE_ENTRY hash entry
        if (first_empty == -1 && curr.ptr == FREE_ENTRY) {
          first_empty = i;
        }
      }

#ifdef RESOLVE_COLLISION

      // handling collisions
      // updated variables as after the loop
      const uint idx_last_entry_in_bucket = (h + 1) * hash_bucket_size_ - 1; // get last index of bucket
      uint i                              = idx_last_entry_in_bucket;        // start with the last entry of the current bucket

      HashEntry curr;
      curr.offset = 0;
      // traverse list until end: memorize idx at list end save offset from last element of
      // bucket to list end int k = 0;

      uint max_iter            = 0;
      uint max_loop_iter_count = linked_list_size_;
#pragma unroll 1
      while (max_iter < max_loop_iter_count) {
        curr = d_hashTable_[i];
        if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
          return -1; // Block already allocated
        }
        if (curr.offset == 0) { // we have found the end of the list
          break;
        }
        i = idx_last_entry_in_bucket + curr.offset;   // go to next element in the list
        i %= (hash_bucket_size_ * hash_num_buckets_); // check for overflow

        max_iter++;
      }

#endif

      // if there is an empty entry and we haven't allocated the current entry before
      if (first_empty != -1) {
        int prev_val = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
        if (prev_val != LOCK_ENTRY) { // only proceed if the bucket has been locked
          HashEntry& entry             = d_hashTable_[first_empty];
          entry.pos                    = pos;
          entry.offset                 = NO_OFFSET;
          entry.resolution             = resolution;
          const int voxel_block_volume = getNumVoxels(entry);
          int ptr_idx                  = -1;
          if (entry.resolution == 0)
            ptr_idx = consumeHeapHigh();
          else if (entry.resolution == 1)
            ptr_idx = consumeHeapLow();
          if (ptr_idx < 0) {
            printf("reallocBlock |  %d mem size exceed (heapCounterHigh: %d | heapCounterLow: %d), not "
                   "inserting hash entry!\n",
                   ptr_idx,
                   *d_heapCounterHigh_,
                   *d_heapCounterLow_);
            return -1;
          }
          entry.ptr = ptr_idx * voxel_block_volume;
          return first_empty;
        } else {
          return LOCK_ENTRY;
        }
      }

#ifdef RESOLVE_COLLISION
      // handling collisions
      int offset = 0;
      // linear search for free entry
      max_iter = 0;
#pragma unroll 1
      while (max_iter < max_loop_iter_count) {
        offset++;
        // go to next hash element
        i = (idx_last_entry_in_bucket + offset) % (total_size_);
        if ((offset % hash_bucket_size_) == 0)
          continue; // cannot insert into a last bucket element (would conflict with other linked
                    // lists)
        curr = d_hashTable_[i];
        if (curr.ptr == FREE_ENTRY) { // this is the first free entry
          int prev_value = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
          if (prev_value != LOCK_ENTRY) {
            HashEntry last_entry_in_bucket = d_hashTable_[idx_last_entry_in_bucket];
            h                              = i / hash_bucket_size_;
            prev_value                     = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
            if (prev_value != LOCK_ENTRY) { // only proceed if the bucket has been locked
              HashEntry& entry             = d_hashTable_[i];
              entry.pos                    = pos;
              entry.offset                 = last_entry_in_bucket.offset;
              entry.resolution             = resolution;
              const int voxel_block_volume = getNumVoxels(entry);
              int ptr_idx                  = -1;
              if (entry.resolution == 0)
                ptr_idx = consumeHeapHigh();
              else if (entry.resolution == 1)
                ptr_idx = consumeHeapLow();
              if (ptr_idx < 0) {
                printf("reallocBlock |  mem size exceed, not inserting hash entry!\n");
                return LOCK_ENTRY;
              }
              entry.ptr = ptr_idx * voxel_block_volume;

              last_entry_in_bucket.offset            = offset;
              d_hashTable_[idx_last_entry_in_bucket] = last_entry_in_bucket;
            }
          }
          return LOCK_ENTRY; // bucket was already locked
        }

        max_iter++;
      }
#endif
      return -1;
    }

    template <typename T>
    __global__ void allocBlocksKernel(const CUDAMatrixf3* point_cloud_img,
                                      const Camera* camera,
                                      const float max_integration_distance,
                                      const float sdf_truncation,
                                      const float sdf_truncation_scale,
                                      VoxelContainer<T>* container) {
      int row = blockDim.y * blockIdx.y + threadIdx.y;
      int col = blockDim.x * blockIdx.x + threadIdx.x;

      if (!point_cloud_img->inside(row, col))
        return;

      const float& depth = camera->getDepth(point_cloud_img->at<1>(row, col));

      if (depth == 0.f)
        return; // set to 0 during pc initialization, if empty

      const float t         = getTruncation(depth, sdf_truncation, sdf_truncation_scale);
      const float min_depth = min(max_integration_distance, depth - t);
      const float max_depth = min(max_integration_distance, depth + t);

      if (min_depth >= max_depth)
        return;

      // clang-format off
    float3 pcam_min = camera->inverseProjection(row, col, min_depth);
    float3 pcam_max = camera->inverseProjection(row, col, max_depth);


    float3 pw_min = camera->camInWorld() * pcam_min;
    float3 pw_max = camera->camInWorld() * pcam_max;
    
    float3 dir = normalize(pw_max - pw_min);

    int3 id_current_voxel = worldPointToSDFBlock(container->virtual_voxel_size_, container->voxel_extents_, pw_min);
    int3 id_end = worldPointToSDFBlock(container->virtual_voxel_size_, container->voxel_extents_, pw_max);

    float3 step = make_float3(sign(dir));

    float3 boundary_pos = SDFBlockToWorldPoint(container->virtual_voxel_size_, id_current_voxel + make_int3(clamp(step, 0.0, 1.f))) - 0.5f * container->virtual_voxel_size_;
    float3 t_max   = (boundary_pos - pw_min) / dir;
    float3 t_delta = (step * sdf_block_size * container->virtual_voxel_size_) / dir;
    int3 id_bound  = make_int3(make_float3(id_end) + step);

    if (fabsf(dir.x) < FLOAT_EPSILON) {
      t_max.x   = numeric_limits<float>::max();
      t_delta.x = numeric_limits<float>::max();
    }
    if (fabsf(boundary_pos.x - dir.x) < FLOAT_EPSILON) {
      t_max.x   = numeric_limits<float>::max();
      t_delta.x = numeric_limits<float>::max();
    }

    if (fabsf(dir.y) < FLOAT_EPSILON) {
      t_max.y   = numeric_limits<float>::max();
      t_delta.y = numeric_limits<float>::max();
    }
    if (fabsf(boundary_pos.y - dir.y) < FLOAT_EPSILON) {
      t_max.y   = numeric_limits<float>::max();
      t_delta.y = numeric_limits<float>::max();
    }

    if (fabsf(dir.z) < FLOAT_EPSILON) {
      t_max.z   = numeric_limits<float>::max();
      t_delta.z = numeric_limits<float>::max();
    }
    if (fabsf(boundary_pos.z - dir.z) < FLOAT_EPSILON) {
      t_max.z   = numeric_limits<float>::max();
      t_delta.z = numeric_limits<float>::max();
    }
      // clang-format on

      uint iter = 0; // iter < max_loop_iter_count
#pragma unroll
      while (iter < max_dda_iteration_count) {
        if (container->isSDFBlockInCameraFrustumApprox(camera, id_current_voxel)) {
          const int is_allocated = container->allocBlock(id_current_voxel);
        }

        // traverse voxel grid
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
          id_current_voxel.x += step.x;
          if (id_current_voxel.x == id_bound.x)
            return;
          t_max.x += t_delta.x;
        } else if (t_max.z < t_max.y) {
          id_current_voxel.z += step.z;
          if (id_current_voxel.z == id_bound.z)
            return;
          t_max.z += t_delta.z;
        } else {
          id_current_voxel.y += step.y;
          if (id_current_voxel.y == id_bound.y)
            return;
          t_max.y += t_delta.y;
        }

        iter++;
      }
    }

    template <typename T>
    __global__ void allocateMemoryLow(VoxelContainer<T>* container) {
      __shared__ int addr_high;
      __shared__ int addr_low;
      if (threadIdx.x == 0) {
        addr_high = atomicSub(&container->d_heapCounterHigh_[0], 1);
        addr_low  = atomicAdd(&container->d_heapCounterLow_[0], octree_branching_factor);
      }
      __syncthreads();
      int idx = threadIdx.x + 1;
      container->d_heap_low_[addr_low + idx] =
        container->d_heap_high_[addr_high] * octree_branching_factor + octree_branching_factor - idx;
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::allocBlocks(const CUDAMatrixf3& point_cloud_img,
                                                                                      const Camera& camera) {
      // fast version, does not guarantee that all blocks are allocated (staggers alloc to the next frame)

      int prev_free_blocks = getHeapHighFreeCount();
      prev_free_blocks += getHeapLowFreeCount();

      resetHashBucketMutex();

      int counter_low;
      CUDA_CHECK(cudaMemcpy(&counter_low, d_heapCounterLow_, sizeof(int), cudaMemcpyDeviceToHost));
      if (sdf_var_threshold_ > 0.f && getHeapLowFreeCount() < low_blocks_to_allocate_) {
        std::cerr << "allocBlocks | need to allocate additional memory in low-res heap" << std::endl;
        const dim3 threads_per_block_lowalloc(octree_branching_factor, 1);
        const dim3 n_blocks(low_blocks_to_allocate_, 1);
        allocateMemoryLow<<<n_blocks, threads_per_block_lowalloc>>>(d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      allocBlocksKernel<<<camera.blocks(), camera.threads()>>>(point_cloud_img.deviceInstance(),
                                                               camera.deviceInstance(),
                                                               max_integration_distance_,
                                                               sdf_truncation_,
                                                               sdf_truncation_scale_,
                                                               d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());

#ifdef RESOLVE_CONFLICT_ALLOCATIONS

      while (1) {
        resetHashBucketMutex();
        allocBlocksKernel<<<camera.blocks(), camera.threads()>>>(point_cloud_img.deviceInstance(),
                                                                 camera.deviceInstance(),
                                                                 max_integration_distance_,
                                                                 sdf_truncation_,
                                                                 sdf_truncation_scale_,
                                                                 d_instance_);

        CUDA_CHECK(cudaDeviceSynchronize());
        int curr_free_blocks = getHeapHighFreeCount();
        curr_free_blocks += getHeapLowFreeCount();
        if (prev_free_blocks == curr_free_blocks) {
          break;
        }
        prev_free_blocks = curr_free_blocks;
      }

#endif
    }

    template <typename T>
    __global__ void allocBlocks3DKernel(const CUDAVectorf3* point_cloud,
                                        const CUDAVectorf3* normals,
                                        const CUDAVectorf* weights,
                                        const Camera* camera,
                                        const float max_integration_distance,
                                        const float sdf_truncation,
                                        const float sdf_truncation_scale,
                                        VoxelContainer<T>* container) {
      const uint point_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (point_idx >= point_cloud->size())
        return;
      const float3& pcam       = point_cloud->at<1>(point_idx);
      const float3 normal      = normals->at<1>(3 * point_idx);
      const float point_weight = weights->at<1>(point_idx);
      const float range        = norm3df(pcam.x, pcam.y, pcam.z);

      if (range == 0.f)
        return;

      const float3 cam_dir  = normalize(pcam);
      const float3 norm_dir = normalize(normal);

      const float t         = getTruncation(range, sdf_truncation, sdf_truncation_scale);
      const float min_depth = min(max_integration_distance, range - t);
      const float max_depth = min(max_integration_distance, range + t);

      if (min_depth >= max_depth)
        return;

      float3 pcam_min, pcam_max;
      if (container->projective_sdf_) {
        pcam_min = pcam + cam_dir * (min_depth - range);
        pcam_max = pcam + cam_dir * (max_depth - range);
      } else {
        pcam_min = pcam + norm_dir * (min_depth - range);
        pcam_max = pcam + norm_dir * (max_depth - range);
      }

      float3 pw_min = camera->camInWorld() * pcam_min;
      float3 pw_max = camera->camInWorld() * pcam_max;

      float3 dir = normalize(pw_max - pw_min);

      int3 id_current_voxel = worldPointToSDFBlock(container->virtual_voxel_size_, container->voxel_extents_, pw_min);
      int3 id_end           = worldPointToSDFBlock(container->virtual_voxel_size_, container->voxel_extents_, pw_max);

      float3 step = make_float3(sign(dir));

      float3 boundary_pos =
        SDFBlockToWorldPoint(container->virtual_voxel_size_, id_current_voxel + make_int3(clamp(step, 0.0, 1.f))) -
        0.5f * container->virtual_voxel_size_;
      float3 t_max   = (boundary_pos - pw_min) / dir;
      float3 t_delta = (step * sdf_block_size * container->virtual_voxel_size_) / dir;
      int3 id_bound  = make_int3(make_float3(id_end) + step);

      if (fabsf(dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.x - dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }

      if (fabsf(dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.y - dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }

      if (fabsf(dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.z - dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      // clang-format on

      uint iter = 0;
#pragma unroll
      while (iter < max_dda_iteration_count) {
        const int& is_allocated = container->allocBlock(id_current_voxel);

        // traverse voxel grid
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
          id_current_voxel.x += step.x;
          if (id_current_voxel.x == id_bound.x)
            return;
          t_max.x += t_delta.x;
        } else if (t_max.z < t_max.y) {
          id_current_voxel.z += step.z;
          if (id_current_voxel.z == id_bound.z)
            return;
          t_max.z += t_delta.z;
        } else {
          id_current_voxel.y += step.y;
          if (id_current_voxel.y == id_bound.y)
            return;
          t_max.y += t_delta.y;
        }

        iter++;
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::allocBlocks3D(const CUDAVectorf3& point_cloud,
                                                                                        const CUDAVectorf3& normals,
                                                                                        const CUDAVectorf weights,
                                                                                        const Camera& camera) {
      // fast version, does not guarantee that all blocks are allocated (staggers alloc to the next frame)

      int prev_free_blocks = getHeapHighFreeCount();
      prev_free_blocks += getHeapLowFreeCount();

      resetHashBucketMutex();

      int counter_low;
      CUDA_CHECK(cudaMemcpy(&counter_low, d_heapCounterLow_, sizeof(int), cudaMemcpyDeviceToHost));

      if (sdf_var_threshold_ > 0.f && getHeapLowFreeCount() < low_blocks_to_allocate_) {
        std::cerr << "allocBlocks | need to allocate additional memory in low-res heap" << std::endl;
        const dim3 threads_per_block_lowalloc(octree_branching_factor, 1);
        const dim3 n_blocks(low_blocks_to_allocate_, 1);
        allocateMemoryLow<<<n_blocks, threads_per_block_lowalloc>>>(d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      const dim3 threads_per_block(n_threads * n_threads, 1);
      const dim3 n_blocks((point_cloud.size() + threads_per_block.x - 1) / threads_per_block.x, 1);
      allocBlocks3DKernel<<<n_blocks, threads_per_block>>>(point_cloud.deviceInstance(),
                                                           normals.deviceInstance(),
                                                           weights.deviceInstance(),
                                                           camera.deviceInstance(),
                                                           max_integration_distance_,
                                                           sdf_truncation_,
                                                           sdf_truncation_scale_,
                                                           d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());

#ifdef RESOLVE_CONFLICT_ALLOCATIONS

      while (1) {
        resetHashBucketMutex();
        allocBlocks3DKernel<<<n_blocks, threads_per_block>>>(point_cloud.deviceInstance(),
                                                             normals.deviceInstance(),
                                                             weights.deviceInstance(),
                                                             camera.deviceInstance(),
                                                             max_integration_distance_,
                                                             sdf_truncation_,
                                                             sdf_truncation_scale_,
                                                             d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
        int curr_free_blocks = getHeapHighFreeCount();
        curr_free_blocks += getHeapLowFreeCount();
        if (prev_free_blocks == curr_free_blocks) {
          break;
        }
        prev_free_blocks = curr_free_blocks;
      }

#endif
    }

    template <typename T>
    __global__ void integrateDepthMapKernel(const CUDAMatrixf3* point_cloud_img,
                                            const CUDAMatrixuc3* rgb_img,
                                            const Camera* camera,
                                            const float sdf_truncation,
                                            const float sdf_truncation_scale,
                                            const float max_integration_distance,
                                            const uchar integration_weight_sample,
                                            const uchar integration_weight_max,
                                            VoxelContainer<T>* container) {
      // we can access this linearly, we have compact representation
      const uint entry_idx   = blockIdx.x * blockDim.x + threadIdx.x;
      const HashEntry& entry = container->d_compactHashTable_[entry_idx];
      if (entry.ptr == FREE_ENTRY)
        return;
      const int num_voxels = container->getNumVoxels(entry);
      const uint voxel_idx = blockIdx.y * blockDim.y + threadIdx.y;
      if (voxel_idx >= num_voxels)
        return;

      const int scaling_factor = 1 << entry.resolution;

      const int3 pi_base      = SDFBlockToVirtualVoxelPos(entry.pos);
      const int3 voxel_coords = scaling_factor * make_int3(delinearizeVoxelPos(voxel_idx, sdf_block_size / scaling_factor));
      const int3 pi           = pi_base + voxel_coords;
      const float3 pf         = virtualVoxelPosToWorld(container->virtual_voxel_size_, pi);

      // get point in screen
      const float3 pcam = camera->camInWorld().inverse() * pf;

      int2 img_point;
      bool is_good = camera->projectPoint(pcam, img_point);

      if (!is_good)
        return;

      const auto& row = img_point.x;
      const auto& col = img_point.y;

      // if depth is good
      const float depth = camera->getDepth(point_cloud_img->at<1>(row, col));
      if (depth == 0.f || depth > max_integration_distance) // from matrix initialization, emtpy is 0
        return;

      const float depth_normalized = camera->normalizeDepth(depth);

      float sdf              = depth - camera->getDepth(pcam);
      const float truncation = getTruncation(depth, sdf_truncation, sdf_truncation_scale);

      if (sdf <= -truncation)
        return;

      // truncate signed distance
      if (sdf >= 0.f)
        sdf = fminf(truncation, sdf);
      else
        sdf = fmaxf(-truncation, sdf);

      // float weight_update = fmaxf(integration_weight_sample * 1.5f * (1.f - depth_normalized), 1);
      float weight_update = integration_weight_sample;

      // construct current voxel
      T curr;
      curr.sdf    = sdf;
      curr.weight = weight_update;
      curr.rgb    = rgb_img->at<1>(row, col);

      // integrate
      uint volume_idx   = entry.ptr + voxel_idx;
      float curr_mean   = 0.f;
      float curr_sum_sq = 0.f;
      if (container->d_SDFBlocks_[volume_idx].weight > 0) {
        curr_mean   = container->d_SDFBlocks_[volume_idx].sdf;
        curr_sum_sq = container->d_SDFBlocks_[volume_idx].sum_squared;
      } else {
        curr_mean = sdf;
      }
      float delta = (sdf - curr_mean) / (container->virtual_voxel_size_ / 2);

      T merged_voxel;
      if (container->d_SDFBlocks_[volume_idx].weight == 0) {
        container->d_SDFBlocks_[volume_idx].rgb = curr.rgb;
      }
      combineVoxel<T>(container->d_SDFBlocks_[volume_idx], curr, integration_weight_max, merged_voxel);
      container->d_SDFBlocks_[volume_idx] = merged_voxel;
      float delta2 = (sdf - container->d_SDFBlocks_[volume_idx].sdf) / (container->virtual_voxel_size_ / 2);
      atomicAdd(&container->d_SDFBlocks_[volume_idx].sum_squared, delta * delta2);
    }

    template <typename T>
    __global__ void checkVoxelsKernel(VoxelContainer<T>* container) {
      const HashEntry& entry = container->d_compactHashTable_[blockIdx.x];
      const int num_voxels   = container->getNumVoxels(entry);
      if (threadIdx.x >= num_voxels)
        return;
      const auto& voxel = container->d_SDFBlocks_[entry.ptr + threadIdx.x];
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateDepthMap(const CUDAMatrixf3& point_cloud_img,
                                                                                            const CUDAMatrixuc3& rgb_img,
                                                                                            const Camera& camera) {
      const dim3 threads_per_block(n_threads, n_threads, 1);
      const dim3 n_blocks((current_occupied_blocks_ + threads_per_block.x - 1) / threads_per_block.x,
                          (voxel_block_volume_ + threads_per_block.y - 1) / threads_per_block.y,
                          1);
      if (current_occupied_blocks_ > 0) {
        integrateDepthMapKernel<<<n_blocks, threads_per_block>>>(point_cloud_img.deviceInstance(),
                                                                 rgb_img.deviceInstance(),
                                                                 camera.deviceInstance(),
                                                                 sdf_truncation_,
                                                                 sdf_truncation_scale_,
                                                                 max_integration_distance_,
                                                                 integration_weight_sample_,
                                                                 integration_weight_max_,
                                                                 d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __global__ void integrate3DKernel(const CUDAVectorf3* point_cloud,
                                      const CUDAVectorf3* normals,
                                      const CUDAVectorf* weights,
                                      const Camera* camera,
                                      const float sdf_truncation,
                                      const float sdf_truncation_scale,
                                      const float max_integration_distance,
                                      const uchar integration_weight_sample,
                                      const uchar integration_weight_max,
                                      VoxelContainer<T>* container) {
      const uint point_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (point_idx >= point_cloud->size())
        return;
      const float3 pcam        = point_cloud->at<1>(point_idx);
      const float3 normal      = normals->at<1>(3 * point_idx);
      const float point_weight = weights->at<1>(point_idx);
      const float3 pw          = camera->camInWorld() * pcam;
      const float range        = norm3df(pcam.x, pcam.y, pcam.z);
      if (range < 1e-6 || range > max_integration_distance) // from matrix initialization, emtpy is 0
        return;

      const float3 cam_dir  = normalize(pcam);
      const float3 norm_dir = normalize(normal);

      const float truncation = getTruncation(range, sdf_truncation, sdf_truncation_scale);
      const float min_depth  = min(max_integration_distance, range - truncation);
      const float max_depth  = min(max_integration_distance, range + truncation);

      if (min_depth >= max_depth)
        return;

      float3 pcam_min;
      float3 pcam_max;
      if (container->projective_sdf_) {
        pcam_min = pcam - cam_dir * truncation;
        pcam_max = pcam + cam_dir * truncation;
      } else {
        pcam_min = pcam + norm_dir * (min_depth - range);
        pcam_max = pcam + norm_dir * (max_depth - range);
      }

      float3 pw_min = camera->camInWorld() * pcam_min;
      float3 pw_max = camera->camInWorld() * pcam_max;

      float3 dir = normalize(pw_max - pw_min);

      int3 id_current_voxel = worldPointToVirtualVoxelPos(container->virtual_voxel_size_, pw_min);
      int3 id_end           = worldPointToVirtualVoxelPos(container->virtual_voxel_size_, pw_max);

      float3 step = make_float3(sign(dir));

      float3 boundary_pos =
        virtualVoxelPosToWorld(container->virtual_voxel_size_, id_current_voxel + make_int3(clamp(step, 0.0, 1.f))) -
        0.5f * container->virtual_voxel_size_;
      float3 t_max   = (boundary_pos - pw_min) / dir;
      float3 t_delta = (step * container->virtual_voxel_size_) / dir;
      int3 id_bound  = make_int3(make_float3(id_end) + step);

      if (fabsf(dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.x - dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }

      if (fabsf(dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.y - dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }

      if (fabsf(dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.z - dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      // clang-format on

      uint iter = 0; // iter < max_loop_iter_count
#pragma unroll
      while (iter < max_dda_iteration_count) {
        // update weight and sdf value of id_current_voxel
        const int3 sdf_block_pos =
          virtualVoxelPosToSDFBlock(id_current_voxel, container->virtual_voxel_size_, container->voxel_extents_);
        const HashEntry& entry = container->getHashEntry(sdf_block_pos);
        if (entry.ptr != FREE_ENTRY) {
          const float3 actual_voxel_pos = virtualVoxelPosToWorld(container->virtual_voxel_size_, id_current_voxel);
          const int scale               = 1 << entry.resolution;
          const int3 voxel_pos_aprox =
            make_int3(id_current_voxel.x / scale, id_current_voxel.y / scale, id_current_voxel.z / scale);
          const float3 voxel_pos        = virtualVoxelPosToWorld(container->getVoxelSize(entry), voxel_pos_aprox);
          const float3 voxel_pos_camera = camera->camInWorld().inverse() * voxel_pos;
          const float voxel_range       = norm3df(voxel_pos_camera.x, voxel_pos_camera.y, voxel_pos_camera.z);
          float sdf;
          if (container->projective_sdf_) {
            sdf = range - voxel_range;
          } else {
            sdf = dot((voxel_pos_camera - pcam), norm_dir);
          }
          if (sdf <= -truncation)
            break;
          if (sdf >= 0.f)
            sdf = fminf(truncation, sdf);
          else
            sdf = fmaxf(-truncation, sdf);

          const float range_normalized = camera->normalizeDepth(range);
          float weight_update;

          if (container->projective_sdf_) {
            // weight_update = fmaxf(integration_weight_sample * 1.5f * (1.f - range_normalized), 1);
            weight_update = integration_weight_sample; // Note: Could implement adaptive weighting
          } else {
            // weight_update = fmaxf(1.5 * point_weight * integration_weight_sample, 1.f);
            weight_update = integration_weight_sample; // Note: Could implement adaptive weighting
          }

          T curr;
          curr.sdf    = sdf;
          curr.weight = weight_update;

          // integrate
          const uint volume_idx = entry.ptr + virtualVoxelPosToSDFBlockIndex(id_current_voxel, sdf_block_size / scale);
          float curr_mean       = 0.f;
          float curr_sum_sq     = 0.f;
          if (container->d_SDFBlocks_[volume_idx].weight > 0) {
            curr_mean   = container->d_SDFBlocks_[volume_idx].sdf;
            curr_sum_sq = container->d_SDFBlocks_[volume_idx].sum_squared;
          }
          float delta = (sdf - curr_mean) / (container->virtual_voxel_size_ / 2);
          T merged_voxel;
          combineVoxel<T>(container->d_SDFBlocks_[volume_idx], curr, integration_weight_max, merged_voxel);
          container->d_SDFBlocks_[volume_idx] = merged_voxel;
          float delta2 = (sdf - container->d_SDFBlocks_[volume_idx].sdf) / (container->virtual_voxel_size_ / 2);
          atomicAdd(&container->d_SDFBlocks_[volume_idx].sum_squared, delta * delta2);
        }

        // traverse voxel grid
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
          id_current_voxel.x += step.x;
          if (id_current_voxel.x == id_bound.x)
            return;
          t_max.x += t_delta.x;
        } else if (t_max.z < t_max.y) {
          id_current_voxel.z += step.z;
          if (id_current_voxel.z == id_bound.z)
            return;
          t_max.z += t_delta.z;
        } else {
          id_current_voxel.y += step.y;
          if (id_current_voxel.y == id_bound.y)
            return;
          t_max.y += t_delta.y;
        }
        iter++;
      }
    }

    template <typename T>
    __host__ void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrate3D(const CUDAVectorf3& point_cloud,
                                                                                               const CUDAVectorf3& normals,
                                                                                               const CUDAVectorf& weights,
                                                                                               const Camera& camera) {
      const dim3 threads_per_block(n_threads * n_threads, 1);
      const dim3 n_blocks((point_cloud.size() + threads_per_block.x - 1) / threads_per_block.x, 1);
      if (current_occupied_blocks_ > 0) {
        integrate3DKernel<<<n_blocks, threads_per_block>>>(point_cloud.deviceInstance(),
                                                           normals.deviceInstance(),
                                                           weights.deviceInstance(),
                                                           camera.deviceInstance(),
                                                           sdf_truncation_,
                                                           sdf_truncation_scale_,
                                                           max_integration_distance_,
                                                           integration_weight_sample_,
                                                           integration_weight_max_,
                                                           d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __global__ void reintegrate3DKernel(const CUDAVectorf3* point_cloud,
                                        const CUDAVectorf3* normals,
                                        const CUDAVectorf* weights,
                                        const Camera* camera,
                                        const float sdf_truncation,
                                        const float sdf_truncation_scale,
                                        const float max_integration_distance,
                                        const uchar integration_weight_sample,
                                        const uchar integration_weight_max,
                                        VoxelContainer<T>* container) {
      const uint point_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (point_idx >= point_cloud->size())
        return;
      const float3 pcam        = point_cloud->at<1>(point_idx);
      const float3 normal      = normals->at<1>(3 * point_idx);
      const float point_weight = weights->at<1>(point_idx);
      const float3 pw          = camera->camInWorld() * pcam;
      const float range        = norm3df(pcam.x, pcam.y, pcam.z);
      if (range == 0.f || range > max_integration_distance) // from matrix initialization, emtpy is 0
        return;

      const float3 cam_dir  = normalize(pcam);
      const float3 norm_dir = normalize(normal);

      const float truncation = getTruncation(range, sdf_truncation, sdf_truncation_scale);
      const float min_depth  = min(max_integration_distance, range - truncation);
      const float max_depth  = min(max_integration_distance, range + truncation);

      if (min_depth >= max_depth)
        return;

      float3 pcam_min;
      float3 pcam_max;
      if (container->projective_sdf_) {
        pcam_min = pcam - cam_dir * truncation;
        pcam_max = pcam + cam_dir * truncation;
      } else {
        pcam_min = pcam + norm_dir * (min_depth - range);
        pcam_max = pcam + norm_dir * (max_depth - range);
      }

      float3 pw_min = camera->camInWorld() * pcam_min;
      float3 pw_max = camera->camInWorld() * pcam_max;

      float3 dir = normalize(pw_max - pw_min);

      int3 id_current_voxel = worldPointToVirtualVoxelPos(container->virtual_voxel_size_, pw_min);
      int3 id_end           = worldPointToVirtualVoxelPos(container->virtual_voxel_size_, pw_max);

      float3 step = make_float3(sign(dir));

      float3 boundary_pos =
        virtualVoxelPosToWorld(container->virtual_voxel_size_, id_current_voxel + make_int3(clamp(step, 0.0, 1.f))) -
        0.5f * container->virtual_voxel_size_;
      float3 t_max   = (boundary_pos - pw_min) / dir;
      float3 t_delta = (step * container->virtual_voxel_size_) / dir;
      int3 id_bound  = make_int3(make_float3(id_end) + step);

      if (fabsf(dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.x - dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }

      if (fabsf(dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.y - dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }

      if (fabsf(dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.z - dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      // clang-format on

      uint iter = 0; // iter < max_loop_iter_count
#pragma unroll
      while (iter < max_dda_iteration_count) {
        const int3 sdf_block_pos =
          virtualVoxelPosToSDFBlock(id_current_voxel, container->virtual_voxel_size_, container->voxel_extents_);
        const HashEntry& entry = container->getHashEntryReintegrate(sdf_block_pos);
        if (entry.ptr != FREE_ENTRY) {
          const float3 actual_voxel_pos = virtualVoxelPosToWorld(container->virtual_voxel_size_, id_current_voxel);
          const int scale               = 1 << entry.resolution;
          const int3 voxel_pos_aprox =
            make_int3(id_current_voxel.x / scale, id_current_voxel.y / scale, id_current_voxel.z / scale);
          const float3 voxel_pos        = virtualVoxelPosToWorld(container->getVoxelSize(entry), voxel_pos_aprox);
          const float3 voxel_pos_camera = camera->camInWorld().inverse() * voxel_pos;
          const float voxel_range       = norm3df(voxel_pos_camera.x, voxel_pos_camera.y, voxel_pos_camera.z);
          float sdf;
          if (container->projective_sdf_) {
            sdf = range - voxel_range;
          } else {
            sdf = dot((voxel_pos_camera - pcam), norm_dir);
          }
          if (sdf <= -truncation)
            break;
          if (sdf >= 0.f)
            sdf = fminf(truncation, sdf);
          else
            sdf = fmaxf(-truncation, sdf);

          const float range_normalized = camera->normalizeDepth(range);
          float weight_update;

          if (container->projective_sdf_) {
            // weight_update = fmaxf(integration_weight_sample * 1.5f * (1.f - range_normalized), 1);
            weight_update = integration_weight_sample; // Note: Could implement adaptive weighting
          } else {
            // weight_update = fmaxf(1.5 * point_weight * integration_weight_sample, 1.f);
            weight_update = integration_weight_sample; // Note: Could implement adaptive weighting
          }

          T curr;
          curr.sdf    = sdf;
          curr.weight = weight_update;

          // integrate
          const uint volume_idx = entry.ptr + virtualVoxelPosToSDFBlockIndex(id_current_voxel, sdf_block_size / scale);
          T merged_voxel;
          combineVoxel<T>(container->d_SDFBlocks_[volume_idx], curr, integration_weight_max, merged_voxel);
          container->d_SDFBlocks_[volume_idx] = merged_voxel;
        }

        // traverse voxel grid
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
          id_current_voxel.x += step.x;
          if (id_current_voxel.x == id_bound.x)
            return;
          t_max.x += t_delta.x;
        } else if (t_max.z < t_max.y) {
          id_current_voxel.z += step.z;
          if (id_current_voxel.z == id_bound.z)
            return;
          t_max.z += t_delta.z;
        } else {
          id_current_voxel.y += step.y;
          if (id_current_voxel.y == id_bound.y)
            return;
          t_max.y += t_delta.y;
        }
        iter++;
      }
    }

    template <typename T>
    __host__ void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::reintegrate3D(const CUDAVectorf3& point_cloud,
                                                                                                 const CUDAVectorf3& normals,
                                                                                                 const CUDAVectorf& weights,
                                                                                                 const Camera& camera) {
      const dim3 threads_per_block(n_threads * n_threads, 1);
      const dim3 n_blocks((point_cloud.size() + threads_per_block.x - 1) / threads_per_block.x, 1);
      if (current_occupied_blocks_ > 0) {
        integrate3DKernel<<<n_blocks, threads_per_block>>>(point_cloud.deviceInstance(),
                                                           normals.deviceInstance(),
                                                           weights.deviceInstance(),
                                                           camera.deviceInstance(),
                                                           sdf_truncation_,
                                                           sdf_truncation_scale_,
                                                           max_integration_distance_,
                                                           integration_weight_sample_,
                                                           integration_weight_max_,
                                                           d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    // ! pack int and float into a single uint64
    __device__ __forceinline__ unsigned long long pack(int a, float b) {
      return (((unsigned long long) (*(reinterpret_cast<unsigned*>(&b)))) << 32) + *(reinterpret_cast<unsigned*>(&a));
    }

    // ! unpack an uint64 into an int and a float
    __device__ __forceinline__ void unpack(int& a, float& b, unsigned long long val) {
      unsigned ma = (unsigned) (val & 0x0FFFFFFFFULL);
      a           = *(reinterpret_cast<int*>(&ma));
      unsigned mb = (unsigned) (val >> 32);
      b           = *(reinterpret_cast<float*>(&mb));
    }

    // ! invalidate voxel kernel, if element is noisy (weight low) we set the weight to zero
    template <typename T>
    __global__ void starveVoxelsKernel(VoxelContainer<T>* container,
                                       const Camera* camera,
                                       const bool is_depth_buffer,
                                       CUDAMatrixu64* depth_buff) {
      const uint idx         = blockIdx.x;
      const HashEntry& entry = container->d_compactHashTable_[idx];

      const int3 pi_base = SDFBlockToVirtualVoxelPos(entry.pos);
      const uint i       = threadIdx.x; // inside an SDF block
      const int3 pi      = pi_base + make_int3(delinearizeVoxelPos(i));
      const float3 pf    = virtualVoxelPosToWorld(container->virtual_voxel_size_, pi); // sdf block in world coordinates

      uint volume_idx = entry.ptr + i;
      const T& vox    = container->d_SDFBlocks_[volume_idx];

      // get point in screen
      const float3 pcam = camera->camInWorld().inverse() * pf;

      float depth = camera->getDepth(pcam);
      if (depth < camera->minDepth())
        return;

      int2 img_point;
      bool is_good = camera->projectPoint(pcam, img_point);

      if (!is_good)
        return;

      const auto& row = img_point.x;
      const auto& col = img_point.y;

      // z-buffer stuff
      const int unique_tid = blockDim.x * blockIdx.x + threadIdx.x;

      if (is_depth_buffer) {
        // unsigned long long candidate_depth_idx = pack(tid, depth);
        // depth buffer implemented as comparison between two uint64
        // packing depth and tid, with depth on the first 32 bits
        // this is required to make reproducible experiment
        // in this way, even if depth is the same, the one with lower tid is preferred
        atomicMin(&(depth_buff->at<1>(row, col)), pack(unique_tid, depth));
        return;
      }

      unsigned long long candidate_depth_idx = pack(unique_tid, depth);
      if (candidate_depth_idx != depth_buff->at<1>(row, col))
        return;

      // should be typically exectued only every n'th frame
      uchar weight                                            = container->d_SDFBlocks_[entry.ptr + threadIdx.x].weight;
      weight                                                  = max(0, weight - 1);
      container->d_SDFBlocks_[entry.ptr + threadIdx.x].weight = weight;
    }

    // ! invalidate voxel if weight is low, noisy elements
    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::starveVoxels(const Camera& camera) {
      if (init_buff) {
        depth_buff = CUDAMatrixu64(camera.rows(), camera.cols());
        init_buff  = false;
      }
      depth_buff.fill(numeric_limits<unsigned long long>::max(), true);
      if (current_occupied_blocks_ > 0) {
        const dim3 n_blocks(current_occupied_blocks_, 1);
        const dim3 threads_per_block(voxel_block_volume_, 1);

        starveVoxelsKernel<<<n_blocks, threads_per_block>>>(
          d_instance_, camera.deviceInstance(), true, depth_buff.deviceInstance());
        CUDA_CHECK(cudaDeviceSynchronize());

        starveVoxelsKernel<<<n_blocks, threads_per_block>>>(
          d_instance_, camera.deviceInstance(), false, depth_buff.deviceInstance());
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __global__ void garbageCollectIdentifyKernel(const VoxelContainer<T>* container, const float truncation_threshold) {
      __shared__ float shared_minSDF[total_sdf_block_size / 2];
      __shared__ float shared_max_weight[total_sdf_block_size / 2];
      const uint idx         = blockIdx.x;
      const HashEntry& entry = container->d_compactHashTable_[idx];
      const int actual_pairs = container->getNumVoxels(entry) / 2;

      if (threadIdx.x >= actual_pairs)
        return;

      const uint idx0 = entry.ptr + 2 * threadIdx.x;
      const uint idx1 = entry.ptr + 2 * threadIdx.x + 1;

      T& v0 = container->d_SDFBlocks_[idx0];
      T& v1 = container->d_SDFBlocks_[idx1];

      float sdf0 = (v0.weight == 0) ? FLT_MAX : fabsf(v0.sdf);
      float sdf1 = (v1.weight == 0) ? FLT_MAX : fabsf(v1.sdf);

      shared_minSDF[threadIdx.x]     = fminf(sdf0, sdf1);
      shared_max_weight[threadIdx.x] = max(v0.weight, v1.weight);

      __syncthreads();

      // Binary tree reduction only over valid threads
      for (int stride = 1; stride < actual_pairs; stride <<= 1) {
        int idx = (threadIdx.x + 1) * (stride * 2) - 1;
        if (idx < actual_pairs && idx - stride >= 0) {
          shared_minSDF[idx]     = fminf(shared_minSDF[idx], shared_minSDF[idx - stride]);
          shared_max_weight[idx] = max(shared_max_weight[idx], shared_max_weight[idx - stride]);
        }
        __syncthreads();
      }

      if (threadIdx.x == actual_pairs - 1) {
        float minSDF                    = shared_minSDF[threadIdx.x];
        uchar maxWeight                 = shared_max_weight[threadIdx.x];
        container->d_hashDecision_[idx] = (minSDF >= truncation_threshold || maxWeight == 0) ? 1 : 0;
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::garbageCollectIdentify(const Camera& camera) {
      if (current_occupied_blocks_ > 0) {
        const dim3 n_blocks(current_occupied_blocks_, 1);
        const dim3 threads_per_block(voxel_block_volume_ / 2, 1);
        const float largest_truncation = getTruncation(camera.maxDepth(), sdf_truncation_, sdf_truncation_scale_);
        garbageCollectIdentifyKernel<<<n_blocks, threads_per_block>>>(d_instance_, largest_truncation);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __device__ bool
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::deleteHashEntryElement(const int3& sdf_block) {
      uint hash      = calculateHash(sdf_block); // hash bucket
      uint start_idx = hash * hash_bucket_size_; // hash position

      for (uint j = 0; j < hash_bucket_size_; j++) {
        uint i                       = start_idx + j;
        const HashEntry& curr        = d_hashTable_[i];
        const int voxel_per_side     = 1 << finest_block_log2_dim - curr.resolution;
        const int voxel_block_volume = voxel_per_side * voxel_per_side * voxel_per_side;
        if (curr.pos.x == sdf_block.x && curr.pos.y == sdf_block.y && curr.pos.z == sdf_block.z && curr.ptr != FREE_ENTRY) {
#ifndef RESOLVE_COLLISION
          if (curr.resolution == 0)
            appendHeapHigh(curr.ptr / voxel_block_volume);
          if (curr.resolution == 1)
            appendHeapLow(curr.ptr / voxel_block_volume);
          deleteHashEntry(d_hashTable_[i]);
          return true;
#endif
#ifdef RESOLVE_COLLISION
          if (curr.offset != 0) { // if there was a pointer set it to the next list element
            // InterlockedExchange(bucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket
            int prev_value = atomicExch(&d_hashTableBucketMutex_[hash], LOCK_ENTRY);
            if (prev_value == LOCK_ENTRY) {
              return false;
            }

            if (prev_value != LOCK_ENTRY) {
              if (curr.resolution == 0)
                appendHeapHigh(curr.ptr / voxel_block_volume);
              if (curr.resolution == 1)
                appendHeapLow(curr.ptr / voxel_block_volume);
              int next_idx    = (i + curr.offset) % total_size_;
              d_hashTable_[i] = d_hashTable_[next_idx];
              deleteHashEntry(d_hashTable_[next_idx]);
              return true;
            }
          } else {
            if (curr.resolution == 0)
              appendHeapHigh(curr.ptr / voxel_block_volume);
            if (curr.resolution == 1)
              appendHeapLow(curr.ptr / voxel_block_volume);
            deleteHashEntry(d_hashTable_[i]);
            return true;
          }
#endif
        }
      }
#ifdef RESOLVE_COLLISION
      const uint idx_last_entry_in_bucket = (hash + 1) * hash_bucket_size_ - 1;
      int i                               = idx_last_entry_in_bucket;
      HashEntry curr;
      curr         = d_hashTable_[i];
      int prev_idx = i;
      i            = idx_last_entry_in_bucket + curr.offset; // go to next element in the list
      i %= total_size_;                                      // check for overflow

      uint max_iter            = 0;
      uint max_loop_iter_count = linked_list_size_;

#pragma unroll 1
      while (max_iter < max_loop_iter_count) {
        curr                         = d_hashTable_[i];
        const int voxel_per_side     = 1 << finest_block_log2_dim - curr.resolution;
        const int voxel_block_volume = voxel_per_side * voxel_per_side * voxel_per_side;
        // found that dude that we need/want to delete
        if (curr.pos.x == sdf_block.x && curr.pos.y == sdf_block.y && curr.pos.z == sdf_block.z && curr.ptr != FREE_ENTRY) {
          int prev_value = atomicExch(&d_hashTableBucketMutex_[hash], LOCK_ENTRY);
          if (prev_value == LOCK_ENTRY) {
            return false;
          }
          if (prev_value != LOCK_ENTRY) {
            if (curr.resolution == 0)
              appendHeapHigh(curr.ptr / voxel_block_volume);
            if (curr.resolution == 1)
              appendHeapLow(curr.ptr / voxel_block_volume);
            deleteHashEntry(d_hashTable_[i]);
            HashEntry prev         = d_hashTable_[prev_idx];
            prev.offset            = curr.offset;
            d_hashTable_[prev_idx] = prev;
            return true;
          }
        }
        // we have found the end of the list
        // should actually never happen because we need to find that guy before
        if (curr.offset == 0) {
          return false;
        }

        prev_idx = i;
        i        = idx_last_entry_in_bucket + curr.offset; // go to next element in the list
        i %= total_size_;                                  // check for overflow

        max_iter++;
      }
#endif
      return false;
    }

    template <typename T>
    __global__ void garbageCollectFreeKernel(VoxelContainer<T>* container) {
      // const uint hashIdx = blockIdx.x;
      const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
      // if in bound and decision to delete the hash entry
      if (idx < container->current_occupied_blocks_ && container->d_hashDecision_[idx] != 0) {
        const HashEntry& entry       = container->d_compactHashTable_[idx];
        const int voxel_per_side     = 1 << finest_block_log2_dim - entry.resolution;
        const int voxel_block_volume = voxel_per_side * voxel_per_side * voxel_per_side;

        // delete hash entry from hash (and performs heap append)
        if (container->deleteHashEntryElement(entry.pos)) {
#pragma unroll 1
          for (uint i = 0; i < voxel_block_volume; ++i) {
            deleteVoxel<T>(container->d_SDFBlocks_[entry.ptr + i]);
          }
        }
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::garbageCollectFree() {
      if (current_occupied_blocks_ > 0) {
        const dim3 threads_per_block(n_threads_cam * n_threads_cam, 1);
        const dim3 n_blocks((current_occupied_blocks_ + threads_per_block.x - 1) / threads_per_block.x, 1);
        garbageCollectFreeKernel<<<n_blocks, threads_per_block>>>(d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __global__ void checkVarSDFKernel(VoxelContainer<T>* container) {
      const uint block_idx = blockIdx.x;
      if (block_idx >= container->current_occupied_blocks_) {
        printf("block_idx>=current_occupied_blocks_");
        return;
      }

      const HashEntry& entry = container->d_compactHashTable_[block_idx];
      if (entry.resolution >= 1)
        return;

      int tid = threadIdx.x;

      const int resolution_factor  = 1 << (finest_block_log2_dim - entry.resolution);
      const int voxel_block_volume = resolution_factor * resolution_factor * resolution_factor;
      if (tid >= voxel_block_volume) {
        return;
      }

      extern __shared__ float shared[];
      float* sum_sq_shared = shared;
      float* weight_shared = shared + blockDim.x;

      float local_sum_sq = 0.f;
      float local_weight = 0.f;

      int gx = (tid % 4) * 2;
      int gy = ((tid / 4) % 4) * 2;
      int gz = (tid / 16) * 2;

      for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx) {
            int x = gx + dx;
            int y = gy + dy;
            int z = gz + dz;

            int local_idx   = z * octree_branching_factor * octree_branching_factor + y * octree_branching_factor + x;
            uint volume_idx = entry.ptr + local_idx;

            const T& voxel = container->d_SDFBlocks_[volume_idx];

            if (voxel.weight > 0) {
              local_sum_sq += voxel.sum_squared;
              local_weight += voxel.weight;
            }
          }

      sum_sq_shared[threadIdx.x] = local_sum_sq;
      weight_shared[threadIdx.x] = local_weight;

      __syncthreads();

      for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
          sum_sq_shared[threadIdx.x] += sum_sq_shared[threadIdx.x + stride];
          weight_shared[threadIdx.x] += weight_shared[threadIdx.x + stride];
        }
        __syncthreads();
      }

      __syncthreads();

      if (threadIdx.x == 0) {
        if (weight_shared[0] < 2)
          return;
        double avg_var = sum_sq_shared[0] / (weight_shared[0] - 1);

        if ((weight_shared[0] - 1) > 1e-6f && avg_var > 0.f && avg_var < container->sdf_var_threshold_) {
          const int3 block_pos = entry.pos;
          const int resolution = entry.resolution + 1;
          if (container->deleteHashEntryElement(block_pos)) {
#pragma unroll 1
            for (uint i = 0; i < voxel_block_volume; ++i) {
              deleteVoxel<T>(container->d_SDFBlocks_[entry.ptr + i]);
            }
            const uint reallocate_idx                    = atomicAdd(container->d_num_reallocate_, 1);
            container->d_reallocate_pos_[reallocate_idx] = block_pos;
            container->d_reallocate_res_[reallocate_idx] = resolution;
          }
        }
      }
    }

    template <typename T>
    __global__ void reintegrateDepthMapKernel(const CUDAMatrixf3* point_cloud_img,
                                              const CUDAMatrixuc3* rgb_img,
                                              const Camera* camera,
                                              const float sdf_truncation,
                                              const float sdf_truncation_scale,
                                              const float max_integration_distance,
                                              const uchar integration_weight_sample,
                                              const uchar integration_weight_max,
                                              VoxelContainer<T>* container) {
      // we can access this linearly, we have compact representation
      const uint entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (entry_idx >= container->d_num_reintegrate_[0])
        return;
      const uint reintegrate_idx = container->d_reintegrate_[entry_idx];
      const HashEntry& entry     = container->d_hashTable_[reintegrate_idx];
      const int num_voxels       = container->getNumVoxels(entry);
      const uint voxel_idx       = blockIdx.y * blockDim.y + threadIdx.y;
      if (voxel_idx >= num_voxels)
        return;

      const int scaling_factor = 1 << entry.resolution;

      const int3 pi_base      = SDFBlockToVirtualVoxelPos(entry.pos);
      const int3 voxel_coords = scaling_factor * make_int3(delinearizeVoxelPos(voxel_idx, sdf_block_size / scaling_factor));
      const int3 pi           = pi_base + voxel_coords;
      const float3 pf         = virtualVoxelPosToWorld(container->virtual_voxel_size_, pi);

      // get point in screen
      const float3 pcam = camera->camInWorld().inverse() * pf;

      int2 img_point;
      bool is_good = camera->projectPoint(pcam, img_point);

      if (!is_good)
        return;

      const auto& row = img_point.x;
      const auto& col = img_point.y;

      // if depth is good
      const float depth = camera->getDepth(point_cloud_img->at<1>(row, col));
      if (depth == 0.f || depth > max_integration_distance) // from matrix initialization, emtpy is 0
        return;

      const float depth_normalized = camera->normalizeDepth(depth);

      float sdf              = depth - camera->getDepth(pcam);
      const float truncation = getTruncation(depth, sdf_truncation, sdf_truncation_scale);

      if (sdf <= -truncation)
        return;

      // truncate signed distance
      if (sdf >= 0.f)
        sdf = fminf(truncation, sdf);
      else
        sdf = fmaxf(-truncation, sdf);

      // float weight_update = fmaxf(integration_weight_sample * 1.5f * (1.f - depth_normalized), 1);
      float weight_update = integration_weight_sample;

      // construct current voxel
      T curr;
      curr.sdf    = sdf;
      curr.weight = weight_update;
      curr.rgb    = rgb_img->at<1>(row, col);

      // integrate
      uint volume_idx = entry.ptr + voxel_idx;

      T merged_voxel;
      if (container->d_SDFBlocks_[volume_idx].weight == 0) {
        container->d_SDFBlocks_[volume_idx].rgb = curr.rgb;
      }
      combineVoxel<T>(container->d_SDFBlocks_[volume_idx], curr, integration_weight_max, merged_voxel);
      container->d_SDFBlocks_[volume_idx] = merged_voxel;
    }

    template <typename T>
    __global__ void reallocBlocksKernel(VoxelContainer<T>* container) {
      const uint entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (entry_idx >= container->d_num_reallocate_[0])
        return;

      const int3& entry_pos = container->d_reallocate_pos_[entry_idx];
      const int& resolution = container->d_reallocate_res_[entry_idx];

      const int& allocation_idx = container->reallocBlock(entry_pos, resolution);
      if (allocation_idx >= 0) {
        const uint reintegrate_idx                 = atomicAdd(container->d_num_reintegrate_, 1);
        container->d_reintegrate_[reintegrate_idx] = allocation_idx;
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::reallocBlocks() {
      uint num_reallocate = 0;
      CUDA_CHECK(cudaMemcpy(&num_reallocate, d_num_reallocate_, sizeof(uint), cudaMemcpyDeviceToHost));
      if (num_reallocate > 0) {
        CUDA_CHECK(cudaMemset(d_reintegrate_, 0, sizeof(uint) * num_sdf_blocks_));
        CUDA_CHECK(cudaMemset(d_num_reintegrate_, 0, sizeof(uint)));
        int prev_free_blocks = getHeapHighFreeCount();
        prev_free_blocks += getHeapLowFreeCount();
        resetHashBucketMutex();
        const dim3 threads_per_block(64, 1);
        const dim3 n_blocks((num_reallocate + threads_per_block.x - 1) / threads_per_block.x);
        reallocBlocksKernel<<<n_blocks, threads_per_block>>>(d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());

#ifdef RESOLVE_CONFLICT_ALLOCATIONS

        while (1) {
          resetHashBucketMutex();

          reallocBlocksKernel<<<n_blocks, threads_per_block>>>(d_instance_);

          CUDA_CHECK(cudaDeviceSynchronize());
          int curr_free_blocks = getHeapHighFreeCount();
          curr_free_blocks += getHeapLowFreeCount();
          if (prev_free_blocks == curr_free_blocks) {
            break;
          }
          prev_free_blocks = curr_free_blocks;
        }

#endif
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::checkVarSDF() {
      if (current_occupied_blocks_ > 0) {
        resetHashBucketMutex();
        CUDA_CHECK(cudaMemset(d_reallocate_pos_, 0, sizeof(int3) * num_sdf_blocks_));
        CUDA_CHECK(cudaMemset(d_reallocate_res_, 0, sizeof(int) * num_sdf_blocks_));
        CUDA_CHECK(cudaMemset(d_num_reallocate_, 0, sizeof(uint)));
        const dim3 threads_per_block(64, 1);
        const dim3 n_blocks(current_occupied_blocks_, 1);
        const size_t shared_mem_size = 2 * threads_per_block.x * sizeof(float);
        checkVarSDFKernel<<<n_blocks, threads_per_block, shared_mem_size>>>(d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::reintegrateDepthMap(const CUDAMatrixf3& point_cloud_img,
                                                                                              const CUDAMatrixuc3& rgb_img,
                                                                                              const Camera& camera) {
      uint num_reintegrate = 0;
      CUDA_CHECK(cudaMemcpy(&num_reintegrate, d_num_reintegrate_, sizeof(uint), cudaMemcpyDeviceToHost));
      if (num_reintegrate > 0) {
        const dim3 threads_per_block(n_threads, n_threads, 1);
        const dim3 n_blocks((num_reintegrate + threads_per_block.x - 1) / threads_per_block.x,
                            (voxel_block_volume_ + threads_per_block.y - 1) / threads_per_block.y,
                            1);
        reintegrateDepthMapKernel<<<n_blocks, n_threads>>>(point_cloud_img.deviceInstance(),
                                                           rgb_img.deviceInstance(),
                                                           camera.deviceInstance(),
                                                           sdf_truncation_,
                                                           sdf_truncation_scale_,
                                                           max_integration_distance_,
                                                           integration_weight_sample_,
                                                           integration_weight_max_,
                                                           d_instance_);
      }
    }

    // ! inserts a hash entry without allocating any memory: used by streaming:
    // ! entry must be modifiable, but just inside the scope of this function, not actually returned
    template <typename T>
    __device__ bool VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::insertHashEntry(HashEntry entry) {
      uint h  = calculateHash(entry.pos);
      uint hp = h * hash_bucket_size_;

      for (uint j = 0; j < hash_bucket_size_; ++j) {
        uint i          = j + hp;
        int prev_weight = 0;
        prev_weight     = atomicCAS(&d_hashTable_[i].ptr, FREE_ENTRY, LOCK_ENTRY);
        if (prev_weight == FREE_ENTRY) {
          d_hashTable_[i] = entry;
          return true;
        }
      }

#ifdef RESOLVE_COLLISION
      // updated variables as after the loop
      const uint idx_last_entry_in_bucket = (h + 1) * hash_bucket_size_ - 1; // get last index of bucket

      uint i = idx_last_entry_in_bucket; // start with the last entry of the current bucket
      HashEntry curr;

      unsigned int max_iter    = 0;
      uint max_loop_iter_count = linked_list_size_;
#pragma unroll 1
      while (max_iter <
             max_loop_iter_count) { // traverse list until end // why find the end? we you are inserting at the start !!!
        curr = d_hashTable_[i];
        if (curr.offset == 0)
          break;                                      // we have found the end of the list
        i = idx_last_entry_in_bucket + curr.offset;   // go to next element in the list
        i %= (hash_bucket_size_ * hash_num_buckets_); // check for overflow

        max_iter++;
      }

      max_iter   = 0;
      int offset = 0;
#pragma unroll 1
      while (max_iter < max_loop_iter_count) { // linear search for free entry
        offset++;
        uint i = (idx_last_entry_in_bucket + offset) % (hash_bucket_size_ * hash_num_buckets_); // go to next hash element
        if ((offset % hash_bucket_size_) == 0)
          continue; // cannot insert into a last bucket element (would conflict with other linked lists)

        int prev_weight = 0;
        uint* d_hash_ui = (uint*) d_hashTable_;
        prev_weight = prev_weight = atomicCAS(&d_hash_ui[3 * idx_last_entry_in_bucket + 1], (uint) FREE_ENTRY, (uint) LOCK_ENTRY);
        if (prev_weight == FREE_ENTRY) { // if free entry found set prev->next = curr & curr->next = prev->next

          HashEntry last_entry_in_bucket = d_hashTable_[idx_last_entry_in_bucket]; // get prev (= lastEntry in Bucket)

          int new_offset_prev =
            (offset << 16) | (last_entry_in_bucket.pos.z & 0x0000ffff); // prev->next = curr (maintain old z-pos)
          int old_offset_prev = 0;
          uint* d_hash_ui     = (uint*) d_hashTable_;
          old_offset_prev = prev_weight = atomicExch(&d_hash_ui[3 * idx_last_entry_in_bucket + 1], new_offset_prev);
          entry.offset                  = old_offset_prev >> 16; // remove prev z-pos from old offset

          d_hashTable_[i] = entry;
          return true;
        }

        max_iter++;
      }
#endif

      printf("insertHashEntry: could not insert entry with hash %d and pos %d %d %d\n", h, entry.pos.x, entry.pos.y, entry.pos.z);

      return false;
    }

  } // namespace cugeoutils
} // namespace cupanutils
