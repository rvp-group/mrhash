#pragma once

#include "camera.cuh"
#include "voxel_hash_utils.cuh"

#include <Eigen/Dense>

namespace cupanutils {
  namespace cugeoutils {

    template <typename T, typename Enable = void>
    class VoxelContainer {
    public:
      // VoxelContainer is only implemented for voxel-derived types
      VoxelContainer() = delete;
    };

    template <typename T>
    class VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>> {
    public:
      // Default constructor deleted - object must be properly initialized
      __host__ explicit VoxelContainer() = delete;
      __host__ explicit VoxelContainer(const uint num_sdf_blocks,
                                       const uint hash_num_buckets,
                                       const float max_integration_distance,
                                       const float sdf_truncation,
                                       const float sdf_truncation_scale,
                                       const float virtual_voxel_size,
                                       const int integration_weight_sample,
                                       const uchar min_weight_threshold,
                                       const float sdf_var_threshold,
                                       const bool projective_sdf,
                                       const bool write_timings,
                                       const std::string memory_allocation_filepath,
                                       const std::string int_profiler_name,
                                       const std::string rendering_profiler_name) :
        num_sdf_blocks_(num_sdf_blocks),
        hash_num_buckets_(hash_num_buckets),
        hash_bucket_size_(hash_bucket_size),
        max_integration_distance_(max_integration_distance),
        sdf_truncation_(sdf_truncation),
        sdf_truncation_scale_(sdf_truncation_scale),
        virtual_voxel_size_(virtual_voxel_size),
        linked_list_size_(linked_list_size),
        integration_weight_sample_(integration_weight_sample),
        integration_weight_max_(integration_weight_max),
        num_integrated_frames_(0),
        min_weight_threshold_(min_weight_threshold),
        sdf_var_threshold_(sdf_var_threshold),
        projective_sdf_(projective_sdf),
        memory_allocation_filepath_(memory_allocation_filepath),
        integration_profiler_(int_profiler_name, write_timings),
        rendering_profiler_(rendering_profiler_name, write_timings) {
        // Memory allocation scaling factors for hierarchical voxel resolution
        // low_blocks_ratio: Reserve 10% of total blocks for medium resolution level
        static constexpr float low_blocks_ratio = 0.1f;

        total_size_             = hash_num_buckets_ * hash_bucket_size;
        voxel_block_volume_     = total_sdf_block_size;
        low_blocks_to_allocate_ = static_cast<uint>(num_sdf_blocks_ * low_blocks_ratio);

        // copy this ptr to device
        CUDA_CHECK(cudaMalloc((void**) &d_instance_, sizeof(VoxelContainer)));
        updateFieldsDevice();

        calculateMemoryUsage();

        // allocate stuff to device
        CUDA_CHECK(cudaMalloc((void**) &d_heap_high_, sizeof(uint) * num_sdf_blocks_));
        CUDA_CHECK(cudaMalloc((void**) &d_heap_low_, sizeof(uint) * num_sdf_blocks_ * octree_branching_factor));
        CUDA_CHECK(cudaMalloc((void**) &d_reallocate_pos_, sizeof(int3) * num_sdf_blocks_));
        CUDA_CHECK(cudaMalloc((void**) &d_reallocate_res_, sizeof(int) * num_sdf_blocks_));
        CUDA_CHECK(cudaMalloc((void**) &d_num_reallocate_, sizeof(uint)));
        CUDA_CHECK(cudaMalloc((void**) &d_reintegrate_, sizeof(uint) * num_sdf_blocks_));
        CUDA_CHECK(cudaMalloc((void**) &d_num_reintegrate_, sizeof(uint)));
        CUDA_CHECK(cudaMalloc((void**) &d_hashTable_, sizeof(HashEntry) * total_size_));
        CUDA_CHECK(cudaMalloc((void**) &d_compactHashTable_, sizeof(HashEntry) * total_size_));
        CUDA_CHECK(cudaMalloc((void**) &d_hashDecision_, sizeof(int) * total_size_));
        CUDA_CHECK(cudaMalloc((void**) &d_compactHashCounter_, sizeof(uint)));
        CUDA_CHECK(cudaMalloc((void**) &d_hashTableBucketMutex_, sizeof(int) * hash_num_buckets_));
        CUDA_CHECK(cudaMalloc((void**) &d_heapCounterHigh_, sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**) &d_heapCounterLow_, sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**) &d_heapCounterLow_, sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**) &d_SDFBlocks_, sizeof(T) * num_sdf_blocks_ * voxel_block_volume_));
        CUDA_CHECK(cudaMalloc((void**) &d_weight_, sizeof(uchar)));

        threads_ = n_threads;
        blocks_  = (total_size_ / threads_) + 1;

        CUDA_CHECK(cudaMemset(d_hashDecision_, 0, sizeof(int) * total_size_));
        CUDA_CHECK(cudaMemset(d_compactHashCounter_, 0, sizeof(uint)));

        int heap_counter_high_init_value = num_sdf_blocks_ - 1;
        CUDA_CHECK(cudaMemcpy(&d_heapCounterHigh_[0], &heap_counter_high_init_value, sizeof(int), cudaMemcpyHostToDevice));

        int heap_counter_low_init_value = -1;
        CUDA_CHECK(cudaMemcpy(&d_heapCounterLow_[0], &heap_counter_low_init_value, sizeof(int), cudaMemcpyHostToDevice));

        updateFieldsDevice();
        resetBuffers();
      }

      // Prevent copying and moving to avoid double-free issues with device memory
      VoxelContainer(const VoxelContainer&)            = delete;
      VoxelContainer& operator=(const VoxelContainer&) = delete;
      VoxelContainer(VoxelContainer&&)                 = delete;
      VoxelContainer& operator=(VoxelContainer&&)      = delete;

      void resetBuffers();
      void calculateMemoryUsage();

      __host__ inline void setIntegrationDistance(const float max_integration_distance) {
        max_integration_distance_ = max_integration_distance;
      };

      __host__ inline void updateFieldsDevice() {
        CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(VoxelContainer), cudaMemcpyHostToDevice));
      }

      ~VoxelContainer() {
        if (d_heap_high_ != nullptr) {
          CUDA_CHECK(cudaFree(d_heap_high_));
        }
        if (d_heap_low_ != nullptr) {
          CUDA_CHECK(cudaFree(d_heap_low_));
        }
        if (d_reallocate_pos_ != nullptr) {
          CUDA_CHECK(cudaFree(d_reallocate_pos_));
        }
        if (d_reallocate_res_ != nullptr) {
          CUDA_CHECK(cudaFree(d_reallocate_res_));
        }
        if (d_num_reallocate_ != nullptr) {
          CUDA_CHECK(cudaFree(d_num_reallocate_));
        }
        if (d_reintegrate_ != nullptr) {
          CUDA_CHECK(cudaFree(d_reintegrate_));
        }
        if (d_num_reintegrate_ != nullptr) {
          CUDA_CHECK(cudaFree(d_num_reintegrate_));
        }
        if (d_hashTable_ != nullptr) {
          CUDA_CHECK(cudaFree(d_hashTable_));
        }
        if (d_compactHashTable_ != nullptr) {
          CUDA_CHECK(cudaFree(d_compactHashTable_));
        }
        if (d_compactHashCounter_ != nullptr) {
          CUDA_CHECK(cudaFree(d_compactHashCounter_));
        }
        if (d_hashDecision_ != nullptr) {
          CUDA_CHECK(cudaFree(d_hashDecision_));
        }
        if (d_hashTableBucketMutex_ != nullptr) {
          CUDA_CHECK(cudaFree(d_hashTableBucketMutex_));
        }
        if (d_heapCounterHigh_ != nullptr) {
          CUDA_CHECK(cudaFree(d_heapCounterHigh_));
        }
        if (d_heapCounterLow_ != nullptr) {
          CUDA_CHECK(cudaFree(d_heapCounterLow_));
        }
        if (d_SDFBlocks_ != nullptr) {
          CUDA_CHECK(cudaFree(d_SDFBlocks_));
        }
        if (d_weight_ != nullptr) {
          CUDA_CHECK(cudaFree(d_weight_));
        }
        if (d_instance_ != nullptr) {
          CUDA_CHECK(cudaFree(d_instance_));
        }
      }

      __host__ void flatAndReduceHashTable(const Camera& camera);
      __host__ void flatAndReduceHashTable();

      __host__ void allocBlocks(const CUDAMatrixf3& point_cloud_img, const Camera& camera);
      __host__ void allocBlocks3D(const CUDAVectorf3& point_cloud,
                                  const CUDAVectorf3& normals,
                                  const CUDAVectorf weights,
                                  const Camera& camera);
      __host__ void integrateDepthMap(const CUDAMatrixf3& point_cloud_img, const CUDAMatrixuc3& rgb_img, const Camera& camera);
      __host__ void reintegrateDepthMap(const CUDAMatrixf3& point_cloud_img, const CUDAMatrixuc3& rgb_img, const Camera& camera);
      __host__ void
      integrate3D(const CUDAVectorf3& point_cloud, const CUDAVectorf3& normals, const CUDAVectorf& weights, const Camera& camera);
      __host__ void reintegrate3D(const CUDAVectorf3& point_cloud,
                                  const CUDAVectorf3& normals,
                                  const CUDAVectorf& weights,
                                  const Camera& camera);
      void integrate(const CUDAMatrixf3& point_cloud_img,
                     const CUDAMatrixuc3& rgb_img,
                     const Camera& camera,
                     const int max_num_frames);
      void integrate(const CUDAVectorf3& point_cloud,
                     const CUDAVectorf3& normals,
                     const CUDAVectorf& weights,
                     const Camera& camera,
                     const int max_num_frames);
      __host__ void garbageCollect(const Camera& camera, const int max_num_frames);

      // some internal methods
      __host__ void resetHashBucketMutex();
      __host__ int getHeapHighFreeCount();
      __host__ int getHeapLowFreeCount();
      __host__ void starveVoxels(const Camera& camera);
      __host__ void garbageCollectIdentify(const Camera& camera);
      __host__ void garbageCollectFree();
      __host__ void checkVarSDF();
      __host__ void reallocBlocks();

      __host__ uchar getVoxelWeight(const Eigen::Vector3f& pw);

      __device__ uchar getVoxelWeightDev(const float3& pw);
      __device__ int consumeHeapHigh();
      __device__ int consumeHeapLow();
      __device__ void appendHeapHigh(const uint ptr);
      __device__ void appendHeapLow(const uint ptr);
      __device__ bool deleteHashEntryElement(const int3& sdf_block);
      __device__ bool isSDFBlockInCameraFrustumApprox(const Camera* camera, const int3& sdf_block);
      __device__ uint64_t calculateHash(const int3& virtual_voxel_pos) const;
      __device__ HashEntry getHashEntry(const int3& sdf_block) const;
      __device__ HashEntry getHashEntryReintegrate(const int3& sdf_block) const;
      __device__ T getVoxel(const int3& virtual_voxel_pos) const;
      __device__ T getVoxel(const float3& pos) const;
      __device__ T getVoxel(const int3& virtual_voxel_pos, int& block_resolution) const;
      __device__ T getVoxel(const float3& pos, int& block_resolution) const;
      __device__ int getNumVoxels(const HashEntry& entry) const;
      __device__ int getNumVoxels(const int3& pos) const;
      __device__ int getNumVoxels(const float3& pos) const;
      __device__ float getVoxelSize(const HashEntry& entry) const;
      __device__ float getVoxelSize(const int3& pos) const;
      __device__ float getVoxelSize(const float3& pos) const;
      __device__ int allocBlock(const int3& pos, const int resolution = 0);
      __device__ int reallocBlock(const int3& pos, const int resolution = 0);
      __device__ bool insertHashEntry(HashEntry entry);
      __device__ bool trilinearInterpolation(const float3& pos, float& dist) const;
      __device__ float3 getVoxelGradientInterp(const float3& pos, const float truncation) const;
      __device__ float3 getVoxelGradientDiscrete(const float3& pos) const;

      /*RAYCASTING*/
      __device__ float findIntersectionLinear(float t_near, float t_far, float d_near, float d_far) const;
      __device__ bool findIntersectionBisection(const Camera* camera,
                                                const float3& world_cam_pos,
                                                const float3& world_dir,
                                                float d0,
                                                float r0,
                                                float d1,
                                                float r1,
                                                float& alpha) const;
      __device__ uint8_t checkVoxelNeighbors(const float3& pw);

      // clang-format off
      VoxelContainer* d_instance_    = nullptr;    // this ptr in device
      uint* d_heap_high_             = nullptr;    // linear buffer with indices to all unallocated blocks, manages free memory
      uint* d_heap_low_              = nullptr;    // linear buffer with resolution of all unallocated blocks, manages resolution of free memory
      int3* d_reallocate_pos_           = nullptr;    // linear buffer with indices of all blocks to be reintegrated after re-sizing
      int* d_reallocate_res_           = nullptr;    // linear buffer with indices of all blocks to be reintegrated after re-sizing
      uint* d_num_reallocate_       = nullptr;    // single uint keeping track of num blocks to be reintegrated
      uint* d_reintegrate_           = nullptr;    // linear buffer with indices of all blocks to be reintegrated after re-sizing
      uint* d_num_reintegrate_       = nullptr;    // single uint keeping track of num blocks to be reintegrated
      int* d_heapCounterHigh_        = nullptr;    // single int keeping track of num blocks allocated
      int* d_heapCounterLow_         = nullptr;    // single int keeping track of num blocks allocated
      int* d_hashDecision_           = nullptr;    // remove noisy elements, bookep elements to be removed because noisy, follows compactHashTable order
      HashEntry* d_hashTable_        = nullptr;    // hash that stores pointers to sdf blocks
      HashEntry* d_compactHashTable_ = nullptr;    // same as before except that only valid pointers are there
      uint* d_compactHashCounter_    = nullptr;    // atomic counter to add compactified entries atomically
      T* d_SDFBlocks_                = nullptr;    // actual underlying 3D geometry in TSDF form, sub-blocks that contain 8x8x8 voxels (linearized) are allocated by heap
      int* d_hashTableBucketMutex_   = nullptr;    // binary flag per hash bucket; used for allocation to atomically lock a bucket
      uchar* d_weight_ = nullptr;
      float virtual_voxel_size_      = 0.f;
      uchar min_weight_threshold_    = 0;          // threshold weight for trilinearinterpolation
      float sdf_var_threshold_       = 0.f;        // threshold sdf variance for voxel merging
      bool projective_sdf_           = false;      // wheter to use projection or non-projective sdf
      // clang-format on

      // query
      CUDAMatrixu64 depth_buff;
      bool init_buff = true;

      // varying
      uint current_occupied_blocks_;
      uint num_integrated_frames_;

      // constants
      uint num_sdf_blocks_;
      uint low_blocks_to_allocate_;
      uint hash_num_buckets_;
      uint hash_bucket_size_;
      uint total_size_;
      uint voxel_block_volume_;

      // Host-device streaming parameters (set by streamer)
      float3 voxel_extents_;

      // Integration parameters
      float max_integration_distance_;
      float sdf_truncation_;
      float sdf_truncation_scale_;
      int integration_weight_sample_;
      int integration_weight_max_;
      int linked_list_size_;

      // CUDA execution configuration
      int blocks_;
      int threads_;

      // Profiling and diagnostics
      CUDAProfiler integration_profiler_;
      CUDAProfiler rendering_profiler_;
      std::string memory_allocation_filepath_;
    };

    template class VoxelContainer<Voxel>;
    using GeometricVoxelContainer = VoxelContainer<Voxel>;

  } // namespace cugeoutils
} // namespace cupanutils
