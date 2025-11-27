#include <opencv2/opencv.hpp>

#include <iostream>
#include <unordered_set>

#include "test_utils.cuh"
#include <sdf/voxel_data_structures.cuh>

using namespace cupanutils::cugeoutils;

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

__global__ void
worldToSDFBlockKernel(const float3 point, const float virtual_voxel_size, const float3 voxel_extents, float3* transformed_point) {
  int3 block           = worldPointToSDFBlock(virtual_voxel_size, voxel_extents, point);
  transformed_point[0] = SDFBlockToWorldPoint(virtual_voxel_size, block);
}

__global__ void worldToVoxelKernel(const float3 point, const float virtual_voxel_size, float3* transformed_point) {
  int3 voxel_pos       = worldPointToVirtualVoxelPos(virtual_voxel_size, point);
  transformed_point[0] = virtualVoxelPosToWorld(virtual_voxel_size, voxel_pos);
}

__global__ void worldToVoxelIndexKernel(const float3 point,
                                        const float virtual_voxel_size,
                                        const float3 voxel_extents,
                                        const int block_size,
                                        float3* transformed_point) {
  int3 block                         = worldPointToSDFBlock(virtual_voxel_size, voxel_extents, point);
  int3 voxel_pos                     = worldPointToVirtualVoxelPos(virtual_voxel_size, point);
  uint voxel_index                   = virtualVoxelPosToSDFBlockIndex(voxel_pos, block_size);
  uint3 delinearized_local_voxel_pos = delinearizeVoxelPos(voxel_index, block_size);
  int3 delinearized_voxel_pos        = SDFBlockToVirtualVoxelPos(block) + delinearized_local_voxel_pos;
  transformed_point[0]               = virtualVoxelPosToWorld(virtual_voxel_size, delinearized_voxel_pos);
}

TEST(VOXEL, BlockTransformation) {
  getDeviceInfo();
  std::cout << "testing world -> block transformations " << std::endl;
  // checking some vec operators
  float virtual_voxel_size   = 1e-6; // the less the size the more accurate transformation is
  float3 point               = make_float3(47.32, 52.45, 150.23);
  const float3 voxel_extents = make_float3(1.f, 1.f, 1.f);
  float3* transformed_point  = new float3[1];
  float3* dtransformed_point;
  CUDA_CHECK(cudaMalloc((void**) &dtransformed_point, sizeof(float3)));
  worldToSDFBlockKernel<<<1, 1>>>(point, virtual_voxel_size, voxel_extents, dtransformed_point);
  CUDA_CHECK(cudaMemcpy(&transformed_point[0], &dtransformed_point[0], sizeof(float3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  ASSERT_NEAR(point.x, transformed_point[0].x, 1e-4);
  ASSERT_NEAR(point.y, transformed_point[0].y, 1e-4);
  ASSERT_NEAR(point.z, transformed_point[0].z, 1e-4);

  printf("block size %d | virtual voxel size %f | %f %f %f = %f %f %f\n",
         sdf_block_size,
         virtual_voxel_size,
         point.x,
         point.y,
         point.z,
         transformed_point[0].x,
         transformed_point[0].y,
         transformed_point[0].z);
}

TEST(VOXEL, VoxelTransformation) {
  getDeviceInfo();
  std::cout << "testing world -> voxel transformations " << std::endl;
  // checking some vec operators
  float virtual_voxel_size  = 1e-6; // the less the size the more accurate transformation is
  float3 point              = make_float3(47.32, 52.45, 150.23);
  float3* transformed_point = new float3[1];
  float3* dtransformed_point;
  CUDA_CHECK(cudaMalloc((void**) &dtransformed_point, sizeof(float3)));
  worldToVoxelKernel<<<1, 1>>>(point, virtual_voxel_size, dtransformed_point);
  CUDA_CHECK(cudaMemcpy(&transformed_point[0], &dtransformed_point[0], sizeof(float3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  ASSERT_NEAR(point.x, transformed_point[0].x, 1e-4);
  ASSERT_NEAR(point.y, transformed_point[0].y, 1e-4);
  ASSERT_NEAR(point.z, transformed_point[0].z, 1e-4);

  printf("block size %d | virtual voxel size %f | %f %f %f = %f %f %f\n",
         sdf_block_size,
         virtual_voxel_size,
         point.x,
         point.y,
         point.z,
         transformed_point[0].x,
         transformed_point[0].y,
         transformed_point[0].z);
}

TEST(VOXEL, VoxelIndexTransformation) {
  getDeviceInfo();
  std::cout << "testing world->voxel index transformations " << std::endl;
  // checking some vec operators
  float virtual_voxel_size   = 1e-6; // the less the size the more accurate transformation is
  const float3 voxel_extents = make_float3(1.f, 1.f, 1.f);
  float3 point               = make_float3(47.32, 52.45, 150.23);
  float3* transformed_point  = new float3[1];
  float3* dtransformed_point;
  int block_size = sdf_block_size;
  CUDA_CHECK(cudaMalloc((void**) &dtransformed_point, sizeof(float3)));
  worldToVoxelIndexKernel<<<1, 1>>>(point, virtual_voxel_size, voxel_extents, block_size, dtransformed_point);
  CUDA_CHECK(cudaMemcpy(&transformed_point[0], &dtransformed_point[0], sizeof(float3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  ASSERT_NEAR(point.x, transformed_point[0].x, 1e-4);
  ASSERT_NEAR(point.y, transformed_point[0].y, 1e-4);
  ASSERT_NEAR(point.z, transformed_point[0].z, 1e-4);

  printf("block size %d | virtual voxel size %f | %f %f %f = %f %f %f\n",
         sdf_block_size,
         virtual_voxel_size,
         point.x,
         point.y,
         point.z,
         transformed_point[0].x,
         transformed_point[0].y,
         transformed_point[0].z);

  block_size = sdf_block_size / 2;
  worldToVoxelIndexKernel<<<1, 1>>>(point, virtual_voxel_size, voxel_extents, block_size, dtransformed_point);
  CUDA_CHECK(cudaMemcpy(&transformed_point[0], &dtransformed_point[0], sizeof(float3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  ASSERT_NEAR(point.x, transformed_point[0].x, 1e-4);
  ASSERT_NEAR(point.y, transformed_point[0].y, 1e-4);
  ASSERT_NEAR(point.z, transformed_point[0].z, 1e-4);

  printf("block size %d | virtual voxel size %f | %f %f %f = %f %f %f\n",
         sdf_block_size,
         virtual_voxel_size,
         point.x,
         point.y,
         point.z,
         transformed_point[0].x,
         transformed_point[0].y,
         transformed_point[0].z);

  block_size = sdf_block_size / 4;
  worldToVoxelIndexKernel<<<1, 1>>>(point, virtual_voxel_size, voxel_extents, block_size, dtransformed_point);
  CUDA_CHECK(cudaMemcpy(&transformed_point[0], &dtransformed_point[0], sizeof(float3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  ASSERT_NEAR(point.x, transformed_point[0].x, 1e-4);
  ASSERT_NEAR(point.y, transformed_point[0].y, 1e-4);
  ASSERT_NEAR(point.z, transformed_point[0].z, 1e-4);

  printf("block size %d | virtual voxel size %f | %f %f %f = %f %f %f\n",
         sdf_block_size,
         virtual_voxel_size,
         point.x,
         point.y,
         point.z,
         transformed_point[0].x,
         transformed_point[0].y,
         transformed_point[0].z);
}

__global__ void zeroWeightSDFs(GeometricVoxelContainer* container) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= container->num_sdf_blocks_)
    return;

  uint base_idx = idx * container->voxel_block_volume_;
  for (uint i = 0; i < container->voxel_block_volume_; ++i)
    container->d_SDFBlocks_[base_idx + i].weight = 0;
}

constexpr float default_depth            = 1.f;
constexpr int hash_num_buckets           = 250000;
constexpr int num_sdf_blocks             = 500000;
constexpr float max_integration_distance = 5;
constexpr float sdf_truncation           = 0.02;
constexpr float sdf_truncation_scale     = 0.01;
constexpr int integration_weight_sample  = 3;
constexpr float virtual_voxel_size       = 0.005;
constexpr float min_depth                = 0;
constexpr float max_depth                = 5;
constexpr int n_frames_invalidate_voxels = 10;

constexpr uchar min_weight_threshold = 0;

constexpr float sdf_var_threshold = 0.f;
constexpr bool projective_sdf     = true;

TEST(HASHTABLE, AllocationDeletion) {
  srand(time(NULL));
  uint rows = 400;
  uint cols = 400;
  CUDAMatrixf depth_img(rows, cols);
  depth_img.fill(default_depth);

  cv::Mat cv_depth(rows, cols, CV_32FC1);
  for (int r = 0; r < cv_depth.rows; ++r) {
    for (int c = 0; c < cv_depth.cols; ++c) {
      cv_depth.at<float>(r, c) = depth_img.at(r, c) / max_depth;
    }
  }

  // some redundancy just to test everything
  Eigen::Matrix3f cam_K;
  cam_K << 400.f, 0.f, cols / 2.f, 0.f, 400.f, rows / 2.f, 0.f, 0.f, 1.f;
  CUDAMat3 d_cam_K(cam_K);

  Camera camera(d_cam_K, rows, cols, min_depth, max_depth);
  camera.setDepthImage(depth_img);

  Eigen::Matrix4f cam_in_world = Eigen::Matrix4f::Identity();
  camera.setCamInWorld(cam_in_world);
  CUDAMatrixf3 point_cloud_img;
  camera.computeCloud(point_cloud_img);

  GeometricVoxelContainer voxelhasher(num_sdf_blocks,
                                      hash_num_buckets,
                                      max_integration_distance,
                                      sdf_truncation,
                                      sdf_truncation_scale,
                                      virtual_voxel_size,
                                      integration_weight_sample,
                                      min_weight_threshold,
                                      sdf_var_threshold,
                                      projective_sdf,
                                      true,
                                      "memory_allocation.txt",
                                      "integration_profiler",
                                      "rendering_profiler");

  voxelhasher.resetBuffers();

  CUDAMatrixuc3 rgb_img;
  rgb_img.resize(rows, cols);
  uchar3 rgb = make_uchar3(255, 0, 0);
  rgb_img.fill(rgb);

  voxelhasher.integrate(point_cloud_img, rgb_img, camera, 0);

  std::cerr << "setting sdf weights to zero and let garbage collector should select the elements for removal" << std::endl;
  // resetting heap and make drop weights sdf blocks
  {
    const dim3 n_blocks((voxelhasher.total_size_ + (n_threads * n_threads) - 1) / (n_threads * n_threads), 1);
    const dim3 threads_per_block((n_threads * n_threads), 1);
    zeroWeightSDFs<<<n_blocks, threads_per_block>>>(voxelhasher.d_instance_);
  }
  // now the garbage collector should remove all the hash entries that are in the compactHashTable, from the general hashTable
  voxelhasher.garbageCollect(camera, n_frames_invalidate_voxels);

  const uint free_heap_count  = voxelhasher.getHeapHighFreeCount();
  uint left_allocated_entries = 0;
  HashEntry* h_hashTable      = new HashEntry[voxelhasher.total_size_];
  CUDA_CHECK(
    cudaMemcpy(h_hashTable, voxelhasher.d_hashTable_, sizeof(HashEntry) * voxelhasher.total_size_, cudaMemcpyDeviceToHost));
  for (uint i = 0; i < voxelhasher.total_size_; ++i) {
    if (h_hashTable[i].ptr != FREE_ENTRY)
      left_allocated_entries++;
  }
  delete[] h_hashTable;

  std::cerr << "heap sanity check" << std::endl;
  const uint tot_sdf_blocks = free_heap_count + left_allocated_entries;
  std::cerr << "tot sdf blocks " << tot_sdf_blocks << " | left_allocated_entries " << left_allocated_entries
            << " + elements on free heap " << free_heap_count << " | num_sdf_blocks: " << voxelhasher.num_sdf_blocks_
            << std::endl;

  ASSERT_EQ(tot_sdf_blocks, voxelhasher.num_sdf_blocks_);

  std::cerr << "remove elements in compact hash table" << std::endl;
  // store this before cleaning
  const int occupied_blocks_to_be_removed = voxelhasher.current_occupied_blocks_;
  // here we clean the compact hash table, this should not be populated again
  // since remained allocated blocks are out of the camera frustum
  voxelhasher.flatAndReduceHashTable(camera);
  ASSERT_EQ(voxelhasher.current_occupied_blocks_, 0);

  std::cerr << "other buffers check, they should be empty" << std::endl;
  // hash decision buffer should contain initial number of hash entry, all marked to be removed (1)
  int* h_hashDecision = new int[voxelhasher.total_size_];
  CUDA_CHECK(
    cudaMemcpy(h_hashDecision, voxelhasher.d_hashDecision_, sizeof(int) * voxelhasher.total_size_, cudaMemcpyDeviceToHost));
  int removed_entries = 0;
  for (int i = 0; i < voxelhasher.total_size_; ++i) {
    if (h_hashDecision[i] > 0)
      removed_entries++;
  }
  ASSERT_EQ(occupied_blocks_to_be_removed, removed_entries);
  delete[] h_hashDecision;

  // mutex should be all set to free entry
  int* h_hashTableBucketMutex = new int[hash_num_buckets];
  CUDA_CHECK(cudaMemcpy(
    h_hashTableBucketMutex, voxelhasher.d_hashTableBucketMutex_, sizeof(int) * hash_num_buckets, cudaMemcpyDeviceToHost));
  for (int i = 0; i < hash_num_buckets; ++i) {
    ASSERT_EQ(h_hashTableBucketMutex[i], FREE_ENTRY);
  }

  uint h_compactHashCounter;
  CUDA_CHECK(cudaMemcpy(&h_compactHashCounter, voxelhasher.d_compactHashCounter_, sizeof(uint), cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_compactHashCounter, 0);
}

TEST(HASHTABLE, BufferInitialization) {
  GeometricVoxelContainer voxelhasher(num_sdf_blocks,
                                      hash_num_buckets,
                                      max_integration_distance,
                                      sdf_truncation,
                                      sdf_truncation_scale,
                                      virtual_voxel_size,
                                      integration_weight_sample,
                                      min_weight_threshold,
                                      sdf_var_threshold,
                                      projective_sdf,
                                      true,
                                      "memory_allocation.txt",
                                      "integration_profiler",
                                      "rendering_profiler");
  voxelhasher.resetBuffers();

  std::cerr << "buffer initialization, check if inizialization actually goes through all elements since this is performed in "
               "device kernels"
            << std::endl;

  uint* h_heap = new uint[num_sdf_blocks];
  CUDA_CHECK(cudaMemcpy(h_heap, voxelhasher.d_heap_high_, sizeof(uint) * num_sdf_blocks, cudaMemcpyDeviceToHost));
  for (uint i = 0; i < num_sdf_blocks; ++i) {
    ASSERT_EQ(h_heap[i], voxelhasher.num_sdf_blocks_ - i - 1);
  }

  HashEntry* h_hashTable = new HashEntry[voxelhasher.total_size_];
  CUDA_CHECK(
    cudaMemcpy(h_hashTable, voxelhasher.d_hashTable_, sizeof(HashEntry) * voxelhasher.total_size_, cudaMemcpyDeviceToHost));
  for (uint i = 0; i < voxelhasher.total_size_; ++i) {
    ASSERT_EQ(h_hashTable[i].pos.x, 0);
    ASSERT_EQ(h_hashTable[i].pos.y, 0);
    ASSERT_EQ(h_hashTable[i].pos.z, 0);
    ASSERT_EQ(h_hashTable[i].offset, NO_OFFSET);
    ASSERT_EQ(h_hashTable[i].ptr, FREE_ENTRY);
  }

  HashEntry* h_compactHashTable = new HashEntry[voxelhasher.total_size_];
  CUDA_CHECK(cudaMemcpy(
    h_compactHashTable, voxelhasher.d_compactHashTable_, sizeof(HashEntry) * voxelhasher.total_size_, cudaMemcpyDeviceToHost));
  for (uint i = 0; i < voxelhasher.total_size_; ++i) {
    ASSERT_EQ(h_compactHashTable[i].pos.x, 0);
    ASSERT_EQ(h_compactHashTable[i].pos.y, 0);
    ASSERT_EQ(h_compactHashTable[i].pos.z, 0);
    ASSERT_EQ(h_compactHashTable[i].offset, NO_OFFSET);
    ASSERT_EQ(h_compactHashTable[i].ptr, FREE_ENTRY);
  }

  int* h_hashDecision = new int[voxelhasher.total_size_];
  CUDA_CHECK(
    cudaMemcpy(h_hashDecision, voxelhasher.d_hashDecision_, sizeof(int) * voxelhasher.total_size_, cudaMemcpyDeviceToHost));
  for (int i = 0; i < voxelhasher.total_size_; ++i) {
    ASSERT_EQ(h_hashDecision[i], 0);
  }

  int* h_hashTableBucketMutex = new int[hash_num_buckets];
  CUDA_CHECK(cudaMemcpy(
    h_hashTableBucketMutex, voxelhasher.d_hashTableBucketMutex_, sizeof(int) * hash_num_buckets, cudaMemcpyDeviceToHost));
  for (int i = 0; i < hash_num_buckets; ++i) {
    ASSERT_EQ(h_hashTableBucketMutex[i], FREE_ENTRY);
  }

  uint h_compactHashCounter;
  CUDA_CHECK(cudaMemcpy(&h_compactHashCounter, voxelhasher.d_compactHashCounter_, sizeof(uint), cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_compactHashCounter, 0);

  uint h_heapCounter;
  CUDA_CHECK(cudaMemcpy(&h_heapCounter, voxelhasher.d_heapCounterHigh_, sizeof(uint), cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_heapCounter, voxelhasher.num_sdf_blocks_ - 1);
}

TEST(HASHTABLE, HeapSanityCheck) {
  uint rows = 400;
  uint cols = 400;
  CUDAMatrixf depth_img(rows, cols);
  depth_img.fill(default_depth);

  cv::Mat cv_depth(rows, cols, CV_32FC1);
  for (int r = 0; r < cv_depth.rows; ++r) {
    for (int c = 0; c < cv_depth.cols; ++c) {
      cv_depth.at<float>(r, c) = depth_img.at(r, c) / max_depth;
    }
  }

  // some redundancy just to test everything
  Eigen::Matrix3f cam_K;
  cam_K << 400.f, 0.f, cols / 2.f, 0.f, 400.f, rows / 2.f, 0.f, 0.f, 1.f;
  CUDAMat3 d_cam_K(cam_K);

  Camera camera(d_cam_K, rows, cols, min_depth, max_depth);
  camera.setDepthImage(depth_img);

  Eigen::Matrix4f cam_in_world = Eigen::Matrix4f::Identity();
  camera.setCamInWorld(cam_in_world);
  CUDAMatrixf3 point_cloud_img;
  camera.computeCloud(point_cloud_img);

  GeometricVoxelContainer voxelhasher(num_sdf_blocks,
                                      hash_num_buckets,
                                      max_integration_distance,
                                      sdf_truncation,
                                      sdf_truncation_scale,
                                      virtual_voxel_size,
                                      integration_weight_sample,
                                      min_weight_threshold,
                                      sdf_var_threshold,
                                      projective_sdf,
                                      true,
                                      "memory_allocation.txt",
                                      "integration_profiler",
                                      "rendering_profiler");
  voxelhasher.resetBuffers();

  CUDAMatrixuc3 rgb_img;
  rgb_img.resize(rows, cols);
  uchar3 rgb = make_uchar3(255, 0, 0);
  rgb_img.fill(rgb);

  voxelhasher.integrate(point_cloud_img, rgb_img, camera, 0);

  HashEntry* h_hashTable = new HashEntry[voxelhasher.total_size_];
  uint* h_heap           = new uint[voxelhasher.num_sdf_blocks_];
  uint h_heapCounter;

  CUDA_CHECK(cudaMemcpy(&h_heapCounter, voxelhasher.d_heapCounterHigh_, sizeof(uint), cudaMemcpyDeviceToHost));
  h_heapCounter++; // points to the first free entry: number of blocks is one more

  CUDA_CHECK(cudaMemcpy(h_heap, voxelhasher.d_heap_high_, sizeof(uint) * voxelhasher.num_sdf_blocks_, cudaMemcpyDeviceToHost));
  CUDA_CHECK(
    cudaMemcpy(h_hashTable, voxelhasher.d_hashTable_, sizeof(HashEntry) * voxelhasher.total_size_, cudaMemcpyDeviceToHost));

  // check for duplicates
  class DummyVoxels {
  public:
    DummyVoxels() {
    }
    ~DummyVoxels() {
    }
    bool operator<(const DummyVoxels& other) const {
      if (x == other.x) {
        if (y == other.y) {
          return z < other.z;
        }
        return y < other.y;
      }
      return x < other.x;
    }

    bool operator==(const DummyVoxels& other) const {
      return x == other.x && y == other.y && z == other.z;
    }

    int x, y, z, i;
    int offset;
    int ptr;
  };

  // duplicate free pointers in heap array
  std::unordered_set<uint> pointers_free_hash;
  std::vector<int> pointers_free_vec(voxelhasher.num_sdf_blocks_, 0);
  for (uint i = 0; i < h_heapCounter; ++i) {
    pointers_free_hash.insert(h_heap[i]);
    pointers_free_vec[h_heap[i]] = FREE_ENTRY;
  }

  ASSERT_EQ(pointers_free_hash.size(), h_heapCounter);

  uint num_occupied     = 0;
  uint num_active_mutex = 0;

  std::list<DummyVoxels> l;

  for (uint i = 0; i < voxelhasher.total_size_; ++i) {
    if (h_hashTable[i].ptr == LOCK_ENTRY) {
      num_active_mutex++;
    }

    if (h_hashTable[i].ptr != FREE_ENTRY) {
      num_occupied++;
      DummyVoxels a;
      a.x = h_hashTable[i].pos.x;
      a.y = h_hashTable[i].pos.y;
      a.z = h_hashTable[i].pos.z;
      l.push_back(a);

      if (pointers_free_hash.find(h_hashTable[i].ptr / voxelhasher.voxel_block_volume_) != pointers_free_hash.end()) {
        FAIL() << "ptr " << (h_hashTable[i].ptr / voxelhasher.voxel_block_volume_)
               << " is on free heap, but also marked as an allocated entry";
      }
      pointers_free_vec[h_hashTable[i].ptr / voxelhasher.voxel_block_volume_] = LOCK_ENTRY;
    }
  }

  uint num_heap_free     = 0;
  uint num_heap_occupied = 0;
  for (uint i = 0; i < voxelhasher.num_sdf_blocks_; ++i) {
    if (pointers_free_vec[i] == FREE_ENTRY)
      num_heap_free++;
    else if (pointers_free_vec[i] == LOCK_ENTRY)
      num_heap_occupied++;
    else {
      FAIL() << "memory leak detected at block " << i << ": neither free nor allocated (value: " << pointers_free_vec[i] << ")";
    }
  }
  // check for duplicates
  l.sort();
  size_t size_before = l.size();
  l.unique();
  size_t size_after = l.size();

  ASSERT_EQ(size_before - size_after, 0);
  std::cout << "num active mutex: " << num_active_mutex << std::endl;
  std::cout << "num occupied: " << num_occupied << "\t num free: " << voxelhasher.getHeapHighFreeCount() << std::endl;
  ASSERT_EQ(voxelhasher.num_sdf_blocks_, num_occupied + voxelhasher.getHeapHighFreeCount());
  std::cout << "num occupied + free: " << num_occupied + voxelhasher.getHeapHighFreeCount() << std::endl;
  std::cout << "num in frustum: " << voxelhasher.current_occupied_blocks_ << std::endl;

  delete[] h_heap;
  delete[] h_hashTable;
}
