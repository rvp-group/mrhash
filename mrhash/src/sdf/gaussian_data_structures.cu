#include "gaussian_data_structures.cuh"
namespace cupanutils {
  namespace cugeoutils {
    template <typename T>
    __global__ void processNodesKernel(const gs::CUDANode* nodes,
                                       const CUDAMatrixuc3* rgb_img,
                                       const CUDAMatrixf* depth_img,
                                       const size_t* num_qtree_nodes,
                                       gs::Point* positions,
                                       gs::Color* colors,
                                       float* scales,
                                       uint* num_valid_qtree_nodes,
                                       const Camera* camera,
                                       const VoxelContainer<T>* container) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= num_qtree_nodes[0])
        return;

      const gs::CUDANode& node = nodes[idx];

      // Compute center in image plane (float2)
      float2 p2d;
      p2d.x = node.getOriginX() + 0.5f * node.getWidth();
      p2d.y = node.getOriginY() + 0.5f * node.getHeight();

      // Convert to pixel indices
      const int px = static_cast<int>(p2d.x + 0.5f);
      const int py = static_cast<int>(p2d.y + 0.5f);

      if (px < 0 || py < 0 || px >= camera->cols() || py >= camera->rows())
        return;

      const float depth_value = depth_img->at<1>(py, px);
      if (depth_value < camera->minDepth())
        return;

      // Inverse projection to camera space
      const float3 center = camera->camInWorld() * camera->inverseProjection(py, px, depth_value);
      const uchar weight  = container->getVoxel(center).weight;
      if (weight != 1)
        return;

      const float half_w = 0.5f * node.getWidth();
      const float half_h = 0.5f * node.getHeight();
      const float scale  = (depth_value * sqrtf(half_w * half_w + half_h * half_h)) / camera->fx();
      if (scale <= 0.0f)
        return;

      uint valid_idx = atomicAdd(&num_valid_qtree_nodes[0], 1);

      positions[valid_idx] = gs::Point{center.x, center.y, center.z};
      scales[valid_idx]    = scale;

      const auto& rgb   = rgb_img->at<1>(py, px);
      colors[valid_idx] = {rgb.x, rgb.y, rgb.z};
    }

    template <typename T>
    void GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::checkNodes(const Camera& camera,
                                                                                        const VoxelContainer<T>& container,
                                                                                        const CUDAMatrixuc3& rgb_img,
                                                                                        const CUDAMatrixf& depth_img) {
      int threads_per_block = 256;
      int num_blocks        = (num_qtree_nodes_ + threads_per_block - 1) / threads_per_block;

      processNodesKernel<<<num_blocks, threads_per_block>>>(d_qtree_nodes_,
                                                            rgb_img.deviceInstance(),
                                                            depth_img.deviceInstance(),
                                                            d_num_qtree_nodes_,
                                                            d_positions_,
                                                            d_colors_,
                                                            d_scales_,
                                                            d_num_valid_qtree_nodes_,
                                                            camera.deviceInstance(),
                                                            container.d_instance_);

      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(&num_valid_qtree_nodes_, d_num_valid_qtree_nodes_, sizeof(uint), cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaMemset(d_num_qtree_nodes_, 0, sizeof(size_t)));
      CUDA_CHECK(cudaMemset(d_num_valid_qtree_nodes_, 0, sizeof(uint)));
    }
  } // namespace cugeoutils
} // namespace cupanutils
