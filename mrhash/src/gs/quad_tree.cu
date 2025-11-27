#include "quad_tree.cuh"
#include <iostream>

namespace gs {

  __device__ void CUDANode::computeError(const cupanutils::cugeoutils::CUDAMatrixuc3* img, float* error_out, float* shared) {
    const int thread_idx = threadIdx.x;
    const int block_dim  = blockDim.x;

    const int x0    = getOriginX();
    const int y0    = getOriginY();
    const int w     = getWidth();
    const int h     = getHeight();
    const int count = w * h;

    float r_sum = 0.f, g_sum = 0.f, b_sum = 0.f;

    for (int idx = thread_idx; idx < count; idx += block_dim) {
      int local_x = idx % w;
      int local_y = idx / w;
      int x       = x0 + local_x;
      int y       = y0 + local_y;
      uchar3 pix  = img->at<1>(y, x);

      r_sum += static_cast<float>(pix.x);
      g_sum += static_cast<float>(pix.y);
      b_sum += static_cast<float>(pix.z);
    }

    shared[thread_idx]                 = r_sum;
    shared[thread_idx + block_dim]     = g_sum;
    shared[thread_idx + 2 * block_dim] = b_sum;
    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride >>= 1) {
      if (thread_idx < stride) {
        shared[thread_idx] += shared[thread_idx + stride];
        shared[thread_idx + block_dim] += shared[thread_idx + block_dim + stride];
        shared[thread_idx + 2 * block_dim] += shared[thread_idx + 2 * block_dim + stride];
      }
      __syncthreads();
    }

    float r_mean = shared[0] / count;
    float g_mean = shared[block_dim] / count;
    float b_mean = shared[2 * block_dim] / count;

    __syncthreads();

    float r_mse = 0.f, g_mse = 0.f, b_mse = 0.f;

    for (int idx = thread_idx; idx < count; idx += block_dim) {
      int local_x = idx % w;
      int local_y = idx / w;
      int x       = x0 + local_x;
      int y       = y0 + local_y;
      uchar3 pix  = img->at<1>(y, x);

      float r_diff = static_cast<float>(pix.x) - r_mean;
      float g_diff = static_cast<float>(pix.y) - g_mean;
      float b_diff = static_cast<float>(pix.z) - b_mean;

      r_mse += r_diff * r_diff;
      g_mse += g_diff * g_diff;
      b_mse += b_diff * b_diff;
    }

    shared[thread_idx]                 = r_mse;
    shared[thread_idx + block_dim]     = g_mse;
    shared[thread_idx + 2 * block_dim] = b_mse;
    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride >>= 1) {
      if (thread_idx < stride) {
        shared[thread_idx] += shared[thread_idx + stride];
        shared[thread_idx + block_dim] += shared[thread_idx + block_dim + stride];
        shared[thread_idx + 2 * block_dim] += shared[thread_idx + 2 * block_dim + stride];
      }
      __syncthreads();
    }

    if (thread_idx == 0) {
      float r_mse_final = shared[0] / count;
      float g_mse_final = shared[block_dim] / count;
      float b_mse_final = shared[2 * block_dim] / count;

      float error = r_mse_final * 0.2989f + g_mse_final * 0.5870f + b_mse_final * 0.1140f;
      *error_out  = error * (img->cols() * img->rows()) / 90000000.0f;
    }
  }
  __global__ void
  precomputeErrorsKernel(CUDANode* nodes_in, float* errors_out, int num_nodes_in, cupanutils::cugeoutils::CUDAMatrixuc3* img) {
    int idx = blockIdx.x;
    if (idx >= num_nodes_in)
      return;

    extern __shared__ float shared[];

    nodes_in[idx].computeError(img, &errors_out[idx], shared);
  }

  __global__ void subdivideKernel(CUDANode* nodes_in,
                                  CUDANode* nodes_out,
                                  CUDANode* leaves,
                                  const float* errors,
                                  const float threshold,
                                  uint* num_leaves,
                                  uint* num_nodes_out,
                                  const int min_pixel_size,
                                  const int num_nodes_in,
                                  const uint leaves_capacity,
                                  const uint nodes_out_capacity,
                                  uint* overflow_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes_in)
      return;

    CUDANode& node = nodes_in[idx];
    float err      = errors[idx];

    if (err <= threshold) {
      uint leaf_idx = atomicAdd(num_leaves, 1);
      if (leaf_idx < leaves_capacity) {
        leaves[leaf_idx] = node;
      } else {
        // revert increment and record overflow
        atomicSub(num_leaves, 1);
        atomicAdd(overflow_counter, 1);
      }
      return;
    }

    int w  = node.getWidth();
    int h  = node.getHeight();
    int w1 = w / 2;
    int w2 = w - w1;
    int h1 = h / 2;
    int h2 = h - h1;

    if (w1 <= min_pixel_size || h1 <= min_pixel_size) {
      uint leaf_idx = atomicAdd(num_leaves, 1);
      if (leaf_idx < leaves_capacity) {
        leaves[leaf_idx] = node;
      } else {
        atomicSub(num_leaves, 1);
        atomicAdd(overflow_counter, 1);
      }
      return;
    }

    uint out_idx = atomicAdd(num_nodes_out, 4);
    if (out_idx + 3 >= nodes_out_capacity) {
      uint leaf_idx = atomicAdd(num_leaves, 1);
      if (leaf_idx < leaves_capacity) {
        leaves[leaf_idx] = node;
      } else {
        atomicSub(num_leaves, 1);
      }
      atomicAdd(overflow_counter, 1);
      return;
    }

    nodes_out[out_idx + 0] = CUDANode(node.getOriginX(), node.getOriginY(), w1, h1, out_idx + 0);
    nodes_out[out_idx + 1] = CUDANode(node.getOriginX(), node.getOriginY() + h1, w1, h2, out_idx + 1);
    nodes_out[out_idx + 2] = CUDANode(node.getOriginX() + w1, node.getOriginY(), w2, h1, out_idx + 2);
    nodes_out[out_idx + 3] = CUDANode(node.getOriginX() + w1, node.getOriginY() + h1, w2, h2, out_idx + 3);
  }

  void CUDAQTree::subdivide() {
    int num_nodes_in = 1;
    CUDA_CHECK(cudaMemset(d_num_leaves_, 0, sizeof(uint)));
    CUDA_CHECK(cudaMemset(d_overflow_count_, 0, sizeof(uint)));

    float* d_errors;
    CUDA_CHECK(cudaMalloc(&d_errors, max_num_qtree_nodes * sizeof(float)));

    uint overflow_total = 0;
    while (num_nodes_in > 0) {
      CUDA_CHECK(cudaMemset(d_num_nodes_out_, 0, sizeof(uint)));

      size_t shared_memory_size = 3 * n_threads_subdivide * sizeof(float);
      precomputeErrorsKernel<<<num_nodes_in, n_threads_subdivide, shared_memory_size>>>(
        d_nodes_in_, d_errors, num_nodes_in, img_.deviceInstance());
      int num_blocks = (num_nodes_in + n_threads_subdivide - 1) / n_threads_subdivide;

      subdivideKernel<<<num_blocks, n_threads_subdivide>>>(d_nodes_in_,
                                                           d_nodes_out_,
                                                           d_leaves_,
                                                           d_errors,
                                                           threshold_,
                                                           d_num_leaves_,
                                                           d_num_nodes_out_,
                                                           min_pixel_size_,
                                                           num_nodes_in,
                                                           leaves_capacity_,
                                                           nodes_out_capacity_,
                                                           d_overflow_count_);

      CUDA_CHECK(cudaDeviceSynchronize());

      uint num_nodes_out;
      CUDA_CHECK(cudaMemcpy(&num_nodes_out, d_num_nodes_out_, sizeof(uint), cudaMemcpyDeviceToHost));
      uint overflow_iter = 0;
      CUDA_CHECK(cudaMemcpy(&overflow_iter, d_overflow_count_, sizeof(uint), cudaMemcpyDeviceToHost));
      overflow_total += overflow_iter;
      // reset for next iteration
      CUDA_CHECK(cudaMemset(d_overflow_count_, 0, sizeof(uint)));

      if (num_nodes_out == 0)
        break;

      std::swap(d_nodes_in_, d_nodes_out_);
      num_nodes_in = static_cast<int>(num_nodes_out);
    }

    CUDA_CHECK(cudaFree(d_errors));

    if (overflow_total > 0) {
      std::cerr << "[CUDAQTree] Warning: Subdivision capacity overflow occurred " << overflow_total
                << " times. Consider increasing leaves capacity (current: " << leaves_capacity_
                << ") or nodes_out capacity (current: " << nodes_out_capacity_ << "), or relaxing threshold/min_pixel_size.\n";
    }
  }
} // namespace gs
