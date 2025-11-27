#ifndef GS_QUAD_TREE_HPP
#define GS_QUAD_TREE_HPP

#include "../sdf/params.h"
#include <cuda_matrix.cuh>
#include <opencv2/opencv.hpp>

namespace gs {

  struct CUDANode {
    __host__ __device__ CUDANode() : x0_(0), y0_(0), width_(0), height_(0), id_(-1) {
    }
    __host__ __device__ CUDANode(int x0, int y0, int width, int height, int id) :
      x0_(x0), y0_(y0), width_(width), height_(height), id_(id) {
    }
    __host__ __device__ CUDANode(const CUDANode& other) :
      x0_(other.x0_), y0_(other.y0_), width_(other.width_), height_(other.height_), id_(other.id_) {
    }
    __host__ __device__ inline bool operator<(const CUDANode& other) const {
      return id_ < other.id_;
    }
    __host__ __device__ inline int getOriginX() const {
      return x0_;
    }
    __host__ __device__ inline int getOriginY() const {
      return y0_;
    }
    __host__ __device__ inline int getWidth() const {
      return width_;
    }
    __host__ __device__ inline int getHeight() const {
      return height_;
    }
    __host__ __device__ inline int getId() const {
      return id_;
    }

    __host__ __device__ inline void setId(int id) {
      id_ = id;
    }

    __device__ void computeError(const cupanutils::cugeoutils::CUDAMatrixuc3* img, float* error_out, float* shared);

  private:
    int x0_;
    int y0_;
    int width_;
    int height_;
    int id_;
  };

  class CUDAQTree {
  public:
    __host__ explicit CUDAQTree(float threshold,
                                int min_pixel_size,
                                CUDANode* d_leaves,
                                cupanutils::cugeoutils::CUDAMatrixuc3& img) :
      threshold_(threshold),
      min_pixel_size_(min_pixel_size),
      d_leaves_(d_leaves),
      img_(img),
      root_(0, 0, img.cols(), img.rows(), 0) {
      CUDA_CHECK(cudaMalloc((void**) &d_nodes_in_, max_num_qtree_nodes * sizeof(CUDANode)));
      CUDA_CHECK(cudaMemcpy(d_nodes_in_, &root_, sizeof(CUDANode), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMalloc((void**) &d_nodes_out_, max_num_qtree_nodes * sizeof(CUDANode)));
      CUDA_CHECK(cudaMalloc((void**) &d_num_nodes_out_, sizeof(uint)));
      CUDA_CHECK(cudaMalloc((void**) &d_num_leaves_, sizeof(uint)));
      CUDA_CHECK(cudaMemset(d_num_leaves_, 0, sizeof(uint)));
      leaves_capacity_    = qtree_leaves_capacity;
      nodes_out_capacity_ = max_num_qtree_nodes; // reuse global node capacity
      CUDA_CHECK(cudaMalloc((void**) &d_overflow_count_, sizeof(uint)));
      CUDA_CHECK(cudaMemset(d_overflow_count_, 0, sizeof(uint)));
    }
    ~CUDAQTree() {
      CUDA_CHECK(cudaFree(d_nodes_in_));
      CUDA_CHECK(cudaFree(d_num_leaves_));
      CUDA_CHECK(cudaFree(d_nodes_out_));
      CUDA_CHECK(cudaFree(d_num_nodes_out_));
      CUDA_CHECK(cudaFree(d_overflow_count_));
    }

    inline std::vector<CUDANode> getAllNodes() {
      CUDA_CHECK(cudaMemcpy(&num_leaves_, d_num_leaves_, sizeof(uint), cudaMemcpyDeviceToHost));
      leaves_.resize(num_leaves_);
      CUDA_CHECK(cudaMemcpy(leaves_.data(), d_leaves_, sizeof(CUDANode) * num_leaves_, cudaMemcpyDeviceToHost));
      return leaves_;
    }

    inline uint getNumLeaves() {
      CUDA_CHECK(cudaMemcpy(&num_leaves_, d_num_leaves_, sizeof(uint), cudaMemcpyDeviceToHost));
      return num_leaves_;
    }

    void subdivide();

  private:
    float threshold_;
    int min_pixel_size_;
    cupanutils::cugeoutils::CUDAMatrixuc3& img_;
    CUDANode root_;
    CUDANode* d_nodes_in_    = nullptr;
    CUDANode* d_nodes_out_   = nullptr;
    uint* d_num_nodes_out_   = nullptr;
    uint num_leaves_         = 0;
    uint* d_num_leaves_      = nullptr;
    CUDANode* d_leaves_      = nullptr;
    uint leaves_capacity_    = 0;
    uint nodes_out_capacity_ = 0;
    uint* d_overflow_count_  = nullptr;
    std::vector<CUDANode> leaves_;
  };

} // namespace gs

#endif // GS_QUAD_TREE_HPP
