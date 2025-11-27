#pragma once
#include "cuda_utils.cuh"
#include <cstring>

namespace cupanutils {
  namespace cugeoutils {

    enum MemType { Host = 0, Device = 1 };

    /**
     * DualMatrix_ class to handle easily matrix type bouncing from GPU to RAM
     */
    template <typename CellType_>
    class DualMatrix_ {
    public:
      using ThisType = DualMatrix_<CellType_>;
      using CellType = CellType_;

      __host__ explicit DualMatrix_(const uint32_t rows, const uint32_t cols) {
        CUDA_CHECK(cudaMalloc((void**) &device_instance_, sizeof(ThisType)));
        resize(rows, cols);
      }

      __host__ explicit DualMatrix_() : buffers_{nullptr, nullptr}, device_instance_(nullptr), rows_(0), cols_(0), capacity_(0) {
        CUDA_CHECK(cudaMalloc((void**) &device_instance_, sizeof(ThisType)));
        CUDA_CHECK(cudaMemcpy(device_instance_, this, sizeof(ThisType), cudaMemcpyHostToDevice));
      }

      __host__ DualMatrix_(const DualMatrix_& src_) : DualMatrix_(src_.rows_, src_.cols_) {
        memcpy(buffers_[Host], src_.buffers_[Host], sizeof(CellType) * capacity_);
        CUDA_CHECK(cudaMemcpy(buffers_[Device], src_.buffers_[Device], sizeof(CellType) * capacity_, cudaMemcpyDeviceToDevice));
      }

      DualMatrix_& operator=(const DualMatrix_& src_) {
        resize(src_.rows_, src_.cols_);
        memcpy(buffers_[Host], src_.buffers_[Host], sizeof(CellType) * capacity_);
        CUDA_CHECK(cudaMemcpy(buffers_[Device], src_.buffers_[Device], sizeof(CellType) * capacity_, cudaMemcpyDeviceToDevice));
        return *this;
      }

      ~DualMatrix_() {
        if (device_instance_)
          cudaFree(device_instance_);
        if (buffers_[Host])
          delete[] buffers_[Host];
        if (buffers_[Device])
          cudaFree(buffers_[Device]);
      }

      __host__ inline void resize(const uint32_t rows, const uint32_t cols) {
        // if size is ok, do nothing
        if (rows == rows_ && cols == cols_)
          return;
        rows_ = rows;
        cols_ = cols;
        sync_();
      }

      // clang-format off
    __host__ __device__ inline ThisType* deviceInstance() { return device_instance_; }
    __host__ inline ThisType* deviceInstance() const { return device_instance_; } 
    __host__ void fill(const CellType& value_, const bool device_only_ = false);
    __host__ inline uint32_t nThreads() const { return n_threads_; }
    __host__ inline uint32_t nBlocks() const { return n_blocks_; }
    __host__ __device__ inline uint32_t rows() const { return rows_; }
    __host__ __device__ inline uint32_t cols() const { return cols_; }
    __host__ __device__ inline uint32_t size() const { return capacity_; }
    __host__ __device__ inline bool empty() const { return capacity_ == 0; };
      // clang-format on

      __host__ __device__ inline bool inside(const uint32_t row, const uint32_t col) const {
        return row < rows_ && col < cols_;
      }

      __host__ __device__ inline bool onBorder(const uint32_t row, const uint32_t col) const {
        return row == 0 || col == 0 || row == rows_ - 1 || col == cols_ - 1;
      }

      template <int MemType = 0>
      __host__ __device__ inline const CellType& at(const uint32_t index_) const {
        return buffers_[MemType][index_];
      }

      template <int MemType = 0>
      __host__ __device__ inline CellType& at(const uint32_t index_) {
        return buffers_[MemType][index_];
      }

      template <int MemType = 0>
      __host__ __device__ inline const CellType& at(const uint32_t row, const uint32_t col) const {
        return buffers_[MemType][row * cols_ + col];
      }

      template <int MemType = 0>
      __host__ __device__ inline CellType& at(const uint32_t row, const uint32_t col) {
        return buffers_[MemType][row * cols_ + col];
      }

      template <int MemType = 0>
      __host__ __device__ inline CellType& operator()(const uint32_t row, const uint32_t col) {
        return buffers_[MemType][row * cols_ + col];
      }

      template <int MemType = 0>
      __host__ __device__ inline CellType& operator[](const uint32_t index_) {
        return buffers_[MemType][index_];
      }

      template <int MemType = 0>
      __host__ __device__ inline const CellType& operator[](const uint32_t index_) const {
        return buffers_[MemType][index_];
      }

      template <int MemType = 0>
      __host__ __device__ inline const CellType* data() const {
        return buffers_[MemType];
      }

      template <int MemType = 0>
      __host__ __device__ inline CellType* data() {
        return buffers_[MemType];
      }

      // copy whole device buffer to host, for debugging at the moment
      __host__ inline void toHost() {
        CUDA_CHECK(cudaMemcpy(buffers_[Host], buffers_[Device], sizeof(CellType) * capacity_, cudaMemcpyDeviceToHost));
      }

      __host__ inline void toDevice() {
        CUDA_CHECK(cudaMemcpy(buffers_[Device], buffers_[Host], sizeof(CellType) * capacity_, cudaMemcpyHostToDevice));
      }

      __host__ inline void clearDeviceBuffer() {
        if (buffers_[Device])
          cudaFree(buffers_[Device]);
      }

    protected:
      inline void sync_() {
        if (capacity_ == (uint32_t) (rows_ * cols_)) {
          copyHeader_();
          return;
        }

        if (buffers_[Device]) {
          cudaFree(buffers_[Device]);
          buffers_[Device] = nullptr;
        }

        if (buffers_[Host]) {
          delete[] buffers_[Host];
          buffers_[Host] = nullptr;
        }
        capacity_ = (uint32_t) (rows_ * cols_);
        if (capacity_) {
          buffers_[Host] = new CellType[capacity_];
          CUDA_CHECK(cudaMalloc((void**) &buffers_[Device], sizeof(CellType) * capacity_));
          CUDA_CHECK(cudaMemcpy(buffers_[Device], buffers_[Host], sizeof(CellType) * capacity_, cudaMemcpyHostToDevice));
        }

        copyHeader_();
      }

      inline void copyHeader_() {
        n_threads_ = n_threads;
        n_blocks_  = (capacity_ + n_threads_ - 1) / n_threads_;
        // once class fields are populated copy ptr on device
        CUDA_CHECK(cudaMemcpy(device_instance_, this, sizeof(ThisType), cudaMemcpyHostToDevice));
      }

      CellType* buffers_[2]      = {nullptr, nullptr};
      ThisType* device_instance_ = nullptr;
      uint32_t rows_             = 0;
      uint32_t cols_             = 0;
      uint32_t capacity_         = 0;
      uint32_t n_threads_        = 0;
      uint32_t n_blocks_         = 0;
    };

#ifdef __CUDACC__

    template <typename CellType_>
    __global__ void fill_kernel(CellType_* data_device, const CellType_ value, const uint32_t capacity) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      if (tid < capacity)
        data_device[tid] = value;
    }

#endif

    template <typename CellType_>
    void DualMatrix_<CellType_>::fill(const CellType_& value, const bool device_only) {
#ifdef __CUDACC__
      fill_kernel<<<n_blocks_, n_threads_>>>(buffers_[Device], value, capacity_);
      CUDA_CHECK(cudaDeviceSynchronize());
#else
      for (size_t i = 0; i < capacity_; ++i)
        buffers_[Host][i] = value;
#endif
      if (!device_only)
        toHost();
    }

    using CUDAMatrixf   = DualMatrix_<float>;
    using CUDAMatrixf3  = DualMatrix_<float3>;
    using CUDAMatrixb   = DualMatrix_<bool>;
    using CUDAMatrixuc3 = DualMatrix_<uchar3>;

    using CUDAVectorf  = DualMatrix_<float>;
    using CUDAVectorf3 = DualMatrix_<float3>;
    using CUDAVectorf4 = DualMatrix_<float4>;

    using CUDAMatrixi  = DualMatrix_<int>;
    using CUDAMatrixi3 = DualMatrix_<int3>;

    using CUDAMatrixu = DualMatrix_<unsigned int>;
    using CUDAVectoru = DualMatrix_<unsigned int>;

    using CUDAMatrixu64 = DualMatrix_<unsigned long long>;

  } // namespace cugeoutils
} // namespace cupanutils
