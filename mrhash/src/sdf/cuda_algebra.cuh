#pragma once
#include "cuda_math.cuh"
#include <ostream>

namespace cupanutils {
  namespace cugeoutils {

    // 3x3 mat for CUDA computation
    struct CUDAMat3 {
      // def constructor, leaves the matrix uninitialized
      __forceinline__ __host__ __device__ CUDAMat3() {
      }

      // copy constructor
      __forceinline__ __host__ CUDAMat3(const CUDAMat3& other) : row0(other.row0), row1(other.row1), row2(other.row2) {
      }

      // constructs the matrix from an array-like matrix object (works with Eigen matrices)
      template <typename T>
      __host__ explicit CUDAMat3(const T& matrix) {
        row0.x = matrix(0, 0);
        row0.y = matrix(0, 1);
        row0.z = matrix(0, 2);
        row1.x = matrix(1, 0);
        row1.y = matrix(1, 1);
        row1.z = matrix(1, 2);
        row2.x = matrix(2, 0);
        row2.y = matrix(2, 1);
        row2.z = matrix(2, 2);
      }

      // constructs the matrix from an array object, row major
      __host__ explicit CUDAMat3(float* arr) {
        row0.x = arr[0];
        row0.y = arr[1];
        row0.z = arr[2];
        row1.x = arr[3];
        row1.y = arr[4];
        row1.z = arr[5];
        row2.x = arr[6];
        row2.y = arr[7];
        row2.z = arr[8];
      }

      __forceinline__ __device__ CUDAMat3 transpose() const {
        CUDAMat3 res;
        res.row0.x = this->row0.x;
        res.row0.y = this->row1.x;
        res.row0.z = this->row2.x;
        res.row1.x = this->row0.y;
        res.row1.y = this->row1.y;
        res.row1.z = this->row2.y;
        res.row2.x = this->row0.z;
        res.row2.y = this->row1.z;
        res.row2.z = this->row2.z;
        return res;
      }

      // assignment operator
      __forceinline__ __host__ __device__ CUDAMat3& operator=(const CUDAMat3& other) {
        this->row0 = other.row0;
        this->row1 = other.row1;
        this->row2 = other.row2;
        return *this;
      }

// define operators only in device code
// we have eigen in host
#ifdef __CUDACC__
      // matrix-vector multiplication
      __forceinline__ __device__ float3 operator*(const float3& point) const {
        return make_float3(row0.x * point.x + row0.y * point.y + row0.z * point.z,
                           row1.x * point.x + row1.y * point.y + row1.z * point.z,
                           row2.x * point.x + row2.y * point.y + row2.z * point.z);
      }

      // matrix-matrix multiplication
      __forceinline__ __device__ CUDAMat3 operator*(const CUDAMat3& mat) const {
        // clang-format off
      CUDAMat3 res;
      res.row0.x = this->row0.x * mat.row0.x + this->row0.y * mat.row1.x + this->row0.z * mat.row2.x; 
      res.row0.y = this->row0.x * mat.row0.y + this->row0.y * mat.row1.y + this->row0.z * mat.row2.y; 
      res.row0.z = this->row0.x * mat.row0.z + this->row0.y * mat.row1.z + this->row0.z * mat.row2.z;
      res.row1.x = this->row1.x * mat.row0.x + this->row1.y * mat.row1.x + this->row1.z * mat.row2.x; 
      res.row1.y = this->row1.x * mat.row0.y + this->row1.y * mat.row1.y + this->row1.z * mat.row2.y; 
      res.row1.z = this->row1.x * mat.row0.z + this->row1.y * mat.row1.z + this->row1.z * mat.row2.z;
      res.row2.x = this->row2.x * mat.row0.x + this->row2.y * mat.row1.x + this->row2.z * mat.row2.x; 
      res.row2.y = this->row2.x * mat.row0.y + this->row2.y * mat.row1.y + this->row2.z * mat.row2.y; 
      res.row2.z = this->row2.x * mat.row0.z + this->row2.y * mat.row1.z + this->row2.z * mat.row2.z;
        // clang-format on
        return res;
      }
#endif

      // row-wise storage.
      float3 row0;
      float3 row1;
      float3 row2;
    };

    struct CUDAMatSE3 {
      // def constructor, leaves the matrix uninitialized
      __forceinline__ __host__ __device__ CUDAMatSE3() {
      }

      // copy constructor
      __forceinline__ __host__ CUDAMatSE3(const CUDAMatSE3& other) : rotation(other.rotation), translation(other.translation) {
      }

      template <typename T>
      __host__ explicit CUDAMatSE3(const T& matrix) {
        rotation.row0.x = matrix(0, 0);
        rotation.row0.y = matrix(0, 1);
        rotation.row0.z = matrix(0, 2);
        rotation.row1.x = matrix(1, 0);
        rotation.row1.y = matrix(1, 1);
        rotation.row1.z = matrix(1, 2);
        rotation.row2.x = matrix(2, 0);
        rotation.row2.y = matrix(2, 1);
        rotation.row2.z = matrix(2, 2);
        translation.x   = matrix(0, 3);
        translation.y   = matrix(1, 3);
        translation.z   = matrix(2, 3);
      }

      // assignment operator
      __forceinline__ __host__ __device__ CUDAMatSE3& operator=(const CUDAMatSE3& other) {
        this->rotation    = other.rotation;
        this->translation = other.translation;
        return *this;
      }

// define operators only in device code
// we have eigen in host
#ifdef __CUDACC__

      __forceinline__ __device__ CUDAMatSE3 inverse() const {
        CUDAMatSE3 res;
        res.rotation    = this->rotation.transpose();
        res.translation = res.rotation * this->translation;
        res.translation = -res.translation;
        return res;
      }

      // matrix-vector multiplication.
      __forceinline__ __device__ float3 operator*(const float3& point) const {
        return rotation * point + translation;
      }

      __forceinline__ __device__ CUDAMatSE3 operator*(const CUDAMatSE3& other) const {
        CUDAMatSE3 res;
        res.rotation    = this->rotation * other.rotation;
        res.translation = this->rotation * other.translation + this->translation;
        return res;
      }

#endif

      friend std::ostream& operator<<(std::ostream& out, const CUDAMatSE3& m) {
        return out << m.rotation.row0.x << " " << m.rotation.row0.y << " " << m.rotation.row0.z << " " << m.translation.x << " "
                   << m.rotation.row1.x << " " << m.rotation.row1.y << " " << m.rotation.row1.z << " " << m.translation.y << " "
                   << m.rotation.row2.x << " " << m.rotation.row2.y << " " << m.rotation.row2.z << " " << m.translation.z << " "
                   << 0 << " " << 0 << " " << 0 << " " << 1 << std::endl;
      }

      CUDAMat3 rotation;
      float3 translation;
    };

  } // namespace cugeoutils
} // namespace cupanutils
