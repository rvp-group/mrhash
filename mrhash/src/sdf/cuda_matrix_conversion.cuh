#pragma once

#include "cuda_algebra.cuh"
#include <Eigen/Core>

namespace cupanutils {
  namespace cugeoutils {

    static CUDAMat3 Eig2CUDA(const Eigen::Matrix3f& mat) {
      CUDAMat3 m(mat);
      return m;
    }

    static CUDAMatSE3 Eig2CUDA(const Eigen::Matrix4f& mat) {
      CUDAMatSE3 m(mat);
      return m;
    }

    static float3 Eig2CUDA(const Eigen::Vector3f& vec) {
      return make_float3(vec.x(), vec.y(), vec.z());
    }

    static int3 Eig2CUDA(const Eigen::Vector3i& vec) {
      return make_int3(vec.x(), vec.y(), vec.z());
    }

    static Eigen::Vector3f CUDA2Eig(const float3& vec) {
      return Eigen::Vector3f(vec.x, vec.y, vec.z);
    }

    static Eigen::Vector3i CUDA2Eig(const int3& vec) {
      return Eigen::Vector3i(vec.x, vec.y, vec.z);
    }

    static Eigen::Vector3i CUDA2Eig(const uint3& vec) {
      return Eigen::Vector3i(vec.x, vec.y, vec.z);
    }

    static Eigen::Matrix3f CUDA2Eig(const CUDAMat3& mat) {
      Eigen::Matrix3f m;
      m(0, 0) = mat.row0.x;
      m(0, 1) = mat.row0.y;
      m(0, 2) = mat.row0.z;
      m(1, 0) = mat.row1.x;
      m(1, 1) = mat.row1.y;
      m(1, 2) = mat.row1.z;
      m(2, 0) = mat.row2.x;
      m(2, 1) = mat.row2.y;
      m(2, 2) = mat.row2.z;
      return m;
    }

    static Eigen::Matrix4f CUDA2Eig(const CUDAMatSE3& mat) {
      Eigen::Matrix4f m;
      m.setIdentity(); // last column not written mat does not have it
      m(0, 0) = mat.rotation.row0.x;
      m(0, 1) = mat.rotation.row0.y;
      m(0, 2) = mat.rotation.row0.z;
      m(1, 0) = mat.rotation.row1.x;
      m(1, 1) = mat.rotation.row1.y;
      m(1, 2) = mat.rotation.row1.z;
      m(2, 0) = mat.rotation.row2.x;
      m(2, 1) = mat.rotation.row2.y;
      m(2, 2) = mat.rotation.row2.z;
      m(0, 3) = mat.translation.x;
      m(1, 3) = mat.translation.y;
      m(2, 3) = mat.translation.z;
      return m;
    }

  } // namespace cugeoutils
} // namespace cupanutils
