#include <sdf/cuda_matrix_conversion.cuh>
#include <sdf/cuda_utils.cuh>
#include <iostream>

#include "test_utils.cuh"

using namespace cupanutils::cugeoutils;

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

__global__ void vec_operations_kernel(const float3 vec1, const float3 vec2, float3* dst) {
  float3 res                = cross(vec1, vec2);
  res                       = res + vec1;
  res                       = vec2 + res;
  res                       = res - vec1;
  res                       = vec2 - res;
  float norm_result         = norm(res);
  float squared_norm_result = squaredEuclideanDistance(res, vec1);
  float dot_result          = dot(vec1, vec2);
  float scalar              = (norm_result - squared_norm_result - dot_result) * 1e-10;
  res                       = scalar * res;
  res                       = res * scalar;
  // copying back
  dst->x = res.x;
  dst->y = res.y;
  dst->z = res.z;
}

__global__ void mat_operations_kernel(const CUDAMat3 mat, const CUDAMatSE3 iso, const float3 vec, float3* dst) {
  CUDAMat3 res_mat   = mat * mat;
  CUDAMatSE3 res_iso = iso * iso;
  float3 res1        = res_mat.transpose() * vec;
  float3 res2        = res_iso.inverse() * vec;
  float3 res3        = res_iso * res1;

  // copying back
  dst->x = res3.x;
  dst->y = res3.y;
  dst->z = res3.z;
}

TEST(ALGEBRA, VectorOperators) {
  // checking some vec operators
  float3 vec1 = make_float3(1e-6, 52.45, 6700.23);
  float3 vec2 = make_float3(49.5, 65.99, 1e-3);

  float3* res_device;
  CUDA_CHECK(cudaMalloc(&res_device, sizeof(float3)));
  vec_operations_kernel<<<1, 1>>>(vec1, vec2, res_device);
  CUDA_CHECK(cudaDeviceSynchronize());

  float3* res_host = new float3();
  CUDA_CHECK(cudaMemcpy(res_host, res_device, sizeof(float3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(res_device));

  // comparing with eigen operators
  Eigen::Vector3f vec1_eigen(vec1.x, vec1.y, vec1.z);
  Eigen::Vector3f vec2_eigen(vec2.x, vec2.y, vec2.z);
  auto res_eigen            = vec1_eigen.cross(vec2_eigen);
  res_eigen                 = res_eigen + vec1_eigen;
  res_eigen                 = vec2_eigen + res_eigen;
  res_eigen                 = res_eigen - vec1_eigen;
  res_eigen                 = vec2_eigen - res_eigen;
  float norm_result         = res_eigen.norm();
  float squared_norm_result = (res_eigen - vec1_eigen).squaredNorm();
  float dot_result          = vec1_eigen.dot(vec2_eigen);
  float scalar              = (norm_result - squared_norm_result - dot_result) * 1e-10;
  res_eigen                 = scalar * res_eigen;
  res_eigen                 = res_eigen * scalar;

  ASSERT_FLOAT_EQ(res_eigen.x(), res_host->x);
  ASSERT_FLOAT_EQ(res_eigen.y(), res_host->y);
  ASSERT_FLOAT_EQ(res_eigen.z(), res_host->z);

  std::cout << res_host->x << " " << res_host->y << " " << res_host->z << std::endl;
  std::cout << res_eigen.x() << " " << res_eigen.y() << " " << res_eigen.z() << std::endl;

  delete res_host;

  Eigen::Vector3i vec10_eigen(2.5, 2.5, 2.5);
  int3 dvec10 = Eig2CUDA(vec10_eigen);

  ASSERT_FLOAT_EQ(vec10_eigen.x(), dvec10.x);
  ASSERT_FLOAT_EQ(vec10_eigen.y(), dvec10.y);
  ASSERT_FLOAT_EQ(vec10_eigen.z(), dvec10.z);

  Eigen::Vector3f vec11_eigen(2.5, 2.5, 2.5);
  float3 dvec11 = Eig2CUDA(vec11_eigen);

  ASSERT_FLOAT_EQ(vec11_eigen.x(), dvec11.x);
  ASSERT_FLOAT_EQ(vec11_eigen.y(), dvec11.y);
  ASSERT_FLOAT_EQ(vec11_eigen.z(), dvec11.z);
}

TEST(ALGEBRA, MatrixOperators) {
  // checking some vec operators
  float3 vec1 = make_float3(47.32, 52.45, 150.23);
  Eigen::Quaternionf quat(0.5, 0.5, 0.5, 0.5);
  quat.normalize();
  Eigen::Matrix3f rot = quat.toRotationMatrix();
  CUDAMat3 curot(rot);
  Eigen::Isometry3f iso = Eigen::Isometry3f::Identity();
  iso.linear()          = rot;
  iso.translation() << 0.1, 0.2, 0.3;
  CUDAMatSE3 cuiso(iso.matrix());

  float3* res_device;
  CUDA_CHECK(cudaMalloc(&res_device, sizeof(float3)));
  mat_operations_kernel<<<1, 1>>>(curot, cuiso, vec1, res_device);
  CUDA_CHECK(cudaDeviceSynchronize());

  float3* res_host = new float3();
  CUDA_CHECK(cudaMemcpy(res_host, res_device, sizeof(float3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(res_device));

  Eigen::Vector3f vec1_eigen(vec1.x, vec1.y, vec1.z);
  Eigen::Matrix3f res_mat_eigen   = rot * rot;
  Eigen::Isometry3f res_iso_eigen = iso * iso;
  Eigen::Vector3f res1_eigen      = res_mat_eigen.transpose() * vec1_eigen;
  Eigen::Vector3f res2_eigen      = res_iso_eigen.inverse() * vec1_eigen;
  Eigen::Vector3f res3_eigen      = res_iso_eigen * res1_eigen;

  ASSERT_FLOAT_EQ(res3_eigen.x(), res_host->x);
  ASSERT_FLOAT_EQ(res3_eigen.y(), res_host->y);
  ASSERT_FLOAT_EQ(res3_eigen.z(), res_host->z);

  std::cout << res_host->x << " " << res_host->y << " " << res_host->z << std::endl;
  std::cout << res3_eigen.x() << " " << res3_eigen.y() << " " << res3_eigen.z() << std::endl;

  delete res_host;

  Eigen::Matrix4f identity_eigen = Eigen::Matrix4f::Identity();
  CUDAMatSE3 identity_cuda(identity_eigen);
  Eigen::Matrix4f new_identity_eigen = CUDA2Eig(identity_cuda);

  ASSERT_FLOAT_EQ((identity_eigen - new_identity_eigen).norm(), 0.f);

  Eigen::Matrix3f mat = Eigen::Matrix3f::Identity() * 5.f;
  CUDAMat3 mat_cuda(mat);
  Eigen::Matrix3f new_mat = CUDA2Eig(mat_cuda);

  ASSERT_FLOAT_EQ((mat - new_mat).norm(), 0.f);
}