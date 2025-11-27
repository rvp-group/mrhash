#include "camera.cuh"

using namespace cupanutils::cugeoutils;

__global__ void calculateCloudKernel(const CUDAMatrixf* depth_img, const Camera* camera, CUDAMatrixf3* point_cloud_img) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (!depth_img->inside(row, col))
    return;

  const float& depth_val = depth_img->at<1>(row, col);

  if (depth_val <= camera->minDepth() || depth_val > camera->maxDepth())
    return;

  float3 pcam                      = camera->inverseProjection(row, col, depth_val);
  point_cloud_img->at<1>(row, col) = pcam;
}

void Camera::computeCloud(CUDAMatrixf3& point_cloud_img) {
  point_cloud_img = CUDAMatrixf3(rows_, cols_);
  point_cloud_img.fill(make_float3(0.f, 0.f, 0.f), true); // fill only in device
  calculateCloudKernel<<<blocks_, threads_>>>(depth_img_.deviceInstance(), d_instance_, point_cloud_img.deviceInstance());
  CUDA_CHECK(cudaDeviceSynchronize());
}