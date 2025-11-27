#pragma once
#include "cuda_algebra.cuh"
#include "cuda_matrix.cuh"
#include "cuda_matrix_conversion.cuh"
#include <Eigen/Core>

namespace cupanutils {
  namespace cugeoutils {
    enum CameraModel : int { Pinhole = 0, Spherical = 1 };
    class Camera {
    public:
      __host__ explicit Camera(){};
      __host__ explicit Camera(CUDAMat3 cam_K,
                               const uint rows,
                               const uint cols,
                               const float min_depth,
                               const float max_depth,
                               const CameraModel model = Pinhole) {
        fx_            = cam_K.row0.x;
        fy_            = cam_K.row1.y;
        ifx_           = 1.f / fx_;
        ify_           = 1.f / fy_;
        cx_            = cam_K.row0.z;
        cy_            = cam_K.row1.z;
        rows_          = rows;
        cols_          = cols;
        row_threshold_ = static_cast<int>(rows * 0.5f);
        col_threshold_ = static_cast<int>(cols * 0.5f);
        min_depth_     = min_depth;
        max_depth_     = max_depth;
        hfov_          = 2 * atanf(cols_ / (2.0f * fx_));
        vfov_          = 2 * atanf(rows_ / (2.0f * fy_));
        threads_       = dim3(n_threads_cam, n_threads_cam);
        blocks_        = dim3((cols_ + n_threads_cam - 1) / n_threads_cam, (rows_ + n_threads_cam - 1) / n_threads_cam);
        model_         = model;

        CUDA_CHECK(cudaMalloc((void**) &d_instance_, sizeof(Camera)));
        CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(Camera), cudaMemcpyHostToDevice));
      }

      __host__ ~Camera() {
        if (d_instance_ != nullptr) {
          CUDA_CHECK(cudaFree(d_instance_));
          d_instance_ = nullptr;
        }
      }

      Camera(const Camera&)            = delete;
      Camera& operator=(const Camera&) = delete;
      Camera(Camera&&)                 = delete;
      Camera& operator=(Camera&&)      = delete;

      // clang-format off
      const Camera* deviceInstance() const { return d_instance_; }
      const dim3& blocks() const { return blocks_; }
      const dim3& threads() const { return threads_; }

      __host__ __device__ uint cols() const { return cols_; }
      __host__ __device__ uint rows() const { return rows_; }

      // Camera intrinsics getters
      __host__ __device__ float fx() const { return fx_; }
      __host__ __device__ float fy() const { return fy_; }
      __host__ __device__ float cx() const { return cx_; }
      __host__ __device__ float cy() const { return cy_; }
      __host__ __device__ float hfov() const { return hfov_; }
      __host__ __device__ float vfov() const { return vfov_; }
      __host__ __device__ CameraModel model() const { return model_; }

      void computeCloud(CUDAMatrixf3& point_cloud_img);
      void setDepthImage(CUDAMatrixf& depth_img){ depth_img_ = depth_img; }
      void setCamInWorld(const Eigen::Matrix4f& cam_in_world){ cam_in_world_ = CUDAMatSE3(cam_in_world);  CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(Camera), cudaMemcpyHostToDevice));}
      __host__ __device__ float maxDepth() const { return max_depth_; }
      __host__ __device__ float minDepth() const { return min_depth_; }

      __host__ __device__ const CUDAMatSE3& camInWorld() const { return cam_in_world_; }
      __host__ __device__ const CUDAMatSE3& camInWorldRef() const { return cam_in_world_; }

      // clang-format on

      CUDAMatSE3 cam_in_world_; // Public for device kernel access - use camInWorld() getter from host code

#ifdef __CUDACC__
      __forceinline__ __host__ __device__ float3 inverseProjection(const uint& row, const uint& col, const float d) const {
        float3 point;
        switch (model_) {
          case Pinhole: {
            point = d * make_float3(ifx_ * (col - cx_ - 0.5f), ify_ * (row - cy_ - 0.5f), 1.f);
            break;
          }
          case Spherical: {
            const float az = ifx_ * (col - cx_ - 0.5f);
            const float el = ify_ * (row - cy_ - 0.5f);
            const float s0 = sinf(az);
            const float c0 = cosf(az);
            const float s1 = sinf(el);
            const float c1 = cosf(el);
            point          = d * make_float3(c0 * c1, s0 * c1, s1);
            break;
          }
        }
        return point;
      }

      __forceinline__ __device__ float normalizeDepth(const float depth) const {
        return (depth - min_depth_) / (max_depth_ - min_depth_);
      }

      __forceinline__ __device__ bool isInCameraFrustumApprox(const float3& pw) const {
        const float3 pcam = cam_in_world_.inverse() * pw;

        int2 pimg;
        const bool is_good = projectPointApprox(pcam, pimg);

        if (!is_good)
          return false;
        return true;
      }

      __forceinline__ __device__ float getDepth(const float3& p) const {
        switch (model_) {
          case Pinhole:
            return p.z;
          case Spherical:
            return sqrtf(p.x * p.x + p.y * p.y + p.z * p.z);
          default:
            return 0.f;
        }
      }

      __forceinline__ __host__ __device__ bool projectPoint(const float3& pc, int2& pimg) const {
        switch (model_) {
          case Pinhole: {
            if (pc.z <= min_depth_ || pc.z > max_depth_)
              return false;

            const int row = (fy_ * pc.y / pc.z + cy_) + 0.5f;
            const int col = (fx_ * pc.x / pc.z + cx_) + 0.5f;

            if (row >= 0 && col >= 0 && row < rows_ && col < cols_) {
              pimg = make_int2(row, col);
              return true;
            }
            break;
          }
          case Spherical: {
            const float range = sqrtf(pc.x * pc.x + pc.y * pc.y + pc.z * pc.z);
            if (range < min_depth_ or range > max_depth_)
              return false;

            const float px = atan2f(pc.y, pc.x);
            const float py = asinf(pc.z / range);

            const int row = (fy_ * py + cy_) + 0.5f;
            const int col = (fx_ * px + cx_) + 0.5f;

            if (row >= 0 && col >= 0 && row < rows_ && col < cols_) {
              pimg = make_int2(row, col);
              return true;
            }
            break;
          }
        }
        return false;
      }

      __forceinline__ __host__ __device__ bool projectPointApprox(const float3& pc, int2& pimg) const {
        switch (model_) {
          case Pinhole: {
            if (pc.z <= min_depth_ || pc.z > max_depth_)
              return false;

            const int row = (fy_ * pc.y / pc.z + cy_) + 0.5f;
            const int col = (fx_ * pc.x / pc.z + cx_) + 0.5f;

            if (row >= -row_threshold_ && col >= -col_threshold_ && row < (int) (rows_ + row_threshold_) &&
                col < (int) (cols_ + col_threshold_)) {
              pimg = make_int2(row, col);
              return true;
            }
            break;
          }
          case Spherical: {
            const float range = sqrtf(pc.x * pc.x + pc.y * pc.y + pc.z * pc.z);
            if (range < min_depth_ or range > max_depth_)
              return false;

            const float px = atan2f(pc.y, pc.x);
            const float py = asinf(pc.z / range);

            const int row = (fy_ * py + cy_) + 0.5f;
            const int col = (fx_ * px + cx_) + 0.5f;

            if (row >= -row_threshold_ && col >= -col_threshold_ && row < (int) (rows_ + row_threshold_) &&
                col < (int) (cols_ + col_threshold_)) {
              pimg = make_int2(row, col);
              return true;
            }
            break;
          }
        }
        return false;
      }

#endif
      __forceinline__ void backProject(const Eigen::Vector2f img_coords, Eigen::Vector3f& pw) {
        pw.x() = ifx_ * (img_coords.x() - cx_ - 0.5f);
        pw.y() = ify_ * (img_coords.y() - cy_ - 0.5f);
        pw.z() = 1.f;
      }

    private:
      Camera* d_instance_ = nullptr;
      CUDAMatrixf depth_img_;
      CUDAMatrixuc3 rgb_img_;
      dim3 blocks_, threads_;
      uint rows_, cols_;
      int row_threshold_, col_threshold_; // Signed to allow negative bounds checking
      float fx_, fy_, ifx_, ify_;         // focal lengths and inverse
      float cx_, cy_;                     // principal point
      float hfov_, vfov_;
      float max_depth_, min_depth_;
      CameraModel model_;
    };

  } // namespace cugeoutils
} // namespace cupanutils
