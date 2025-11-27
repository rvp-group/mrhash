#include <opencv2/opencv.hpp>

#include <sdf/cuda_matrix.cuh>
#include <sdf/streamer.cuh>

#include "test_utils.cuh"

using namespace cupanutils::cugeoutils;

constexpr float default_depth                              = 1.f;
constexpr int hash_num_buckets                             = 250000;
constexpr int num_sdf_blocks                               = 500000;
constexpr float max_integration_distance                   = 5;
constexpr float sdf_truncation                             = 0.02;
constexpr float sdf_truncation_scale                       = 0.01;
constexpr int integration_weight_sample                    = 3;
constexpr float virtual_voxel_size                         = 0.005;
constexpr float min_depth                                  = 0;
constexpr float max_depth                                  = 5;
constexpr float max_radius_for_stream                      = 3;
constexpr int n_frames_invalidate_voxels                   = 10;
constexpr int max_num_sdf_block_integrate_from_global_hash = 10000;

constexpr uchar min_weight_threshold = 0;

constexpr float sdf_var_threshold = 0.f;
constexpr bool projective_sdf     = true;

constexpr float translation_step = 1.5;
constexpr int steps              = 100; // to discretize angle path

constexpr int rows = 300;
constexpr int cols = 300;

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(STREAMER, SingleStream) {
  srand(time(NULL));

  // dummy depth
  CUDAMatrixf depth_img(rows, cols);
  depth_img.fill(default_depth, true); // device only

  // dummy rgb image
  CUDAMatrixuc3 rgb_img(rows, cols);
  rgb_img.fill(make_uchar3(255, 0, 0), true); // device only

  // some redundancy just to test everything
  Eigen::Matrix3f cam_K;
  cam_K << 400.f, 0.f, cols / 2.f, 0.f, 400.f, rows / 2.f, 0.f, 0.f, 1.f;
  CUDAMat3 d_cam_K(cam_K);

  Camera camera(d_cam_K, rows, cols, min_depth, max_depth);
  camera.setDepthImage(depth_img);

  Eigen::Matrix4f cam_in_world = Eigen::Matrix4f::Identity();
  camera.setCamInWorld(cam_in_world);

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

  GeometricStreamer streamer(&voxelhasher, false, "memory_allocation.txt", "streamer");
  const Eigen::Vector3f voxel_extents = Eigen::Vector3f::Ones();
  uint initial_chunk_list_size        = 0;
  streamer.create(voxel_extents, max_num_sdf_block_integrate_from_global_hash, initial_chunk_list_size);
  // simulate a circular path
  auto path = makeCameraCircularTrajectory(translation_step, steps);

  for (int i = 0; i < path.size(); ++i) {
    // set camera motion
    const auto& T = path[i];
    camera.setCamInWorld(T.matrix());
    // inverse projection to get point cloud once
    CUDAMatrixf3 point_cloud_img;
    camera.computeCloud(point_cloud_img);

    Eigen::Vector3f cam_pose_in_world = T.translation();
    streamer.stream(cam_pose_in_world, max_radius_for_stream);
    voxelhasher.integrate(point_cloud_img, rgb_img, camera, n_frames_invalidate_voxels);
    streamer.debugCheckForDuplicates();
  }

  double duplicates_ratio = 0;
  // just stream in and out
  for (int i = 0; i < path.size(); ++i) {
    // set camera motion
    const auto& T = path[i];
    camera.setCamInWorld(T.matrix());
    // inverse projection to get point cloud once
    CUDAMatrixf3 point_cloud_img;
    camera.computeCloud(point_cloud_img);

    Eigen::Vector3f cam_pose_in_world = T.translation();
    streamer.stream(cam_pose_in_world, max_radius_for_stream);
    duplicates_ratio = streamer.debugCheckForDuplicates();
  }

  streamer.streamAllOut();
  streamer.printStatistics();

  ASSERT_LT(duplicates_ratio, 0.15);
}
