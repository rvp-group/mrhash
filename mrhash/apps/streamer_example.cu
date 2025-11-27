#include <sdf/cuda_matrix.cuh>
#include <sdf/marching_cubes.cuh>
#include <sdf/serializer.h>
#include <sdf/streamer.cuh>
#include <sdf/voxel_data_structures.cuh>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

using namespace cupanutils::cugeoutils;

std::vector<Eigen::Isometry3f> makeCameraCircularTrajectory(const float t_step, const int steps) {
  std::vector<Eigen::Isometry3f> path;
  const float angle_step   = 2.f * M_PI / (float) steps;
  Eigen::Isometry3f T_abs  = Eigen::Isometry3f::Identity();
  Eigen::Isometry3f T_step = Eigen::Isometry3f::Identity();
  T_step.linear() << cos(angle_step), 0.f, sin(angle_step), 0.f, 1.f, 0.f, -sin(angle_step), 0.f, cos(angle_step);
  T_step.translation() << t_step, 0, t_step;
  for (int i = 0; i < steps; ++i) {
    path.push_back(T_abs);
    T_abs = T_abs * T_step;
  }
  return path;
}

std::vector<Eigen::Isometry3f> makeCameraStraightTrajectory(const float t_step, const int steps) {
  std::vector<Eigen::Isometry3f> path;
  Eigen::Isometry3f T_abs  = Eigen::Isometry3f::Identity();
  Eigen::Isometry3f T_step = Eigen::Isometry3f::Identity();
  T_step.translation() << 0, 0, t_step;
  for (int i = 0; i < steps; ++i) {
    path.push_back(T_abs);
    T_abs = T_abs * T_step;
  }
  return path;
}

int main(int argc, char* argv[]) {
  srand(time(NULL));
  YAML::Node config;
  if (argc > 1) {
    config = YAML::LoadFile(argv[1]);
  } else {
    std::cerr << "missing arguments <config-file>" << std::endl;
    return -1;
  }

  const float default_depth                              = config["default_depth"].as<float>();
  const int hash_num_buckets                             = config["hash_num_buckets"].as<int>();
  const int num_sdf_blocks                               = config["num_sdf_blocks"].as<int>();
  const int hash_bucket_size                             = config["hash_bucket_size"].as<int>();
  const float max_integration_distance                   = config["max_integration_distance"].as<float>();
  const float sdf_truncation                             = config["sdf_truncation"].as<float>();
  const float sdf_truncation_scale                       = config["sdf_truncation_scale"].as<float>();
  const int integration_weight_sample                    = config["integration_weight_sample"].as<int>();
  const int integration_weight_max                       = config["integration_weight_max"].as<int>();
  const float virtual_voxel_size                         = config["virtual_voxel_size"].as<float>();
  const int linked_list_size                             = config["linked_list_size"].as<int>();
  const float min_depth                                  = config["min_depth"].as<float>();
  const float max_depth                                  = config["max_depth"].as<float>();
  const float max_radius_for_stream                      = config["max_radius_for_stream"].as<float>();
  const int n_frames_invalidate_voxels                   = config["n_frames_invalidate_voxels"].as<int>();
  const int max_num_sdf_block_integrate_from_global_hash = config["max_num_sdf_block_integrate_from_global_hash"].as<int>();
  const float voxel_extents_scale                        = config["voxel_extents_scale"].as<float>();

  const uint max_num_triangles_mesh      = config["max_num_triangles_mesh"].as<uint>();
  const float marching_cubes_threshold   = config["marching_cubes_threshold"].as<float>();
  const uchar min_weight_threshold       = (uchar) config["min_weight_threshold"].as<uint>();
  const float sdf_var_threshold          = config["sdf_var_threshold"].as<float>();
  const float vertices_merging_threshold = config["vertices_merging_threshold"].as<float>();

  // trajectory params
  const float translation_step = config["translation_step"].as<float>();
  const int steps              = config["steps"].as<int>(); // to discritize angle path

  const int rows = config["rows"].as<int>();
  const int cols = config["cols"].as<int>();

  // dummy depth
  CUDAMatrixf depth_img(rows, cols);
  depth_img.fill(default_depth, false);

  // some redundancy just to test everything
  Eigen::Matrix3f cam_K;
  cam_K << 400.f, 0.f, cols / 2.f, 0.f, 400.f, rows / 2.f, 0.f, 0.f, 1.f;
  CUDAMat3 d_cam_K(cam_K);

  Camera camera(d_cam_K, rows, cols, min_depth, max_depth);

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
                                      true,
                                      true,
                                      "memory_allocation.txt",
                                      "integration_profiler",
                                      "rendering_profiler");

  GeometricStreamer streamer(&voxelhasher, true, "memory_allocation.txt", "streamer_profiler");

  GeometricMarchingCubes mesh_extractor(marching_cubes_threshold, false, max_num_triangles_mesh, vertices_merging_threshold);

  const Eigen::Vector3f voxel_extents = Eigen::Vector3f::Ones() * voxel_extents_scale;
  uint initial_chunk_list_size        = 0;
  streamer.create(voxel_extents, max_num_sdf_block_integrate_from_global_hash, initial_chunk_list_size);
  // simulate a circular path
  // auto path = makeCameraCircularTrajectory(translation_step, steps);
  auto path = makeCameraStraightTrajectory(translation_step, steps);

  CUDAMatrixi alloc_map(rows, cols);
  std::cerr << "generated alloc_map: " << alloc_map.rows() << "x" << alloc_map.cols() << std::endl;

  for (int i = 0; i < path.size(); ++i) {
    // set camera motion
    const auto& T = path[i];
    camera.setCamInWorld(T.matrix());
    // fill with random values default_depth + random within 0.05 and 0.1
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        if (c > cols / 2) {
          depth_img(r, c) = default_depth + ((float) rand() / (float) RAND_MAX) * 0.005f;
        }
      }
    }
    // set all the borders of depth_img to 0, with num_pixels
    const int num_pixels = rows / 5;
    for (int r = 0; r < num_pixels; r++) {
      for (int c = 0; c < num_pixels; c++) {
        depth_img(r, c)                       = 0;
        depth_img(r, cols - c - 1)            = 0;
        depth_img(rows - r - 1, c)            = 0;
        depth_img(rows - r - 1, cols - c - 1) = 0;
      }
    }
    depth_img.toDevice();

    camera.setDepthImage(depth_img);
    // inverse projection to get point cloud once
    CUDAMatrixf3 point_cloud_img;
    camera.computeCloud(point_cloud_img);

    // dummy rgb image
    CUDAMatrixuc3 rgb_img(rows, cols);

    rgb_img.fill(make_uchar3(0, 0, 64), false);

    Eigen::Vector3f cam_pose_in_world = T.translation();
    streamer.stream(cam_pose_in_world, max_radius_for_stream);

    voxelhasher.integrate(point_cloud_img, rgb_img, camera, n_frames_invalidate_voxels);
  }

  streamer.printStatistics();
  streamer.streamAllOut();
  streamer.printStatistics();
  streamer.serializeData();
  Serializer<Voxel>::serialize(streamer.getGrid());
  std::unordered_map<Eigen::Vector3i, std::unique_ptr<ChunkDesc<Voxel>>, Vector3iHash> grid;
  Serializer<Voxel>::deserialize(grid);

  streamer.clearGrid();

  return 0;
}
