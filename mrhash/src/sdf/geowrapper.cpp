#include "geowrapper.h"
#include "cuda_matrix.cuh"
#include "serializer.h"
#include "surface_normal_estimator/mad_tree.h"
#include <vector>

namespace pygeowrapper {

  GeoWrapper::GeoWrapper(float sdf_truncation,
                         float sdf_truncation_scale,
                         int integration_weight_sample,
                         float virtual_voxel_size,
                         int n_frames_invalidate_voxels,
                         int voxel_extents_scale,
                         bool viewer_active,
                         float marching_cubes_threshold,
                         uchar min_weight_threshold,
                         float min_depth,
                         float max_depth,
                         const std::string& gs_optimization_param_path,
                         float sdf_var_threshold,
                         float vertices_merging_threshold,
                         bool projective_sdf) :
    sdf_truncation_(sdf_truncation),
    sdf_truncation_scale_(sdf_truncation_scale),
    integration_weight_sample_(integration_weight_sample),
    integration_weight_max_(integration_weight_max),
    virtual_voxel_size_(virtual_voxel_size),
    linked_list_size_(linked_list_size),
    n_frames_invalidate_voxels_(n_frames_invalidate_voxels),
    voxel_extents_scale_(voxel_extents_scale),
    min_weight_threshold_(min_weight_threshold),
    sdf_var_threshold_(sdf_var_threshold),
    vertices_merging_threshold_(vertices_merging_threshold),
    camera_in_lidar_(Eigen::Isometry3f::Identity()),
    gs_optimization_param_path_(gs_optimization_param_path) {
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) {
      throw std::runtime_error("GeoWrapper::GeoWrapper | Failed to get CUDA memory info: " +
                               std::string(cudaGetErrorString(err)));
    }

    if (!gs_optimization_param_path.empty()) {
      free *= gs_scaling_ratio;
      gs_container_ = std::make_unique<cupanutils::cugeoutils::GeometricGaussianContainer>(gs_optimization_param_path_);
    }
    size_t to_alloc         = free * SDFBlocks_ratio;
    num_sdf_blocks_         = (to_alloc * SDFBlocks_ratio) / (sizeof(cupanutils::cugeoutils::Voxel) * total_sdf_block_size);
    hash_num_buckets_       = num_sdf_blocks_;
    hash_bucket_size_       = hash_bucket_size;
    max_num_triangles_mesh_ = (to_alloc * mesh_ratio) / sizeof(cupanutils::cugeoutils::Triangle);
    max_num_sdf_block_integrate_from_global_hash_ =
      (to_alloc * SDFBlocks_stream_ratio) / (sizeof(cupanutils::cugeoutils::Voxel) * total_sdf_block_size);

    const Eigen::Vector3f voxel_extents = Eigen::Vector3f::Ones() * voxel_extents_scale;
    uint initial_chunk_list_size        = 0;

    voxelhasher_ = std::make_unique<cupanutils::cugeoutils::GeometricVoxelContainer>(num_sdf_blocks_,
                                                                                     hash_num_buckets_,
                                                                                     0.f,
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

    streamer_ = std::make_unique<cupanutils::cugeoutils::GeometricStreamer>(
      voxelhasher_.get(), true, "memory_allocation.txt", "streamer_profiler");
    streamer_->create(voxel_extents, max_num_sdf_block_integrate_from_global_hash_, initial_chunk_list_size);
    mesh_extractor_ = std::make_unique<cupanutils::cugeoutils::GeometricMarchingCubes>(
      marching_cubes_threshold, viewer_active, max_num_triangles_mesh_, vertices_merging_threshold);

    setCamera(1.f, 1.f, 0.f, 0.f, 1, 1, min_depth, max_depth, 1);
  }

  GeoWrapper::~GeoWrapper() {
  }

  void GeoWrapper::setCurrPose(Eigen::Vector3f pose, Eigen::Vector4f orientation) {
    // invert order for eigen constructor qw, qx, qy, qz
    const Eigen::Quaternionf quat(orientation(3), orientation(0), orientation(1), orientation(2));
    curr_pose_.setIdentity();
    curr_pose_.linear()      = quat.toRotationMatrix();
    curr_pose_.translation() = pose;
  }

  void GeoWrapper::setCameraInLidar(const Eigen::Matrix4f& camera_in_lidar) {
    camera_in_lidar_ = camera_in_lidar;
  }

  void GeoWrapper::setCamera(const float fx,
                             const float fy,
                             const float cx,
                             const float cy,
                             const int rows,
                             const int cols,
                             const float min_depth,
                             const float max_depth,
                             const int camera_model) {
    // create camera matrix for cuda stuff
    Eigen::Matrix3f camera_matrix;
    camera_matrix << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.f;
    cupanutils::cugeoutils::CUDAMat3 d_cam_K(camera_matrix);
    voxelhasher_->setIntegrationDistance(max_depth);

    const cupanutils::cugeoutils::CameraModel model = (cupanutils::cugeoutils::CameraModel) camera_model;
    camera_     = std::make_unique<cupanutils::cugeoutils::Camera>(d_cam_K, rows, cols, min_depth, max_depth, model);
    view_depth_ = cupanutils::cugeoutils::CUDAMatrixf(camera_->rows(), camera_->cols());
  }

  void GeoWrapper::compute() {
    // set absolute pose
    camera_->setCamInWorld(curr_pose_.matrix());

    cupanutils::cugeoutils::CUDAMatrixf3 point_cloud_img;
    if (depth_img_.size()) {
      // inverse projection to get point cloud once
      depth_img_.toDevice();
      rgb_img_.toDevice();
      camera_->setDepthImage(depth_img_);
      camera_->computeCloud(point_cloud_img);
    }

    if (point_cloud_.size()) {
      point_cloud_.toDevice();
      eigenvectors_.toDevice();
      weights_.toDevice();
    }

    if (voxelhasher_->getHeapHighFreeCount() <= stream_threshold * num_sdf_blocks_)
      streamer_->stream(curr_pose_.translation(), camera_->maxDepth());

    if (depth_img_.size() && rgb_img_.size()) {
      voxelhasher_->integrate(point_cloud_img, rgb_img_, *camera_, n_frames_invalidate_voxels_);
      if (gs_container_)
        gs_container_->runGS(*camera_, *voxelhasher_, rgb_img_, depth_img_);
    }

    if (point_cloud_.size())
      voxelhasher_->integrate(point_cloud_, eigenvectors_, weights_, *camera_, n_frames_invalidate_voxels_);
  }

  void GeoWrapper::extractMesh(const std::string& filename) {
    mesh_extractor_->mesh_ready_ = false;
    mesh_extractor_->mesh_cv_.notify_one();
    streamer_->streamAllOut();
    if (mesh_extractor_->max_num_triangles_mesh_ <= 0) {
      std::cerr << "GeoWrapper::extractMesh | no triangles to extract" << std::endl;
      return;
    }
    mesh_extractor_->vertices_ = Eigen::MatrixXd();
    mesh_extractor_->faces_    = Eigen::MatrixXi();
    std::cout << "GeoWrapper::extractMesh | extracting..." << std::endl;
    mesh_extractor_->merge_mesh_      = true;
    const float radius                = radius_scale_chunk * camera_->maxDepth();
    const int radiusi                 = (int) radius;
    auto [min_grid_pos, max_grid_pos] = streamer_->computeBounds();

    if (min_grid_pos.x() == max_grid_pos.x()) {
      max_grid_pos.x() += 1;
    }
    if (min_grid_pos.y() == max_grid_pos.y()) {
      max_grid_pos.y() += 1;
    }
    if (min_grid_pos.z() == max_grid_pos.z()) {
      max_grid_pos.z() += 1;
    }

    for (int x = min_grid_pos.x(); x < max_grid_pos.x(); x += radiusi) {
      for (int y = min_grid_pos.y(); y < max_grid_pos.y(); y += radiusi) {
        for (int z = min_grid_pos.z(); z < max_grid_pos.z(); z += radiusi) {
          const Eigen::Vector3i chunk(x, y, z);
          streamer_->streamInToGPU(streamer_->chunkToWorld(chunk), radius);
          mesh_extractor_->extractMesh(*voxelhasher_);
          if (mesh_extractor_->num_triangles_ > 0) {
            mesh_extractor_->processTriangles();
          }
          streamer_->streamAllOut();
        }
      }
    }

    const auto& V = mesh_extractor_->vertices_;
    const auto& F = mesh_extractor_->faces_;
    const auto& C = mesh_extractor_->colors_;

    std::ofstream ply(filename);
    if (!ply.is_open()) {
      std::cerr << "GeoWrapper::extractMesh | Failed to open file for writing: " << filename << std::endl;
      return;
    }

    // Header
    ply << "ply\n";
    ply << "format ascii 1.0\n";
    ply << "element vertex " << V.rows() << "\n";
    ply << "property float x\n";
    ply << "property float y\n";
    ply << "property float z\n";
    ply << "property uchar red\n";
    ply << "property uchar green\n";
    ply << "property uchar blue\n";
    ply << "element face " << F.rows() << "\n";
    ply << "property list uchar int vertex_indices\n";
    ply << "end_header\n";

    // Vertex data with colors
    for (int i = 0; i < V.rows(); ++i) {
      unsigned char color[3] = {(unsigned char) (C(i, 0)), (unsigned char) (C(i, 1)), (unsigned char) (C(i, 2))};

      ply << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << " " << static_cast<int>(color[0]) << " " << static_cast<int>(color[1])
          << " " << static_cast<int>(color[2]) << "\n";
    }

    // Faces
    for (int i = 0; i < F.rows(); ++i) {
      ply << "3 " << F(i, 0) << " " << F(i, 1) << " " << F(i, 2) << "\n";
    }

    ply.close();
    std::cout << "GeoWrapper::extractMesh | written " << V.rows() << " vertices and " << F.rows() << " faces to " << filename
              << std::endl;
  }

  void GeoWrapper::GSSavePointCloud(const std::string& folder) {
    if (!gs_container_) {
      std::cerr << "GeoWrapper::GSSavePointCloud | GS container not initialized" << std::endl;
      return;
    }
    gs_container_->gs_model_.Save_ply(folder, voxelhasher_->num_integrated_frames_, true);
    std::cout << "GeoWrapper::GSSavePointCloud | written gaussians to " << folder << std::endl;
  }

  void GeoWrapper::GSFinalOpt() {
    if (gs_container_)
      gs_container_->optimizeGSFinal();
  }

  void GeoWrapper::setRGBImage(nb::ndarray<uint8_t> input_rgb_array) {
    // check the dimensions of the input array
    if (input_rgb_array.ndim() != 3) {
      throw std::runtime_error("GeoWrapper::setRGBImage|input should be a 3D numpy array");
    }

    // get the dimensions of the array
    const size_t rows     = input_rgb_array.shape(0);
    const size_t cols     = input_rgb_array.shape(1);
    const size_t channels = input_rgb_array.shape(2);

    if (channels != 3) {
      throw std::runtime_error("GeoWrapper::setRGBImage|input should have 3 channels");
    }

    uint8_t* ptr = input_rgb_array.data();

    // resize the image to match the input array dimensions
    rgb_img_.resize(rows, cols);

    // populate the image
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        rgb_img_.at(r, c).x = ptr[r * cols * channels + c * channels];
        rgb_img_.at(r, c).y = ptr[r * cols * channels + c * channels + 1];
        rgb_img_.at(r, c).z = ptr[r * cols * channels + c * channels + 2];
      }
    }
  }

  void GeoWrapper::setRGBImage(const cv::Mat& input_rgb_image) {
    // check that the input is 3-channel 8-bit
    if (input_rgb_image.type() != CV_8UC3) {
      throw std::runtime_error("GeoWrapper::setRGBImage|input Mat should be CV_8UC3");
    }

    const int rows = input_rgb_image.rows;
    const int cols = input_rgb_image.cols;

    // resize the image to match the input dimensions
    rgb_img_.resize(rows, cols);

    // populate the image (OpenCV stores in BGR order by default)
    for (int r = 0; r < rows; ++r) {
      const uint8_t* row_ptr = input_rgb_image.ptr<uint8_t>(r);
      for (int c = 0; c < cols; ++c) {
        // Note: OpenCV typically uses BGR order, adjust if needed
        rgb_img_.at(r, c).x = row_ptr[c * 3];     // B
        rgb_img_.at(r, c).y = row_ptr[c * 3 + 1]; // G
        rgb_img_.at(r, c).z = row_ptr[c * 3 + 2]; // R
      }
    }
  }

  void GeoWrapper::setDepthImage(nb::ndarray<float> input_depth_array) {
    // check the dimensions of the input array
    if (input_depth_array.ndim() != 2) {
      throw std::runtime_error("GeoWrapper::setDepthImage|input should be a 2D numpy array");
    }

    // get the dimensions of the array
    const size_t rows = input_depth_array.shape(0);
    const size_t cols = input_depth_array.shape(1);

    // get a pointer to the data as a float*
    float* ptr = input_depth_array.data();

    // resize the matrix to match the input array dimensions
    depth_img_.resize(rows, cols);
    depth_img_.fill(0.f);
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        depth_img_.at(r, c) = ptr[r * cols + c];
      }
    }
  }

  void GeoWrapper::setDepthImage(const cv::Mat& input_depth_image) {
    // check that the input is single-channel float
    if (input_depth_image.type() != CV_32FC1) {
      throw std::runtime_error("GeoWrapper::setDepthImage|input Mat should be CV_32FC1");
    }

    const int rows = input_depth_image.rows;
    const int cols = input_depth_image.cols;

    // resize the matrix to match the input dimensions
    depth_img_.resize(rows, cols);
    depth_img_.fill(0.f);

    // populate the depth image using OpenCV's efficient row access
    for (int r = 0; r < rows; ++r) {
      const float* row_ptr = input_depth_image.ptr<float>(r);
      for (int c = 0; c < cols; ++c) {
        depth_img_.at(r, c) = row_ptr[c];
      }
    }
  }

  void GeoWrapper::setPointCloud(nb::ndarray<float> input_point_cloud_array, const bool compute_normals) {
    if (input_point_cloud_array.ndim() != 2) {
      throw std::runtime_error("GeoWrapper::setPointCloud|input should be a 2D numpy array");
    }

    const float* ptr        = input_point_cloud_array.data();
    const size_t num_points = input_point_cloud_array.shape(0);

    auto points = std::make_unique<std::vector<Eigen::Vector3d>>();

    // Resize point_cloud_ before writing to it
    point_cloud_.resize(num_points, 1);

    // convert py::array_to std::vector<Eigen::Vector3f>
    for (size_t n = 0; n < num_points; ++n) {
      Eigen::Vector3d point;
      point.x() = ptr[n * 3 + 0];
      point.y() = ptr[n * 3 + 1];
      point.z() = ptr[n * 3 + 2];
      points->push_back(point);

      point_cloud_.at(n).x = point.x();
      point_cloud_.at(n).y = point.y();
      point_cloud_.at(n).z = point.z();
    }

    eigenvectors_.resize(3 * num_points, 1);
    weights_.resize(num_points, 1);

    if (compute_normals) {
      // build tree
      MADtree tree(points.get(), points->begin(), points->end(), 0.4, 0.4, 0, 3, nullptr, nullptr);

      // extract points and normals from mad-tree
      LeafList leafs;

      tree.getLeafs(std::back_inserter(leafs));

      size_t n = 0;
      for (auto& leaf : leafs) {
        if (leaf->mean_.dot(leaf->eigenvectors_.col(0)) > 0)
          leaf->eigenvectors_.col(0) *= -1.0;

        // extract point from points vec considering leaf->begin and leaf->end
        for (auto it = leaf->begin_; it != leaf->end_; ++it) {
          eigenvectors_.at(3 * n).x     = leaf->eigenvectors_.col(0).x();
          eigenvectors_.at(3 * n).y     = leaf->eigenvectors_.col(0).y();
          eigenvectors_.at(3 * n).z     = leaf->eigenvectors_.col(0).z();
          eigenvectors_.at(3 * n + 1).x = leaf->eigenvectors_.col(1).x();
          eigenvectors_.at(3 * n + 1).y = leaf->eigenvectors_.col(1).y();
          eigenvectors_.at(3 * n + 1).z = leaf->eigenvectors_.col(1).z();
          eigenvectors_.at(3 * n + 2).x = leaf->eigenvectors_.col(2).x();
          eigenvectors_.at(3 * n + 2).y = leaf->eigenvectors_.col(2).y();
          eigenvectors_.at(3 * n + 2).z = leaf->eigenvectors_.col(2).z();
          weights_.at(n)                = leaf->weight_;

          n++;
        }
      }
    }
  }

  void GeoWrapper::setPointCloud(const std::vector<Eigen::Vector3f>& input_point_cloud, bool compute_normals) {
    if (input_point_cloud.empty()) {
      std::cerr << "GeoWrapper::setPointCloud | empty input point cloud" << std::endl;
      return;
    }

    const size_t num_points = input_point_cloud.size();

    auto points = std::make_unique<std::vector<Eigen::Vector3d>>();

    // Resize point_cloud_ before writing to it
    point_cloud_.resize(num_points, 1);

    for (size_t n = 0; n < num_points; ++n) {
      Eigen::Vector3d point;
      point.x() = input_point_cloud.at(n)(0);
      point.y() = input_point_cloud.at(n)(1);
      point.z() = input_point_cloud.at(n)(2);
      points->push_back(point);

      point_cloud_.at(n).x = point.x();
      point_cloud_.at(n).y = point.y();
      point_cloud_.at(n).z = point.z();
    }

    eigenvectors_.resize(3 * num_points, 1);
    weights_.resize(num_points, 1);

    if (compute_normals) {
      // build tree
      MADtree tree(points.get(), points->begin(), points->end(), 0.4, 0.4, 0, 3, nullptr, nullptr);

      // extract points and normals from mad-tree
      LeafList leafs;

      tree.getLeafs(std::back_inserter(leafs));

      size_t n = 0;
      for (auto& leaf : leafs) {
        if (leaf->mean_.dot(leaf->eigenvectors_.col(0)) > 0)
          leaf->eigenvectors_.col(0) *= -1.0;

        // extract point from points vec considering leaf->begin and leaf->end
        for (auto it = leaf->begin_; it != leaf->end_; ++it) {
          eigenvectors_.at(3 * n).x     = leaf->eigenvectors_.col(0).x();
          eigenvectors_.at(3 * n).y     = leaf->eigenvectors_.col(0).y();
          eigenvectors_.at(3 * n).z     = leaf->eigenvectors_.col(0).z();
          eigenvectors_.at(3 * n + 1).x = leaf->eigenvectors_.col(1).x();
          eigenvectors_.at(3 * n + 1).y = leaf->eigenvectors_.col(1).y();
          eigenvectors_.at(3 * n + 1).z = leaf->eigenvectors_.col(1).z();
          eigenvectors_.at(3 * n + 2).x = leaf->eigenvectors_.col(2).x();
          eigenvectors_.at(3 * n + 2).y = leaf->eigenvectors_.col(2).y();
          eigenvectors_.at(3 * n + 2).z = leaf->eigenvectors_.col(2).z();
          weights_.at(n)                = leaf->weight_;

          n++;
        }
      }
    }
  }

  void GeoWrapper::setPointCloud(nb::ndarray<float> input_point_cloud, nb::ndarray<float> normals) {
    if (input_point_cloud.ndim() != 2) {
      throw std::runtime_error("GeoWrapper::setPointCloud|point cloud input should be a 2D numpy array");
    }

    if (normals.ndim() != 2) {
      throw std::runtime_error("GeoWrapper::setPointCloud|normals input should be a 2D numpy array");
    }

    if (input_point_cloud.shape(0) != normals.shape(0)) {
      throw std::runtime_error(
        "GeoWrapper::setPointCloud|point_cloud input and normals input should have the same number of points");
    }

    const float* point_cloud_ptr = input_point_cloud.data();
    const float* normals_ptr     = normals.data();
    const size_t num_points      = input_point_cloud.shape(0);

    point_cloud_.resize(num_points, 1);
    eigenvectors_.resize(num_points, 1);

    // convert py::array_to std::vector<Eigen::Vector3f>
    for (size_t n = 0; n < num_points; ++n) {
      point_cloud_.at(n).x  = point_cloud_ptr[n * 3 + 0];
      point_cloud_.at(n).y  = point_cloud_ptr[n * 3 + 1];
      point_cloud_.at(n).z  = point_cloud_ptr[n * 3 + 2];
      eigenvectors_.at(n).x = normals_ptr[n * 3 + 0];
      eigenvectors_.at(n).y = normals_ptr[n * 3 + 1];
      eigenvectors_.at(n).z = normals_ptr[n * 3 + 2];
    }
  }

  void GeoWrapper::setPointCloud(const std::vector<Eigen::Vector3f>& input_point_cloud,
                                 const std::vector<Eigen::Vector3f>& input_normals) {
    if (input_point_cloud.empty()) {
      std::cerr << "GeoWrapper::setPointCloud | empty input point cloud" << std::endl;
      return;
    }
    if (input_normals.empty()) {
      std::cerr << "GeoWrapper::setPointCloud | empty input normals" << std::endl;
      return;
    }

    if (input_point_cloud.size() != input_normals.size()) {
      std::cerr << "GeoWrapper::setPointCloud | input point cloud size does not match input normal size" << std::endl;
      return;
    }

    const size_t num_points = input_point_cloud.size();

    point_cloud_.resize(num_points, 1);
    eigenvectors_.resize(num_points, 1);

    // convert py::array_to std::vector<Eigen::Vector3f>
    for (size_t n = 0; n < num_points; ++n) {
      point_cloud_.at(n).x  = input_point_cloud.at(n)(0);
      point_cloud_.at(n).y  = input_point_cloud.at(n)(1);
      point_cloud_.at(n).z  = input_point_cloud.at(n)(2);
      eigenvectors_.at(n).x = input_normals.at(n)(0);
      eigenvectors_.at(n).y = input_normals.at(n)(1);
      eigenvectors_.at(n).z = input_normals.at(n)(2);
    }
  }

  Eigen::MatrixX3f GeoWrapper::getPointCloud() {
    Eigen::MatrixX3f point_cloud(point_cloud_.rows(), 3);
    for (size_t r = 0; r < point_cloud_.rows(); ++r) {
      point_cloud(r, 0) = point_cloud_.at(r).x;
      point_cloud(r, 1) = point_cloud_.at(r).y;
      point_cloud(r, 2) = point_cloud_.at(r).z;
    }
    return point_cloud;
  }

  Eigen::MatrixX3f GeoWrapper::getNormals() {
    Eigen::MatrixX3f normals(eigenvectors_.rows() / 3, 3);
    for (size_t r = 0; r < eigenvectors_.rows() / 3; ++r) {
      normals(r, 0) = eigenvectors_.at<0>(r).x;
      normals(r, 1) = eigenvectors_.at<0>(r).y;
      normals(r, 2) = eigenvectors_.at<0>(r).z;
    }
    return normals;
  }

  void GeoWrapper::clearBuffers() {
    streamer_->streamAllOut();
    std::cout << "clearing buffers..." << std::endl;
    streamer_->clearGrid();
    streamer_->printStatistics();
  }

  void GeoWrapper::streamAllOut() {
    streamer_->streamAllOut();
  }

  void GeoWrapper::serializeData(const std::string& filename_hash, const std::string& filename_voxel) {
    streamer_->serializeData(filename_hash, filename_voxel);
  }

  void GeoWrapper::serializeGrid(const std::string& filename) {
    cupanutils::cugeoutils::Serializer<cupanutils::cugeoutils::Voxel>::serialize(streamer_->grid_, filename);
  }

  void GeoWrapper::deserializeGrid(const std::string& filename) {
    cupanutils::cugeoutils::Serializer<cupanutils::cugeoutils::Voxel>::deserialize(streamer_->grid_, filename);
  }

  // template class GeoWrapper<cupanutils::cugeoutils::Voxel>;

} // namespace pygeowrapper
