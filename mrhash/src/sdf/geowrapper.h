#pragma once
#include "camera.cuh"
#include "gaussian_data_structures.cuh"
#include "marching_cubes.cuh"
#include "streamer.cuh"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <nanobind/ndarray.h>

#include <opencv2/core.hpp>

namespace nb = nanobind;

namespace pygeowrapper {

  class GeoWrapper {
  private:
    int hash_num_buckets_;
    int num_sdf_blocks_;
    int hash_bucket_size_;
    float sdf_truncation_;
    float sdf_truncation_scale_;
    int integration_weight_sample_;
    int integration_weight_max_;
    float virtual_voxel_size_;
    int linked_list_size_;

    int n_frames_invalidate_voxels_;
    uint max_num_sdf_block_integrate_from_global_hash_;
    int voxel_extents_scale_;

    uchar min_weight_threshold_;
    float sdf_var_threshold_          = 0.f;
    uint max_num_triangles_mesh_      = 0;
    float vertices_merging_threshold_ = 0.f;

    Eigen::Isometry3f curr_pose_;

    std::unique_ptr<cupanutils::cugeoutils::Camera> camera_;
    Eigen::Isometry3f camera_in_lidar_;

    cupanutils::cugeoutils::CUDAMatrixf depth_img_;
    cupanutils::cugeoutils::CUDAMatrixuc3 rgb_img_;
    cupanutils::cugeoutils::CUDAVectorf3 point_cloud_;
    cupanutils::cugeoutils::CUDAVectorf weights_;
    cupanutils::cugeoutils::CUDAVectorf3 eigenvectors_;

    std::unique_ptr<cupanutils::cugeoutils::GeometricVoxelContainer> voxelhasher_;
    std::unique_ptr<cupanutils::cugeoutils::GeometricStreamer> streamer_;
    std::unique_ptr<cupanutils::cugeoutils::GeometricMarchingCubes> mesh_extractor_;

    std::unique_ptr<cupanutils::cugeoutils::GeometricGaussianContainer> gs_container_;
    std::string gs_optimization_param_path_;

    cupanutils::cugeoutils::CUDAMatrixf view_depth_;

  public:
    ~GeoWrapper();

    GeoWrapper(float sdf_truncation,
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
               const std::string& gs_optimization_param_path = default_gs_optimization_param_path,
               float sdf_var_threshold                       = default_sdf_var_threshold,
               float vertices_merging_threshold              = default_vertices_merging_threshold,
               bool projective_sdf                           = default_projective_sdf);

    // clang-format off
    int getHashNumBuckets() const { return hash_num_buckets_; }
    int getNumSdfBlocks() const { return num_sdf_blocks_; }
    int getHashBucketSize() const { return hash_bucket_size_; }
    float getSdfTruncation() const { return sdf_truncation_; }
    float getSdfTruncationScale() const { return sdf_truncation_scale_; }
    int getIntegrationWeightSample() const { return integration_weight_sample_; }
    int getIntegrationWeightMax() const { return integration_weight_max_; }
    float getVirtualVoxelSize() const { return virtual_voxel_size_; }
    int getLinkedListSize() const { return linked_list_size_; }
    int getNFramesInvalidateVoxels() const { return n_frames_invalidate_voxels_; }
    int getMaxNumSdfBlockIntegrateFromGlobalHash() const { return max_num_sdf_block_integrate_from_global_hash_; }
    int getVoxelExtentsScale() const { return voxel_extents_scale_; }
    const Eigen::MatrixXd& getVertices() const { return mesh_extractor_->getVertices();}
    const Eigen::MatrixXi& getFaces() const { return mesh_extractor_->getFaces();}
    const Eigen::MatrixXd& getColors() const { return mesh_extractor_->getColors();}
    const Eigen::Matrix4f& getCurrPose() const { return curr_pose_.matrix(); }
    Eigen::MatrixX3f getPointCloud();
    Eigen::MatrixX3f getNormals();
    
    void setHashNumBuckets(int hash_num_buckets) { hash_num_buckets_ = hash_num_buckets; }
    void setNumSdfBlocks(int num_sdf_blocks) { num_sdf_blocks_ = num_sdf_blocks; }
    void setHashBucketSize(int hash_bucket_size) { hash_bucket_size_ = hash_bucket_size; }
    void setSdfTruncation(float sdf_truncation) { sdf_truncation_ = sdf_truncation; }
    void setSdfTruncationScale(float sdf_truncation_scale) { sdf_truncation_scale_ = sdf_truncation_scale; }
    void setIntegrationWeightSample(int integration_weight_sample) { integration_weight_sample_ = integration_weight_sample; }
    void setIntegrationWeightMax(int integration_weight_max) { integration_weight_max_ = integration_weight_max; }
    void setVirtualVoxelSize(float virtual_voxel_size) { virtual_voxel_size_ = virtual_voxel_size; }
    void setLinkedListSize(int linked_list_size) { linked_list_size_ = linked_list_size; }
    void setNFramesInvalidateVoxels(int n_frames_invalidate_voxels) { n_frames_invalidate_voxels_ = n_frames_invalidate_voxels; }
    void setMaxNumSdfBlockIntegrateFromGlobalHash(int max_num_sdf_block_integrate_from_global_hash) { max_num_sdf_block_integrate_from_global_hash_ = max_num_sdf_block_integrate_from_global_hash; }
    void setVoxelExtentsScale(int voxel_extents_scale) { voxel_extents_scale_ = voxel_extents_scale; }
    // clang-format on

    // translation <tx, ty, tz> and quaternion <qx, qy, qz, qw>
    /**
     * @brief Set the current pose of the referenceframe.
     *
     * @param pose The position <tx, ty, tz> of the reference frame.
     * @param orientation The orientation <qx, qy, qz, qw> of the reference frame
     */
    void setCurrPose(Eigen::Vector3f pose, Eigen::Vector4f orientation);

    /**
     * @brief Set the transformation from camera reference frame to lidar reference frame.
     *
     * @param camera_in_lidar The SE(3) transformation lidar_T_camera
     */
    void setCameraInLidar(const Eigen::Matrix4f& camera_in_lidar);

    /**
     * @brief Set the camera parameters
     *
     * @param fx X-Axis focal length
     * @param fy Y-Axis focal length
     * @param cx X-Axis camera center
     * @param cy Y-Axis camera center
     * @param rows Image rows
     * @param cols Image columns
     * @param min_depth Measurements minimum depth
     * @param max_depth Measurements maximum depth
     * @param camera_model Type of camera model, currently implemented: { Pinhole = 0, Spherical = 1 }
     */
    void setCamera(const float fx,
                   const float fy,
                   const float cx,
                   const float cy,
                   const int rows,
                   const int cols,
                   const float min_depth,
                   const float max_depth,
                   const int camera_model);
    /**
     * @brief Set the current input RGB image (pybinded function)
     *
     * @param input_rgb_array RGB image as a numpy array of dtype uint8_t
     */
    void setRGBImage(nb::ndarray<uint8_t> input_rgb_array);
    /**
     * @brief Set the current input RGB image from OpenCV Mat
     *
     * @param input_rgb_image RGB image as cv::Mat (CV_8UC3)
     */
    void setRGBImage(const cv::Mat& input_rgb_image);
    /**
     * @brief Set the current input depth image (pybinded function)
     *
     * @param input_depth_array Depth image as a numpy array of dtype float32
     */
    void setDepthImage(nb::ndarray<float> input_depth_array);
    /**
     * @brief Set the current input depth image from OpenCV Mat
     *
     * @param input_depth_image Depth image as cv::Mat (CV_32FC1)
     */
    void setDepthImage(const cv::Mat& input_depth_image);
    /**
     * @brief Set the current input point cloud and optionally compute normals (pybinded function)
     *
     * @param input_point_cloud Point Cloud as a numpy array of dtype float32
     * @param compute_normals Boolean flag to activate normals computation for the current cloud*/
    void setPointCloud(nb::ndarray<float> input_point_cloud, bool compute_normals = false);
    /**
     * @brief Set the current input point cloud and optionally compute normals
     *
     * @param input_point_cloud Point Cloud as an std::vector of Eigen::Vector3f
     * @param compute_normals Boolean flag to activate normals computation for the current cloud*/
    void setPointCloud(const std::vector<Eigen::Vector3f>& input_point_cloud, bool compute_normals = false);
    /**
     * @brief Set the current input point cloud and input normals (pybinded function)
     *
     * @param input_point_cloud Point cloud as a numpy array of dtype float32
     * @param input_normals Normals as a numpy array of dtype float32
     * */
    void setPointCloud(nb::ndarray<float> input_point_cloud, nb::ndarray<float> normals);
    /**
     * @brief Set the current input point cloud and input normals
     *
     * @param input_point_cloud Point cloud as an std::vector of Eigen::Vector3f
     * @param input_normals Normals as an std::vector of Eigen::Vector3f
     * */
    void setPointCloud(const std::vector<Eigen::Vector3f>& input_point_cloud, const std::vector<Eigen::Vector3f>& normals);

    /**
     * @brief Process current input RGB/depth images, point cloud, normals to update the SDF grid
     *
     * This function integrates the current sensor data (depth, RGB, point cloud) into the TSDF voxel grid.
     * It performs streaming to manage GPU memory, integrates the data into voxel blocks, and optionally
     * runs Gaussian Splatting optimization if a GS container is initialized.
     */
    void compute();

    /**
     * @brief Extract triangle mesh from SDF grid
     *
     * @param filename Path to output PLY file for the extracted mesh
     *
     * Streams all voxel data to CPU, delegates to mesh_extractor_ for iso-surface extraction
     * (typically Marching Cubes), and writes the combined mesh with vertex colors to a PLY file.
     * Vertex merging is applied based on vertices_merging_threshold configured in mesh_extractor_.
     */
    void extractMesh(const std::string& filename);
    /**
     * @brief Perform the final refinement of the Gaussians
     *
     * Executes final optimization pass on the Gaussian splatting model
     */
    void GSFinalOpt();

    /**
     * @brief Save the Gaussians point cloud
     *
     * @param folder Path to output folder for the Gaussian splatting representation
     *
     * Serializes the Gaussian splatting representation to disk
     */
    void GSSavePointCloud(const std::string& folder);

    /**
     * @brief Move all the voxel blocks from GPU memory to CPU memory
     * */
    void streamAllOut();

    /**
     * @brief Reset all the buffers both on GPU and in CPUu
     * */
    void clearBuffers();
    /**
     * @brief Save the SDF grid as .ply file
     *
     * @note Only voxel up whose SDF is less than 5*voxel_size will be saved
     * */
    void serializeData(const std::string& filename_hash  = "./data/hash_points.ply",
                       const std::string& filename_voxel = "./data/voxel_points.ply");
    /**
     * @brief Save all the data of the grid as a binary file for later retrieval
     * */
    void serializeGrid(const std::string& filename);
    /**
     * @brief Load all the data of the grid from a binary file
     * */
    void deserializeGrid(const std::string& filename);
  };

} // namespace pygeowrapper
