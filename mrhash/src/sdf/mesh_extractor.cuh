#pragma once
#include <Eigen/Dense>
#include <vector>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "voxel_data_structures.cuh"
#include "voxel_hash_utils.cuh"

namespace cupanutils {
  namespace cugeoutils {

    using TriangleFace = std::array<int, 3>;
    struct TriangleFaceCompare {
      bool operator()(const TriangleFace& a, const TriangleFace& b) const {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
      }
    };

    struct Vector3dHash {
      size_t operator()(const Eigen::Vector3d& v) const {
        const char* data_ptr = reinterpret_cast<const char*>(v.data());
        return std::hash<std::string_view>()(std::string_view(data_ptr, sizeof(double) * 3));
      }
    };

    struct Vector3dEqual {
      bool operator()(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const {
        return a == b;
      }
    };

    struct Vec3iHash {
      size_t operator()(const Eigen::Vector3i& key) const {
        return ((key.x() * p0) ^ (key.y() * p1) ^ (key.z() * p2));
      }
    };

    template <typename T>
    class MeshExtractor {
    public:
      __host__ explicit MeshExtractor(const bool viewer_active,
                                      const uint max_num_triangles_mesh,
                                      const float vertices_merging_threshold) :
        viewer_active_(viewer_active),
        max_num_triangles_mesh_(max_num_triangles_mesh),
        vertices_merging_threshold_(vertices_merging_threshold) {
        CUDA_CHECK(cudaMalloc((void**) &d_triangles_, sizeof(Triangle) * max_num_triangles_mesh_));
        CUDA_CHECK(cudaMalloc((void**) &d_trianglesCounter_, sizeof(uint)));

        CUDA_CHECK(cudaMallocHost(&h_triangles_, sizeof(Triangle) * max_num_triangles_mesh_));

        uint triangles_counter_init_value = 0;
        CUDA_CHECK(cudaMemcpy(&d_trianglesCounter_[0], &triangles_counter_init_value, sizeof(uint), cudaMemcpyHostToDevice));

        num_triangles_ = 0;
        if (viewer_active_ && max_num_triangles_mesh_ > 0) {
          triangle_processing_thread_ = std::make_unique<std::thread>(&MeshExtractor::processTrianglesThread, this);
        }
      }
      ~MeshExtractor() {
        if (triangle_processing_thread_ && triangle_processing_thread_->joinable())
          triangle_processing_thread_->join();
      }
      /**
       * @brief Combine multiple sets of vertices and faces into single consolidated matrices.
       *
       * This function concatenates lists of vertex and face matrices into unified vertex and face matrices.
       * Face indices are adjusted to account for the offset caused by concatenating vertices.
       *
       * @param vertices_src Vector of vertex matrices, each with rows as vertices and columns as vertex attributes.
       * @param faces_src Vector of face matrices, each with rows as faces and columns as vertex indices.
       * @param[out] vertices_dst Matrix to store the combined vertices.
       * @param[out] faces_dst Matrix to store the combined faces with updated indices.
       *
       * @throws std::runtime_error if the sizes of vertices_src and faces_src differ,
       *         or if vertex or face dimensions mismatch across inputs.
       */
      void combine(const std::vector<Eigen::MatrixXd>& vertices_src,
                   const std::vector<Eigen::MatrixXi>& faces_src,
                   Eigen::MatrixXd& vertices_dst,
                   Eigen::MatrixXi& faces_dst);

      /**
       * @brief Remove duplicate triangular faces from a mesh.
       *
       * This function identifies and removes duplicate faces from the input face matrix.
       * Faces are assumed to be triangles (3 vertex indices per face).
       * The output matrix contains only unique faces, with vertex indices sorted within each face.
       *
       * @param faces_src Input matrix of triangular faces (rows = faces, cols = 3 vertex indices).
       * @param[out] faces_dst Output matrix containing unique faces only.
       */
      void removeDuplicateFacesTriangle(const Eigen::MatrixXi& faces_src, Eigen::MatrixXi& faces_dst);

      /**
       * @brief Remove duplicate vertices from a set of vertices within a given tolerance,
       *        and update the faces accordingly.
       *
       * This function identifies unique vertices from the input vertex matrix by comparing vertices
       * within a specified epsilon distance. Vertices closer than epsilon are considered duplicates.
       * It outputs a matrix of unique vertices, a mapping from old vertex indices to new indices,
       * and updates the faces matrix to reflect the new vertex indices.
       *
       * @param vertices_src Input matrix of vertices (rows = vertices, cols = 3 coordinates).
       * @param faces_src Input matrix of faces (rows = faces, cols = vertex indices).
       * @param epsilon Distance tolerance for considering vertices as duplicates.
       * @param[out] vertices_dst Output matrix containing unique vertices.
       * @param[out] faces_dst Output matrix containing updated faces with new vertex indices.
       * @param[out] old_to_new_mapping Output vector mapping each original vertex index to the new unique vertex index.
       *
       * @note The function assumes 3D vertices and uses squared Euclidean distance for comparison.
       */
      void removeDuplicateVerticesTriangle(const Eigen::MatrixXd& vertices_src,
                                           const Eigen::MatrixXi& faces_src,
                                           const double epsilon,
                                           Eigen::MatrixXd& vertices_dst,
                                           Eigen::MatrixXi& faces_dst,
                                           Eigen::VectorXi& old_to_new_mapping);

      void extractMesh(VoxelContainer<T>& container);

      void processTriangles();
      void processTrianglesThread();

      __host__ virtual void extractIsoSurface(VoxelContainer<T>& container) = 0;
      __device__ Vertex vertexInterp(const float isolevel,
                                     const float3& p1,
                                     const float3& p2,
                                     const float d1,
                                     const float d2,
                                     const uchar3& c1,
                                     const uchar3& c2);
      __device__ uint append();
      __device__ void appendTriangle(const Triangle& triangle);

      const Eigen::MatrixXd& getVertices() const {
        return vertices_;
      }

      const Eigen::MatrixXi& getFaces() const {
        return faces_;
      }

      const Eigen::MatrixXd& getColors() const {
        return colors_;
      }

      Triangle* d_triangles_    = nullptr;
      Triangle* h_triangles_    = nullptr;
      uint* d_trianglesCounter_ = nullptr;
      uint num_triangles_;
      uint max_num_triangles_mesh_;
      bool viewer_active_;
      float vertices_merging_threshold_ = 0.f;
      Eigen::MatrixXd vertices_;
      Eigen::MatrixXi faces_;
      Eigen::MatrixXd colors_;
      std::unique_ptr<std::thread> triangle_processing_thread_ = nullptr;
      std::mutex mesh_mutex_;
      std::condition_variable mesh_cv_;
      std::atomic<bool> mesh_ready_ = false;
      std::atomic<bool> merge_mesh_ = false;
    };

    template class MeshExtractor<Voxel>;
    using GeometricMeshExtractor = MeshExtractor<Voxel>;
  } // namespace cugeoutils
} // namespace cupanutils
