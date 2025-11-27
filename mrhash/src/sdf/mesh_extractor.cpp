#include "mesh_extractor.cuh"
#include <set>
#include <stdexcept>

namespace cupanutils {
  namespace cugeoutils {

    template <typename T>
    void MeshExtractor<T>::processTriangles() {
      uint num_vertices            = num_triangles_ * 3;
      Eigen::MatrixXd new_vertices = Eigen::MatrixXd::Zero(num_vertices, 3);
      Eigen::MatrixXd new_colors   = Eigen::MatrixXd::Zero(num_vertices, 3);
      Eigen::MatrixXi new_faces    = Eigen::MatrixXi::Zero(num_triangles_, 3);
      for (int i = 0; i < num_triangles_; ++i) {
        const auto& t = h_triangles_[i];
        new_vertices.row(i * 3 + 0) << t.v0.p.x, t.v0.p.y, t.v0.p.z;
        new_vertices.row(i * 3 + 1) << t.v1.p.x, t.v1.p.y, t.v1.p.z;
        new_vertices.row(i * 3 + 2) << t.v2.p.x, t.v2.p.y, t.v2.p.z;
        new_colors.row(i * 3 + 0) << t.v0.c.x, t.v0.c.y, t.v0.c.z;
        new_colors.row(i * 3 + 1) << t.v1.c.x, t.v1.c.y, t.v1.c.z;
        new_colors.row(i * 3 + 2) << t.v2.c.x, t.v2.c.y, t.v2.c.z;
        new_faces.row(i) << i * 3, i * 3 + 1, i * 3 + 2;
      }
      if (!merge_mesh_) {
        vertices_ = new_vertices;
        faces_    = new_faces;
        colors_   = new_colors;
      } else {
        if (vertices_.rows() == 0 && faces_.rows() == 0 && colors_.rows() == 0) {
          vertices_ = new_vertices;
          faces_    = new_faces;
          colors_   = new_colors;
        } else {
          combine({vertices_, new_vertices}, {faces_, new_faces}, vertices_, faces_);
          Eigen::MatrixXd combined_colors(colors_.rows() + new_colors.rows(), 3);
          combined_colors << colors_, new_colors;
          colors_ = combined_colors;
        }
      }
      Eigen::VectorXi map_old_to_new, map_new_to_old;
      Eigen::MatrixXd unique_vertices;
      Eigen::MatrixXi unique_faces;
      removeDuplicateVerticesTriangle(
        vertices_, faces_, vertices_merging_threshold_, unique_vertices, unique_faces, map_old_to_new);

      vertices_ = unique_vertices;
      faces_    = unique_faces;
      Eigen::MatrixXd unique_colors(unique_vertices.rows(), 3);

      std::vector<bool> assigned(unique_vertices.rows(), false);

      for (int i = 0; i < map_old_to_new.size(); ++i) {
        int new_idx = map_old_to_new[i];
        if (!assigned[new_idx]) {
          unique_colors.row(new_idx) = colors_.row(i);
          assigned[new_idx]          = true;
        }
      }
      colors_ = unique_colors;

      Eigen::Array<bool, Eigen::Dynamic, 1> keep = (faces_.col(0).array() != faces_.col(1).array()) &&
                                                   (faces_.col(0).array() != faces_.col(2).array()) &&
                                                   (faces_.col(1).array() != faces_.col(2).array());

      Eigen::MatrixXi filtered_faces(keep.count(), faces_.cols());
      int index = 0;
      for (int i = 0; i < faces_.rows(); ++i) {
        if (keep(i)) {
          filtered_faces.row(index++) = faces_.row(i);
        }
      }
      faces_ = filtered_faces;

      removeDuplicateFacesTriangle(faces_, unique_faces);
      faces_ = unique_faces;
    }

    template <typename T>
    void MeshExtractor<T>::processTrianglesThread() {
      while (true) {
        {
          std::unique_lock<std::mutex> lock(mesh_mutex_);
          mesh_cv_.wait(lock);
        }
        if (mesh_ready_) {
          processTriangles();
          mesh_ready_ = false;
        } else {
          break;
        }
      }
    }

    template <typename T>
    void MeshExtractor<T>::extractMesh(VoxelContainer<T>& container) {
      container.flatAndReduceHashTable();
      extractIsoSurface(container);
    }

    template <typename T>
    void MeshExtractor<T>::combine(const std::vector<Eigen::MatrixXd>& vertices_src,
                                   const std::vector<Eigen::MatrixXi>& faces_src,
                                   Eigen::MatrixXd& vertices_dst,
                                   Eigen::MatrixXi& faces_dst) {
      if (vertices_src.size() != faces_src.size()) {
        throw std::runtime_error("List of vertices size != List of faces size");
      }

      if (vertices_src.size() == 0) {
        return;
      }

      const int vert_dim = vertices_src[0].cols();
      const int face_dim = faces_src[0].cols();

      std::vector<int> vert_sizes(vertices_src.size());
      std::vector<int> faces_sizes(vertices_src.size());
      int total_vert  = 0;
      int total_faces = 0;
      for (int i = 0; i < vertices_src.size(); i++) {
        if (vertices_src[i].cols() != vert_dim) {
          throw std::runtime_error("mismatch in vertex dim");
        }
        if (faces_src[i].cols() != face_dim) {
          throw std::runtime_error("mismatch in face dim");
        }
        vert_sizes[i]  = vertices_src[i].rows();
        faces_sizes[i] = faces_src[i].rows();
        total_vert += vertices_src[i].rows();
        total_faces += faces_src[i].rows();
      }

      vertices_dst.resize(total_vert, vert_dim);
      faces_dst.resize(total_faces, face_dim);

      int curr_vert  = 0;
      int curr_faces = 0;

      for (int i = 0; i < vertices_src.size(); i++) {
        const auto& vert_i  = vertices_src[i];
        const int n_vert_i  = vert_i.rows();
        const auto& faces_i = faces_src[i];
        const int n_faces_i = faces_i.rows();
        if (n_faces_i > 0) {
          faces_dst.block(curr_faces, 0, n_faces_i, face_dim) = faces_i.array() + curr_vert;
          curr_faces += n_faces_i;
        }
        if (n_vert_i > 0) {
          vertices_dst.block(curr_vert, 0, n_vert_i, vert_dim) = vert_i;
          curr_vert += n_vert_i;
        }
      }
    }

    template <typename T>
    void MeshExtractor<T>::removeDuplicateFacesTriangle(const Eigen::MatrixXi& faces_src, Eigen::MatrixXi& faces_dst) {
      if (faces_src.cols() != 3) {
        throw std::runtime_error("Expecting triangular meshes (input faces cols!=3)");
      }

      std::set<std::tuple<int, int, int>> seen;
      std::vector<std::tuple<int, int, int>> unique_faces;

      for (int i = 0; i < faces_src.rows(); ++i) {
        auto face_tuple = std::make_tuple(faces_src(i, 0), faces_src(i, 1), faces_src(i, 2));
        if (seen.find(face_tuple) == seen.end()) {
          seen.insert(face_tuple);
          unique_faces.push_back(face_tuple);
        }
      }

      faces_dst.resize(unique_faces.size(), 3);
      for (int i = 0; i < unique_faces.size(); ++i) {
        faces_dst(i, 0) = std::get<0>(unique_faces[i]);
        faces_dst(i, 1) = std::get<1>(unique_faces[i]);
        faces_dst(i, 2) = std::get<2>(unique_faces[i]);
      }
    }

    template <typename T>
    void MeshExtractor<T>::removeDuplicateVerticesTriangle(const Eigen::MatrixXd& vertices_src,
                                                           const Eigen::MatrixXi& faces_src,
                                                           const double epsilon,
                                                           Eigen::MatrixXd& vertices_dst,
                                                           Eigen::MatrixXi& faces_dst,
                                                           Eigen::VectorXi& old_to_new_mapping) {
      const size_t num_vertices = vertices_src.rows();
      old_to_new_mapping.resize(num_vertices);

      if (epsilon == 0.0) {
        std::unordered_map<Eigen::Vector3d, int, Vector3dHash, Vector3dEqual> vertex_map;
        vertex_map.reserve(num_vertices);
        std::vector<Eigen::Vector3d> unique_vertices;
        unique_vertices.reserve(num_vertices);

        for (int i = 0; i < num_vertices; ++i) {
          Eigen::Vector3d v = vertices_src.row(i);
          auto it           = vertex_map.find(v);
          if (it != vertex_map.end()) {
            old_to_new_mapping(i) = it->second;
          } else {
            int new_idx   = static_cast<int>(unique_vertices.size());
            vertex_map[v] = new_idx;
            unique_vertices.push_back(v);
            old_to_new_mapping(i) = new_idx;
          }
        }

        vertices_dst.resize(unique_vertices.size(), 3);
        for (int i = 0; i < unique_vertices.size(); ++i)
          vertices_dst.row(i) = unique_vertices[i];

        faces_dst.resize(faces_src.rows(), faces_src.cols());
        for (int i = 0; i < faces_src.rows(); ++i)
          for (int j = 0; j < faces_src.cols(); ++j) {
            int old_idx = faces_src(i, j);
            if (old_idx < 0 || old_idx >= old_to_new_mapping.size()) {
              throw std::runtime_error("Face vertex index out of range in removeDuplicateVerticesTriangle");
            }
            faces_dst(i, j) = old_to_new_mapping(old_idx);
          }
      } else {
        const double inv_eps = 1.0 / epsilon;

        std::unordered_map<Eigen::Vector3i, int, Vec3iHash> vertex_map;
        vertex_map.reserve(num_vertices);
        std::vector<Eigen::Vector3d> unique_vertices;
        unique_vertices.reserve(num_vertices);

        for (int i = 0; i < num_vertices; ++i) {
          Eigen::Vector3d v  = vertices_src.row(i);
          Eigen::Vector3i qv = (v * inv_eps).array().floor().cast<int>();

          auto it = vertex_map.find(qv);
          if (it != vertex_map.end()) {
            old_to_new_mapping(i) = it->second;
          } else {
            int new_idx    = static_cast<int>(unique_vertices.size());
            vertex_map[qv] = new_idx;
            unique_vertices.push_back(v);
            old_to_new_mapping(i) = new_idx;
          }
        }

        vertices_dst.resize(unique_vertices.size(), 3);
        for (int i = 0; i < unique_vertices.size(); ++i)
          vertices_dst.row(i) = unique_vertices[i];

        faces_dst.resize(faces_src.rows(), faces_src.cols());
        for (int i = 0; i < faces_src.rows(); ++i)
          for (int j = 0; j < faces_src.cols(); ++j) {
            int old_idx = faces_src(i, j);
            if (old_idx < 0 || old_idx >= old_to_new_mapping.size()) {
              throw std::runtime_error("Face vertex index out of range in removeDuplicateVerticesTriangle");
            }
            faces_dst(i, j) = old_to_new_mapping(old_idx);
          }
      }
    }
  } // namespace cugeoutils
} // namespace cupanutils
