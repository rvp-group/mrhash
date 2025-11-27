#pragma once
#include "point_cloud.h"
#include "utils.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace utils {

  class PointCloudSerializer {
  public:
    static PointCloud loadFromFile(const std::string& filename) {
      PointCloud pc;
      loadFromFile(filename, pc);
      return pc;
    }

    static void loadFromFile(const std::string& filename, PointCloud& point_cloud) {
      point_cloud.clear();
      std::string extension = utils::getFileExtension(filename);

      if (extension == "ply") {
        loadFromPLY(filename, point_cloud);
      } else {
        throw std::runtime_error("PointCloudSerializer|unknown file extension" + filename);
      }

      if (!point_cloud.isConsistent())
        throw std::runtime_error("PointCloudSerializer|inconsistent point cloud");
    }

    static void saveToFile(const std::string& filename, const std::vector<Eigen::Vector3f>& points) {
      PointCloud pc;
      pc.points_ = points;
      saveToFile(filename, pc);
    }

    static void saveToFile(const std::string& filename, const PointCloud& point_cloud) {
      if (point_cloud.isEmpty()) {
        std::cerr << "PointCloudSerializer|empty point cloud" << std::endl;
        return;
      }
      std::string extension = utils::getFileExtension(filename);
      if (extension == "ply") {
        writeToPLY(filename, point_cloud);
      } else {
        throw std::runtime_error("PointCloudSerializer|unknown file extension" + filename);
      }
    }

    /************************************************************************/
    /* Read Functions													    */
    /************************************************************************/

    static void loadFromPLY(const std::string& filename, PointCloud& pc);

    /************************************************************************/
    /* Write Functions													    */
    /************************************************************************/

    static void writeToPLY(const std::string& filename, const PointCloud& pc) {
      std::ofstream file(filename, std::ios::binary);
      if (!file.is_open())
        throw std::runtime_error("Could not open file for writing " + filename);
      file << "ply\n";
      file << "format binary_little_endian 1.0\n";
      file << "element vertex " << pc.points_.size() << "\n";
      file << "property float x\n";
      file << "property float y\n";
      file << "property float z\n";

      const bool has_sdf_values = pc.hasSDFValues();
      const bool has_weights    = pc.hasWeights();
      const bool has_colors     = pc.hasColors();
      const bool has_normals    = pc.hasNormals();

      if (has_sdf_values) {
        file << "property float sdf\n";
      }
      if (has_weights) {
        file << "property float weight\n";
      }
      if (has_colors) {
        file << "property uchar red\n";
        file << "property uchar green\n";
        file << "property uchar blue\n";
        file << "property uchar alpha\n";
      }
      if (has_normals) {
        file << "property float nx\n";
        file << "property float ny\n";
        file << "property float nz\n";
      }
      file << "end_header\n";
      file.flush();

      if (has_sdf_values || has_weights || has_colors || has_normals) {
        size_t vertexByteSize = sizeof(float) * 3;
        if (has_sdf_values)
          vertexByteSize += sizeof(float);
        if (has_weights)
          vertexByteSize += sizeof(float);
        if (has_colors)
          vertexByteSize += sizeof(unsigned char) * 4;
        if (has_normals)
          vertexByteSize += sizeof(float) * 3;
        std::vector<unsigned char> data(vertexByteSize * pc.points_.size());
        size_t byteOffset = 0;
        for (size_t i = 0; i < pc.points_.size(); i++) {
          memcpy(&data[byteOffset], &pc.points_[i], sizeof(float) * 3);
          byteOffset += sizeof(float) * 3;
          if (has_sdf_values) {
            memcpy(&data[byteOffset], &pc.sdf_values_[i], sizeof(float));
            byteOffset += sizeof(float);
          }
          if (has_weights) {
            memcpy(&data[byteOffset], &pc.weights_[i], sizeof(float));
            byteOffset += sizeof(float);
          }
          if (has_colors) {
            unsigned char color[4] = {(unsigned char) (pc.colors_[i](0) * 255),
                                      (unsigned char) (pc.colors_[i](1) * 255),
                                      (unsigned char) (pc.colors_[i](2) * 255),
                                      (unsigned char) (pc.colors_[i](3) * 255)};
            memcpy(&data[byteOffset], &color, sizeof(unsigned char) * 4);
            byteOffset += sizeof(unsigned char) * 4;
          }
          if (has_normals) {
            memcpy(&data[byteOffset], &pc.normals_[i], sizeof(float) * 3);
            byteOffset += sizeof(float) * 3;
          }
        }
        file.write((const char*) data.data(), byteOffset);
      } else {
        file.write((const char*) &pc.points_[0], sizeof(float) * 3 * pc.points_.size());
      }

      file.close();
    }
  };

} // namespace utils
