#pragma once
#include <Eigen/Core>
#include <memory>
#include <set>
#include <vector>

namespace utils {

  using Vector4f = Eigen::Matrix<float, 4, 1>;

  class PointCloud {
  public:
    PointCloud() {
    }

    PointCloud(const std::vector<Eigen::Vector3f>& points) {
      points_ = points;
    }

    PointCloud(PointCloud&& pc) {
      points_     = std::move(pc.points_);
      normals_    = std::move(pc.normals_);
      colors_     = std::move(pc.colors_);
      weights_    = std::move(pc.weights_);
      sdf_values_ = std::move(pc.sdf_values_);
    }
    void operator=(PointCloud&& pc) {
      points_     = std::move(pc.points_);
      normals_    = std::move(pc.normals_);
      colors_     = std::move(pc.colors_);
      weights_    = std::move(pc.weights_);
      sdf_values_ = std::move(pc.sdf_values_);
    }

    bool hasNormals() const {
      return normals_.size() > 0;
    }
    bool hasColors() const {
      return colors_.size() > 0;
    }
    bool hasWeights() const {
      return weights_.size() > 0;
    }
    bool hasSDFValues() const {
      return sdf_values_.size() > 0;
    }

    void clear() {
      points_.clear();
      normals_.clear();
      colors_.clear();
      weights_.clear();
      sdf_values_.clear();
    }

    bool isConsistent() const {
      bool is = true;
      if (normals_.size() > 0 && normals_.size() != points_.size())
        is = false;
      if (colors_.size() > 0 && colors_.size() != points_.size())
        is = false;
      if (weights_.size() > 0 && weights_.size() != points_.size())
        is = false;
      if (sdf_values_.size() > 0 && sdf_values_.size() != points_.size())
        is = false;
      return is;
    }

    size_t size() const {
      return points_.size();
    }

    bool isEmpty() const {
      return points_.size() == 0;
    }

    void push_back(const Eigen::Vector3f& point, const Eigen::Vector4f& color) {
      points_.push_back(point);
      colors_.push_back(color);
    }
    void push_back(const Eigen::Vector3f& point, const Eigen::Vector4f& color, float weight) {
      points_.push_back(point);
      colors_.push_back(color);
      weights_.push_back(weight);
    }
    void push_back(const Eigen::Vector3f& point, const Eigen::Vector4f& color, float weight, float sdf_value) {
      points_.push_back(point);
      colors_.push_back(color);
      weights_.push_back(weight);
      sdf_values_.push_back(sdf_value);
    }

    void remove_elements(const std::set<size_t>& indices) {
      const auto& num_indices_to_remove = indices.size();
      const auto& size_before           = size();
      points_.erase(std::remove_if(points_.begin(),
                                   points_.end(),
                                   [&](const Eigen::Vector3f& point) {
                                     int index = &point - &points_[0];
                                     return indices.find(index) != indices.end();
                                   }),
                    points_.end());
      colors_.erase(std::remove_if(colors_.begin(),
                                   colors_.end(),
                                   [&](const Eigen::Vector4f& color) {
                                     int index = &color - &colors_[0];
                                     return indices.find(index) != indices.end();
                                   }),
                    colors_.end());
      const auto& num_indices_removed = size_before - size();

      if (num_indices_to_remove != num_indices_removed) {
        throw std::runtime_error("PointCloud::remove_elements | size mismatch");
      }
    }

    std::vector<Eigen::Vector3f> points_;
    std::vector<Eigen::Vector3f> normals_;
    std::vector<Eigen::Vector4f> colors_;
    std::vector<float> weights_;
    std::vector<float> sdf_values_;
  };

  using PointCloudPtr = std::shared_ptr<PointCloud>;

} // namespace utils
