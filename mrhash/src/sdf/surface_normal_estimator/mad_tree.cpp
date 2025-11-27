// Copyright 2024 R(obots) V(ision) and P(erception) group
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "mad_tree.h"

#include <Eigen/Eigenvalues>
#include <fstream>
#include <future>

MADtree::MADtree(const ContainerTypePtr vec,
                 const IteratorType begin,
                 const IteratorType end,
                 const double b_max,
                 const double b_min,
                 const int level,
                 const int max_parallel_level,
                 MADtree* parent,
                 MADtree* plane_predecessor) {
  build(vec, begin, end, b_max, b_min, level, max_parallel_level, parent, plane_predecessor);
}

void MADtree::build(const ContainerTypePtr vec,
                    const IteratorType begin,
                    const IteratorType end,
                    const double b_max,
                    const double b_min,
                    const int level,
                    const int max_parallel_level,
                    MADtree* parent,
                    MADtree* plane_predecessor) {
  begin_  = begin;
  end_    = end;
  parent_ = parent;
  Eigen::Matrix3d cov;
  computeMeanAndCovariance(mean_, cov, begin, end);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
  es.computeDirect(cov);
  eigenvectors_ = es.eigenvectors();
  num_points_   = computeBoundingBox(bbox_, mean_, eigenvectors_.transpose(), begin, end);

  if (bbox_(2) < b_max) {
    if (plane_predecessor) {
      eigenvectors_.col(0) = plane_predecessor->eigenvectors_.col(0);
    } else {
      if (num_points_ < 3) {
        MADtree* node = this;
        while (node->parent_ && node->num_points_ < 3)
          node = node->parent_;
        eigenvectors_.col(0) = node->eigenvectors_.col(0);
      }
    }

    Eigen::Vector3d& nearest_point = *begin;
    double shortest_dist           = std::numeric_limits<double>::max();
    for (IteratorType it = begin; it != end; ++it) {
      const Eigen::Vector3d& v = *it;
      const double dist        = (v - mean_).norm();
      if (dist < shortest_dist) {
        nearest_point = v;
        shortest_dist = dist;
      }
    }
    mean_ = nearest_point;
    // *************** BEGIN OF BEAM SIMULATION *************** //
    // const double beam_divergence_deg = 0.35; // os0
    const double beam_divergence_deg = 0.18; // os1

    const double beam_divergence = beam_divergence_deg * M_PI / 180.0;
    const int root_num_beams     = 11; // must be odd
    const double beam_delta      = beam_divergence / double(root_num_beams - 1);

    const double mean_azimuth            = std::atan2(mean_(1), mean_(0));
    const double mean_elevation          = std::asin(mean_(2) / mean_.norm());
    const Eigen::Vector3d mean_direction = mean_ / mean_.norm();

    std::vector<double> waveform_ranges;
    for (int i = -root_num_beams / 2; i <= root_num_beams / 2; ++i) {
      const double azimuth = mean_azimuth + i * beam_delta;
      for (int j = -root_num_beams / 2; j <= root_num_beams / 2; ++j) {
        const double elevation = mean_elevation + j * beam_delta;
        const Eigen::Vector3d direction(
          std::cos(azimuth) * std::cos(elevation), std::sin(azimuth) * std::cos(elevation), std::sin(elevation));
        const Eigen::Vector3d direction_normalized = direction / direction.norm();

        const double mean_dot = mean_direction.dot(direction_normalized);
        const double angle    = std::acos(mean_dot);

        // If line is inside the cone
        if (angle < beam_divergence / 2.0) {
          Eigen::Vector3d plane_point    = mean_;
          Eigen::Vector3d plane_normal   = eigenvectors_.col(0);
          Eigen::Vector3d line_direction = direction_normalized;

          // Compute the intersection point of the plane and the line
          const double denominator = plane_normal.dot(line_direction);
          if (std::abs(denominator) < 1e-6) {
            continue;
          }
          const double numerator                   = plane_normal.dot(plane_point);
          const double d                           = numerator / denominator;
          const Eigen::Vector3d intersection_point = d * line_direction;

          const double range = intersection_point.norm();
          waveform_ranges.push_back(range);
        }
      }
    }

    double std_dev          = 0;
    const double mean_range = mean_.norm();
    for (const double& range : waveform_ranges) {
      std_dev += std::pow(range - mean_range, 2);
    }
    if (!waveform_ranges.empty()) {
      std_dev = std::sqrt(std_dev / waveform_ranges.size());
    }
    const double meas_sucks_with_std_dev = 0.25;          // more than half of sub-beams are below 0.25m of error
    weight_ = std::min(std_dev, meas_sucks_with_std_dev); // cut the std_dev to be at most bad as meas_sucks_with_std_dev
    weight_ = weight_ / meas_sucks_with_std_dev;          // normalize weight_ to be between 0 and 1
    weight_ = 1. - weight_;                               // flip the weight_

    // *************** END OF BEAM SIMULATION *************** //
    return;
  }
  if (!plane_predecessor) {
    if (bbox_(0) < b_min)
      plane_predecessor = this;
  }

  const Eigen::Vector3d& _split_plane_normal = eigenvectors_.col(2);
  IteratorType middle =
    split(begin, end, [&](const Eigen::Vector3d& p) -> bool { return (p - mean_).dot(_split_plane_normal) < double(0); });

  if (level >= max_parallel_level) {
    left_ = new MADtree(vec, begin, middle, b_max, b_min, level + 1, max_parallel_level, this, plane_predecessor);

    right_ = new MADtree(vec, middle, end, b_max, b_min, level + 1, max_parallel_level, this, plane_predecessor);
  } else {
    std::future<MADtree*> l =
      std::async(MADtree::makeSubtree, vec, begin, middle, b_max, b_min, level + 1, max_parallel_level, this, plane_predecessor);

    std::future<MADtree*> r =
      std::async(MADtree::makeSubtree, vec, middle, end, b_max, b_min, level + 1, max_parallel_level, this, plane_predecessor);
    left_  = l.get();
    right_ = r.get();
  }
}

MADtree* MADtree::makeSubtree(const ContainerTypePtr vec,
                              const IteratorType begin,
                              const IteratorType end,
                              const double b_max,
                              const double b_min,
                              const int level,
                              const int max_parallel_level,
                              MADtree* parent,
                              MADtree* plane_predecessor) {
  return new MADtree(vec, begin, end, b_max, b_min, level, max_parallel_level, parent, plane_predecessor);
}

const MADtree* MADtree::bestMatchingLeafFast(const Eigen::Vector3d& query) const {
  const MADtree* node = this;
  while (node->left_ || node->right_) {
    const Eigen::Vector3d& _split_plane_normal = node->eigenvectors_.col(2);
    node = ((query - node->mean_).dot(_split_plane_normal) < double(0)) ? node->left_ : node->right_;
  }

  return node;
}

void MADtree::getLeafs(std::back_insert_iterator<std::vector<MADtree*>> it) {
  if (!left_ && !right_) {
    ++it = this;
    return;
  }
  if (left_)
    left_->getLeafs(it);
  if (right_)
    right_->getLeafs(it);
}

void MADtree::applyTransform(const Eigen::Matrix3d& r, const Eigen::Vector3d& t) {
  mean_         = r * mean_ + t;
  eigenvectors_ = r * eigenvectors_;
  if (left_)
    left_->applyTransform(r, t);
  if (right_)
    right_->applyTransform(r, t);
}