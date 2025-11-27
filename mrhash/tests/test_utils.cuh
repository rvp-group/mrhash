#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <gtest/gtest.h>

// ! generate random depth between bound, only needed for test stuff
float generateRandomDepth(const float min_depth, const float max_depth) {
  return (min_depth + 0.1f) +
         static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / ((max_depth - 0.1) - (min_depth + 0.1))));
}

float3 generateRandomPoint(const float min_bound, const float max_bound) {
  return make_float3(generateRandomDepth(min_bound, max_bound),
                     generateRandomDepth(min_bound, max_bound),
                     generateRandomDepth(min_bound, max_bound));
}

// ! make a circular trajectory on costant motion model, assumed camera optical frame, z-forward
// ! motion is on plane x-z
std::vector<Eigen::Isometry3f> makeCameraCircularTrajectory(const float t_step, const int steps) {
  std::vector<Eigen::Isometry3f> path;
  const float angle_step   = 2.f * M_PI / (float) steps;
  Eigen::Isometry3f T_abs  = Eigen::Isometry3f::Identity();
  Eigen::Isometry3f T_step = Eigen::Isometry3f::Identity();
  T_step.linear() << cos(angle_step), 0.f, sin(angle_step), 0.f, 1.f, 0.f, -sin(angle_step), 0.f, cos(angle_step);
  T_step.translation() << t_step, 0, t_step;
  for (int i = 0; i < steps + 1; ++i) {
    T_abs = T_abs * T_step;
    path.push_back(T_abs);
  }
  return path;
}
