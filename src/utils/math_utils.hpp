#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace b2::utils {

template <std::size_t N>
inline std::array<double, N> clip_command(
    const std::array<double, N> &target_rad,
    const std::array<double, N> &previous_rad,
    const std::array<double, N> &q_min_rad,
    const std::array<double, N> &q_max_rad,
    const std::array<double, N> &dq_max_rad_s,
    const std::array<double, N> &max_angle_step_rad,
    double dt_s) {
  if (!std::isfinite(dt_s) || dt_s <= 0.0) {
    throw std::invalid_argument("dt_s must be finite and positive");
  }

  std::array<double, N> result{};
  for (std::size_t i = 0; i < N; ++i) {
    if (!std::isfinite(target_rad[i]) || !std::isfinite(previous_rad[i]) ||
        !std::isfinite(q_min_rad[i]) || !std::isfinite(q_max_rad[i]) ||
        !std::isfinite(dq_max_rad_s[i]) ||
        !std::isfinite(max_angle_step_rad[i]) ||
        q_min_rad[i] > q_max_rad[i] ||
        previous_rad[i] < q_min_rad[i] ||
        previous_rad[i] > q_max_rad[i] || dq_max_rad_s[i] < 0.0 ||
        max_angle_step_rad[i] < 0.0) {
      throw std::invalid_argument("invalid joint command limit");
    }

    const double bounded_target =
        std::clamp(target_rad[i], q_min_rad[i], q_max_rad[i]);
    const double allowed_step =
        std::min(max_angle_step_rad[i], dq_max_rad_s[i] * dt_s);
    const double delta =
        std::clamp(bounded_target - previous_rad[i], -allowed_step,
                   allowed_step);
    result[i] =
        std::clamp(previous_rad[i] + delta, q_min_rad[i], q_max_rad[i]);
  }
  return result;
}

template <std::size_t N>
inline std::array<double, N> linear_interpolate(
    const std::array<double, N> &initial_rad,
    const std::array<double, N> &target_rad,
    double elapsed_s,
    double duration_s) {
  if (!std::isfinite(elapsed_s) || !std::isfinite(duration_s) ||
      duration_s <= 0.0) {
    throw std::invalid_argument(
        "elapsed_s must be finite and duration_s must be finite and positive");
  }
  const double alpha = std::clamp(elapsed_s / duration_s, 0.0, 1.0);
  std::array<double, N> result{};
  for (std::size_t i = 0; i < N; ++i) {
    if (!std::isfinite(initial_rad[i]) || !std::isfinite(target_rad[i])) {
      throw std::invalid_argument("interpolation endpoints must be finite");
    }
    result[i] = initial_rad[i] + alpha * (target_rad[i] - initial_rad[i]);
  }
  return result;
}

}  // namespace b2::utils
