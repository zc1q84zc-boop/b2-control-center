#include "hardware_adapter.hpp"

#include <cmath>
#include <cstddef>

namespace b2 {
namespace {

bool valid_requested_action(const ProcessedAction &requested) {
  for (std::size_t i = 0; i < requested.q_desired.size(); ++i) {
    if (!std::isfinite(requested.q_desired[i]) ||
        !std::isfinite(requested.kp[i]) || requested.kp[i] < 0.0 ||
        !std::isfinite(requested.kd[i]) || requested.kd[i] < 0.0 ||
        !std::isfinite(requested.tau[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

void HardwareAdapter::update_safety_state(
    const std::array<double, 3> &safety_vector) {
  if (!std::isfinite(safety_vector[0]) ||
      !std::isfinite(safety_vector[1]) ||
      std::abs(safety_vector[0]) > 0.6 ||
      std::abs(safety_vector[1]) > 0.6) {
    safety_state_ = SafetyState::SAFE_DAMPING;
  }
}

ProcessedAction HardwareAdapter::process_action(
    const ProcessedAction &requested) {
  if (!valid_requested_action(requested)) {
    safety_state_ = SafetyState::SAFE_DAMPING;
  }
  if (safety_state_ != SafetyState::SAFE_DAMPING) return requested;

  ProcessedAction safe = requested;
  safe.q_desired.fill(0.0);
  safe.kp.fill(0.0);
  safe.kd.fill(3.0);
  safe.tau.fill(0.0);
  return safe;
}

SafetyState HardwareAdapter::safety_state() const { return safety_state_; }

}  // namespace b2
