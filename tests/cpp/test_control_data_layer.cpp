#include "hardware_adapter.hpp"
#include "utils/math_utils.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
using Vector12 = std::array<double, 12>;

void require(bool condition) {
  if (!condition) throw std::runtime_error("control data layer test failed");
}

void test_clip_command() {
  const Vector12 previous{};
  Vector12 target{};
  target.fill(2.0);
  Vector12 lower{};
  lower.fill(-1.0);
  Vector12 upper{};
  upper.fill(1.0);
  Vector12 velocity{};
  velocity.fill(2.0);
  Vector12 step{};
  step.fill(0.1);
  const auto result = b2::utils::clip_command(
      target, previous, lower, upper, velocity, step, 0.02);
  for (double value : result) require(std::abs(value - 0.04) < 1e-12);
  velocity.fill(0.0);
  const auto held = b2::utils::clip_command(
      target, previous, lower, upper, velocity, step, 0.02);
  for (double value : held) require(value == 0.0);

  Vector12 invalid_previous{};
  invalid_previous[0] = 2.0;
  bool rejected = false;
  try {
    (void)b2::utils::clip_command(
        target, invalid_previous, lower, upper, velocity, step, 0.02);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected);
}

void test_linear_interpolate() {
  Vector12 initial{};
  Vector12 target{};
  target.fill(0.4);
  const auto halfway =
      b2::utils::linear_interpolate(initial, target, 1.0, 2.0);
  for (double value : halfway) require(std::abs(value - 0.2) < 1e-12);
  const auto finished =
      b2::utils::linear_interpolate(initial, target, 3.0, 2.0);
  for (double value : finished) require(std::abs(value - 0.4) < 1e-12);
}

void test_safe_damping_is_latched() {
  b2::HardwareAdapter adapter;
  b2::ProcessedAction requested;
  requested.kp.fill(10.0);
  requested.kd.fill(1.0);
  requested.tau.fill(5.0);
  require(adapter.process_action(requested).kp[0] == 10.0);
  adapter.update_safety_state({-0.61, 0.0, 100.0});
  require(adapter.safety_state() == b2::SafetyState::SAFE_DAMPING);
  adapter.update_safety_state({0.0, 0.0, 0.0});
  require(adapter.safety_state() == b2::SafetyState::SAFE_DAMPING);
  const auto safe = adapter.process_action(requested);
  for (std::size_t i = 0; i < 12; ++i) {
    require(safe.kp[i] == 0.0);
    require(safe.kd[i] == 3.0);
    require(safe.tau[i] == 0.0);
  }
}

void test_nonfinite_tilt_fails_safe() {
  b2::HardwareAdapter adapter;
  adapter.update_safety_state(
      {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0});
  require(adapter.safety_state() == b2::SafetyState::SAFE_DAMPING);
}

void test_nonfinite_action_fails_safe() {
  b2::HardwareAdapter adapter;
  b2::ProcessedAction requested;
  requested.q_desired[3] = std::numeric_limits<double>::infinity();
  const auto safe = adapter.process_action(requested);
  require(adapter.safety_state() == b2::SafetyState::SAFE_DAMPING);
  for (std::size_t i = 0; i < 12; ++i) {
    require(safe.q_desired[i] == 0.0);
    require(safe.kp[i] == 0.0);
    require(safe.kd[i] == 3.0);
    require(safe.tau[i] == 0.0);
  }
}
}  // namespace

int main() {
  test_clip_command();
  test_linear_interpolate();
  test_safe_damping_is_latched();
  test_nonfinite_tilt_fails_safe();
  test_nonfinite_action_fails_safe();
  return 0;
}
