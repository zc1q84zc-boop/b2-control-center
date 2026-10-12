#pragma once

#include <array>

namespace b2 {

enum class SafetyState { NORMAL, SAFE_DAMPING };

struct ProcessedAction {
  std::array<double, 12> q_desired{};
  std::array<double, 12> kp{};
  std::array<double, 12> kd{};
  std::array<double, 12> tau{};
};

class HardwareAdapter {
 public:
  void update_safety_state(const std::array<double, 3> &safety_vector);
  [[nodiscard]] ProcessedAction process_action(
      const ProcessedAction &requested);
  [[nodiscard]] SafetyState safety_state() const;

 private:
  SafetyState safety_state_{SafetyState::NORMAL};
};

}  // namespace b2
