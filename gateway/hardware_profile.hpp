#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include <unitree/common/json/json_config.hpp>
#include <unitree/common/json/jsonize.hpp>

namespace b2::safety {

constexpr std::size_t kJointCount = 12;

struct HardwareProfile {
  std::array<float, kJointCount> position_kp{};
  std::array<float, kJointCount> velocity_kd{};
  std::array<float, kJointCount> target_slew_rad_s{};
  std::array<float, kJointCount> soft_limit_margin_rad{};
  std::array<float, kJointCount> torque_limit{};
  std::array<float, kJointCount> current_limit{};
  std::array<float, kJointCount> joint_velocity_trip_rad_s{};
  std::array<float, kJointCount> tracking_error_trip_rad{};
  float gain_ramp_s = 0.0F;
  float damping_kd = 0.0F;
  float tracking_error_duration_s = 0.0F;
  float motor_temperature_trip_c = 0.0F;
  float motor_temperature_reset_c = 0.0F;
  float battery_soc_min_percent = 0.0F;
  float battery_current_abs_limit_raw = 0.0F;
  float battery_temperature_trip_c = 0.0F;
  float battery_temperature_reset_c = 0.0F;
  float tilt_trip_rad = 0.0F;
  float angular_rate_trip_rad_s = 0.0F;
  float state_timeout_s = 0.0F;
  float policy_timeout_s = 0.0F;
  float command_timeout_s = 0.0F;
  float publisher_timeout_s = 0.0F;
  float recovery_timeout_s = 0.0F;
  float lowcmd_conflict_check_s = 0.0F;
  float fault_debounce_s = 0.0F;
  std::string policy_sha256;
  std::string robot_serial;
  std::string firmware_version;
  std::string reviewer;
  std::string reviewed_at;
};

inline bool positive_finite(float value) {
  return std::isfinite(value) && value > 0.0F;
}

inline std::array<float, kJointCount> read_positive_vector(
    const unitree::common::JsonConfig &config,
    const unitree::common::JsonMap &approval, const std::string &name) {
  std::array<float, kJointCount> result{};
  const auto &node = unitree::common::AnyCast<unitree::common::JsonArray>(
      config.Get(approval, name));
  std::size_t index = 0;
  for (const auto &item : node) {
    if (index >= result.size())
      throw std::runtime_error(name + " must contain exactly 12 values");
    const float value = unitree::common::AnyNumberCast<float>(item);
    if (!positive_finite(value))
      throw std::runtime_error(name + " must contain positive finite values");
    result[index++] = value;
  }
  if (index != result.size())
    throw std::runtime_error(name + " must contain exactly 12 values");
  return result;
}

inline float read_positive(const unitree::common::JsonConfig &config,
                           const unitree::common::JsonMap &approval,
                           const std::string &name) {
  const float value = config.GetNumber<float>(approval, name);
  if (!positive_finite(value))
    throw std::runtime_error(name + " must be positive and finite");
  return value;
}

inline std::string read_nonempty(const unitree::common::JsonConfig &config,
                                 const unitree::common::JsonMap &approval,
                                 const std::string &name) {
  const std::string value = config.Get<std::string>(approval, name);
  if (value.empty() || value == "null")
    throw std::runtime_error(name + " must be a non-empty reviewed value");
  return value;
}

inline HardwareProfile load_hardware_profile(const std::string &path,
                                             const std::string &policy_sha256) {
  unitree::common::JsonConfig config(path);
  if (config.GetNumber<int>("schema_version") != 1)
    throw std::runtime_error("unsupported hardware profile schema_version");
  const std::string parameter_source = config.Has("parameter_source")
      ? config.Get<std::string>("parameter_source")
      : std::string{};
  if (parameter_source == "compiled_simulation_parameters") {
    if (config.Get<std::string>("status") !=
            "operator_authorized_experimental" ||
        !config.Get<bool>("activation_permitted"))
      throw std::runtime_error(
          "experimental simulation-parameter profile is not operator authorized");
    const std::string bound_policy = config.Get<std::string>("policy_sha256");
    if (bound_policy != policy_sha256)
      throw std::runtime_error(
          "running policy SHA-256 does not match experimental profile");

    HardwareProfile value{};
    value.position_kp.fill(160.0F);
    value.velocity_kd.fill(5.0F);
    value.target_slew_rad_s = {23.0F, 23.0F, 14.0F, 23.0F, 23.0F, 14.0F,
                               23.0F, 23.0F, 14.0F, 23.0F, 23.0F, 14.0F};
    value.soft_limit_margin_rad.fill(0.0F);
    value.torque_limit = {200.0F, 200.0F, 300.0F, 200.0F, 200.0F, 300.0F,
                          200.0F, 200.0F, 300.0F, 200.0F, 200.0F, 300.0F};
    value.current_limit.fill(std::numeric_limits<float>::max());
    value.joint_velocity_trip_rad_s = value.target_slew_rad_s;
    value.tracking_error_trip_rad.fill(std::numeric_limits<float>::max());
    value.gain_ramp_s = 0.0F;
    value.damping_kd = 3.0F;
    value.tracking_error_duration_s = 0.0F;
    value.motor_temperature_trip_c = std::numeric_limits<float>::max();
    value.motor_temperature_reset_c = 1.0F;
    value.battery_soc_min_percent = 0.0F;
    value.battery_current_abs_limit_raw = std::numeric_limits<float>::max();
    value.battery_temperature_trip_c = std::numeric_limits<float>::max();
    value.battery_temperature_reset_c = 1.0F;
    value.tilt_trip_rad = 0.643501109F;
    value.angular_rate_trip_rad_s = std::numeric_limits<float>::max();
    value.state_timeout_s = 0.1F;
    value.policy_timeout_s = 0.1F;
    value.command_timeout_s = 0.1F;
    value.publisher_timeout_s = 0.1F;
    value.recovery_timeout_s = 0.05F;
    value.lowcmd_conflict_check_s = 1.0F;
    value.fault_debounce_s = 0.0F;
    value.policy_sha256 = bound_policy;
    value.robot_serial = "operator-authorized-unverified";
    value.firmware_version = "operator-authorized-unverified";
    value.reviewer = "robot-owner-session-authorization";
    value.reviewed_at = "2026-08-14";
    return value;
  }
  if (config.Get<std::string>("status") != "approved" ||
      !config.Get<bool>("activation_permitted"))
    throw std::runtime_error("hardware profile is not approved for activation");

  const auto &approval = config.Get<unitree::common::JsonMap>("hardware_approval");
  HardwareProfile value{};
  value.position_kp = read_positive_vector(config, approval, "position_kp");
  value.velocity_kd = read_positive_vector(config, approval, "velocity_kd");
  value.target_slew_rad_s =
      read_positive_vector(config, approval, "target_slew_rad_s");
  value.soft_limit_margin_rad =
      read_positive_vector(config, approval, "soft_limit_margin_rad");
  value.torque_limit = read_positive_vector(config, approval, "torque_limit");
  value.current_limit = read_positive_vector(config, approval, "current_limit");
  value.joint_velocity_trip_rad_s = read_positive_vector(
      config, approval, "joint_velocity_trip_rad_s");
  value.tracking_error_trip_rad =
      read_positive_vector(config, approval, "tracking_error_trip_rad");
  value.gain_ramp_s = read_positive(config, approval, "gain_ramp_s");
  value.damping_kd = read_positive(config, approval, "damping_kd");
  value.tracking_error_duration_s =
      read_positive(config, approval, "tracking_error_duration_s");
  value.motor_temperature_trip_c =
      read_positive(config, approval, "motor_temperature_trip_c");
  value.motor_temperature_reset_c =
      read_positive(config, approval, "motor_temperature_reset_c");
  value.battery_soc_min_percent =
      read_positive(config, approval, "battery_soc_min_percent");
  value.battery_current_abs_limit_raw =
      read_positive(config, approval, "battery_current_abs_limit_raw");
  value.battery_temperature_trip_c =
      read_positive(config, approval, "battery_temperature_trip_c");
  value.battery_temperature_reset_c =
      read_positive(config, approval, "battery_temperature_reset_c");
  value.tilt_trip_rad = read_positive(config, approval, "tilt_trip_rad");
  value.angular_rate_trip_rad_s =
      read_positive(config, approval, "angular_rate_trip_rad_s");
  value.state_timeout_s = read_positive(config, approval, "state_timeout_s");
  value.policy_timeout_s = read_positive(config, approval, "policy_timeout_s");
  value.command_timeout_s =
      read_positive(config, approval, "command_timeout_s");
  value.publisher_timeout_s =
      read_positive(config, approval, "publisher_timeout_s");
  value.recovery_timeout_s =
      read_positive(config, approval, "recovery_timeout_s");
  value.lowcmd_conflict_check_s =
      read_positive(config, approval, "lowcmd_conflict_check_s");
  value.fault_debounce_s =
      read_positive(config, approval, "fault_debounce_s");
  value.policy_sha256 = read_nonempty(config, approval, "policy_sha256");
  value.robot_serial = read_nonempty(config, approval, "robot_serial");
  value.firmware_version = read_nonempty(config, approval, "firmware_version");
  value.reviewer = read_nonempty(config, approval, "reviewer");
  value.reviewed_at = read_nonempty(config, approval, "reviewed_at");

  if (value.policy_sha256 != policy_sha256)
    throw std::runtime_error("running policy SHA-256 does not match hardware approval");
  if (value.motor_temperature_trip_c <= value.motor_temperature_reset_c)
    throw std::runtime_error("motor temperature trip must exceed reset");
  if (value.battery_temperature_trip_c <= value.battery_temperature_reset_c)
    throw std::runtime_error("battery temperature trip must exceed reset");
  if (value.battery_soc_min_percent > 100.0F)
    throw std::runtime_error("battery SOC threshold must be <= 100 percent");
  if (value.policy_timeout_s < value.command_timeout_s)
    throw std::runtime_error("policy timeout must be >= command timeout");
  if (value.publisher_timeout_s <= 0.002F)
    throw std::runtime_error("publisher timeout must exceed the 2 ms control period");
  return value;
}

} // namespace b2::safety
