#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <unistd.h>

#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "hardware_profile.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelPublisherPtr;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;

constexpr std::uint32_t kStateMagic = 0x54533242;
constexpr std::uint32_t kCommandMagic = 0x4D433242;
constexpr std::uint32_t kVersion = 4;
constexpr std::uint16_t kStatePort = 15000;
constexpr std::uint16_t kCommandPort = 15001;
constexpr std::size_t kJoints = 12;
constexpr float kPosStop = 2.146E9F;
constexpr float kVelStop = 16000.0F;
constexpr float kTrainingKp = 160.0F;
constexpr float kTrainingKd = 5.0F;
constexpr float kDampingKd = 3.0F;
constexpr std::uint8_t kB2ServoMode = 0x0A;
constexpr auto kControlPeriod = std::chrono::microseconds(2000);
constexpr auto kRemoteTimeout = std::chrono::milliseconds(250);
constexpr auto kHardwareInterpolation = std::chrono::seconds(3);
constexpr auto kHandoverPoseStableDuration = std::chrono::milliseconds(500);
constexpr float kHandoverMeanQErrorLimit = 0.25F;
constexpr float kHandoverMaxQErrorLimit = 0.55F;
constexpr float kHandoverMaxDqLimit = 0.20F;
constexpr std::uint32_t kCommandEnabled = 1U << 0U;
constexpr std::uint32_t kCommandEmergencyStop = 1U << 1U;
constexpr std::uint32_t kCommandRemoteRequired = 1U << 2U;
constexpr std::uint32_t kKnownCommandFlags =
    kCommandEnabled | kCommandEmergencyStop | kCommandRemoteRequired;
constexpr std::uint32_t kRemoteValid = 1U << 0U;
constexpr std::uint32_t kRemoteEmergencyStopLatched = 1U << 1U;
constexpr std::uint32_t kSafetyFaultLatched = 1U << 0U;
constexpr std::size_t kBlackBoxPreRows = 30U * 50U;
constexpr auto kBlackBoxPostDuration = std::chrono::seconds(30);
constexpr std::uint16_t kRemoteL2 = 1U << 5U;
constexpr std::uint16_t kRemoteB = 1U << 9U;
constexpr std::array<float, kJoints> kPolicyDefaultQ{
    -0.1F, 0.8F, -1.5F, 0.1F, 0.8F, -1.5F,
    -0.1F, 1.0F, -1.5F, 0.1F, 1.0F, -1.5F};
constexpr std::array<float, kJoints> kQMin{
    -0.87F, -0.94F, -2.82F, -0.87F, -0.94F, -2.82F,
    -0.87F, -0.94F, -2.82F, -0.87F, -0.94F, -2.82F};
constexpr std::array<float, kJoints> kQMax{
    0.87F, 4.69F, -0.43F, 0.87F, 4.69F, -0.43F,
    0.87F, 4.69F, -0.43F, 0.87F, 4.69F, -0.43F};
constexpr std::array<float, kJoints> kVelocityLimit{
    23.0F, 23.0F, 14.0F, 23.0F, 23.0F, 14.0F,
    23.0F, 23.0F, 14.0F, 23.0F, 23.0F, 14.0F};
constexpr std::array<const char *, kJoints> kJointNames{
    "FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"};

#pragma pack(push, 1)
struct StatePacket {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t sequence;
  std::uint64_t monotonic_ns;
  float q[kJoints];
  float dq[kJoints];
  float gyro[3];
  float quaternion_wxyz[4];
  float remote_axes[4];
  std::uint32_t remote_keys;
  std::uint32_t remote_flags;
  float tau_est[kJoints];
  std::uint8_t motor_temperature[kJoints];
  std::uint32_t motor_lost[kJoints];
  std::uint8_t battery_soc;
  std::uint8_t battery_status;
  std::uint16_t battery_reserved;
  std::int32_t battery_current_raw;
  std::uint8_t battery_temperature[4];
  std::uint16_t battery_cell_mv[15];
  std::uint32_t safety_flags;
  std::uint32_t gateway_mode;
  std::uint64_t state_samples;
  std::uint64_t state_crc_errors;
  std::uint64_t invalid_states;
  std::uint64_t published_commands;
  std::uint64_t policy_command_sequence;
  float dds_lowstate_hz;
  float publish_jitter_ms;
  float publish_jitter_max_ms;
  float lowstate_age_ms;
  float policy_command_age_ms;
};
struct CommandPacket {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t sequence;
  std::uint64_t monotonic_ns;
  std::uint32_t flags;
  float q_desired[kJoints];
};
#pragma pack(pop)
static_assert(sizeof(StatePacket) == 390);
static_assert(sizeof(CommandPacket) == 76);

std::atomic<bool> running{true};
void stop(int) { running.store(false); }

std::uint64_t monotonic_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch()).count());
}

std::uint32_t gateway_mode_code(const std::string &mode) {
  if (mode == "WAIT_FIRST_POLICY") return 1U;
  if (mode == "INTERPOLATE") return 2U;
  if (mode == "SAFE_HOLD") return 3U;
  if (mode == "POLICY") return 4U;
  if (mode == "SAFE_DAMPING") return 5U;
  if (mode == "SAFE_HOLD_EXIT") return 6U;
  return 0U;
}

std::string csv_quote(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size() + 2U);
  escaped.push_back('"');
  for (char character : value) {
    if (character == '"') escaped.push_back('"');
    escaped.push_back(character);
  }
  escaped.push_back('"');
  return escaped;
}

std::string json_number(float value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream output;
  output << std::setprecision(9) << value;
  return output.str();
}

bool valid(const CommandPacket &packet) {
  if (packet.magic != kCommandMagic || packet.version != kVersion ||
      (packet.flags & kCommandEnabled) == 0U ||
      (packet.flags & ~kKnownCommandFlags) != 0U) return false;
  for (float value : packet.q_desired) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool valid(const unitree_go::msg::dds_::WirelessController_ &remote) {
  const std::array<float, 4> axes{
      remote.lx(), remote.ly(), remote.rx(), remote.ry()};
  return std::all_of(axes.begin(), axes.end(), [](float value) {
    return std::isfinite(value) && std::fabs(value) <= 1.25F;
  });
}

bool emergency_stop_pressed(std::uint16_t keys) {
  return (keys & (kRemoteL2 | kRemoteB)) == (kRemoteL2 | kRemoteB);
}

bool decode_lowstate_remote(
    const unitree_go::msg::dds_::LowState_ &state,
    unitree_go::msg::dds_::WirelessController_ &remote) {
  const auto &raw = state.wireless_remote();
  if (raw[0] != 0x55U || raw[1] != 0x51U) return false;
  std::uint16_t keys = 0;
  float lx = 0.0F;
  float rx = 0.0F;
  float ry = 0.0F;
  float ly = 0.0F;
  std::memcpy(&keys, raw.data() + 2, sizeof(keys));
  std::memcpy(&lx, raw.data() + 4, sizeof(lx));
  std::memcpy(&rx, raw.data() + 8, sizeof(rx));
  std::memcpy(&ry, raw.data() + 12, sizeof(ry));
  std::memcpy(&ly, raw.data() + 20, sizeof(ly));
  remote.lx() = lx;
  remote.ly() = ly;
  remote.rx() = rx;
  remote.ry() = ry;
  remote.keys() = keys;
  return valid(remote);
}

bool valid(const unitree_go::msg::dds_::LowState_ &state) {
  for (std::size_t i = 0; i < kJoints; ++i) {
    if (!std::isfinite(state.motor_state()[i].q()) ||
        !std::isfinite(state.motor_state()[i].dq())) return false;
  }
  double norm_squared = 0.0;
  for (float value : state.imu_state().quaternion()) {
    if (!std::isfinite(value)) return false;
    norm_squared += static_cast<double>(value) * value;
  }
  for (float value : state.imu_state().gyroscope())
    if (!std::isfinite(value)) return false;
  const double norm = std::sqrt(norm_squared);
  return norm >= 0.5 && norm <= 1.5;
}

float tilt_radians(const unitree_go::msg::dds_::LowState_ &state) {
  const auto &quaternion = state.imu_state().quaternion();
  const float w = quaternion[0];
  const float x = quaternion[1];
  const float y = quaternion[2];
  const float z = quaternion[3];
  const float projected_gravity_x = 2.0F * (w * y - x * z);
  const float projected_gravity_y = -2.0F * (y * z + w * x);
  const float horizontal = std::sqrt(projected_gravity_x * projected_gravity_x +
                                     projected_gravity_y * projected_gravity_y);
  return std::asin(std::clamp(horizontal, 0.0F, 1.0F));
}

struct HandoverPoseMetrics {
  float mean_q_error = 0.0F;
  float max_q_error = 0.0F;
  float max_abs_dq = 0.0F;
};

struct FaultDetail {
  std::string reason;
  std::string metric;
  std::string joint;
  float value = std::numeric_limits<float>::quiet_NaN();
  float threshold = std::numeric_limits<float>::quiet_NaN();
  float duration_s = 0.0F;
};

HandoverPoseMetrics handover_pose_metrics(
    const unitree_go::msg::dds_::LowState_ &state) {
  HandoverPoseMetrics metrics{};
  for (std::size_t i = 0; i < kJoints; ++i) {
    const float q_error =
        std::fabs(state.motor_state()[i].q() - kPolicyDefaultQ[i]);
    metrics.mean_q_error += q_error / static_cast<float>(kJoints);
    metrics.max_q_error = std::max(metrics.max_q_error, q_error);
    metrics.max_abs_dq =
        std::max(metrics.max_abs_dq, std::fabs(state.motor_state()[i].dq()));
  }
  return metrics;
}

bool handover_pose_ready(const HandoverPoseMetrics &metrics) {
  return metrics.mean_q_error <= kHandoverMeanQErrorLimit &&
         metrics.max_q_error <= kHandoverMaxQErrorLimit &&
         metrics.max_abs_dq <= kHandoverMaxDqLimit;
}

struct Options {
  bool simulator;
  std::string interface;
  int domain;
  b2::safety::HardwareProfile safety;
  std::chrono::milliseconds initial_interpolation;
  bool limit_target_rate;
  std::string diagnostics_csv;
  std::string hardware_profile;
  std::string hardware_profile_sha256;
  std::string policy_sha256;
};

bool sha256_hex(const std::string &value) {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return std::isxdigit(character) != 0;
         });
}

b2::safety::HardwareProfile simulation_safety_profile() {
  b2::safety::HardwareProfile profile{};
  profile.position_kp.fill(kTrainingKp);
  profile.velocity_kd.fill(kTrainingKd);
  profile.target_slew_rad_s = kVelocityLimit;
  profile.soft_limit_margin_rad.fill(0.0F);
  profile.torque_limit = {200.0F, 200.0F, 300.0F, 200.0F, 200.0F, 300.0F,
                          200.0F, 200.0F, 300.0F, 200.0F, 200.0F, 300.0F};
  profile.current_limit.fill(1.0F);
  profile.joint_velocity_trip_rad_s = kVelocityLimit;
  profile.tracking_error_trip_rad.fill(10.0F);
  profile.damping_kd = kDampingKd;
  profile.state_timeout_s = 0.1F;
  profile.policy_timeout_s = 0.1F;
  profile.command_timeout_s = 0.1F;
  profile.publisher_timeout_s = 0.1F;
  profile.recovery_timeout_s = 0.05F;
  return profile;
}

Options parse_options(int argc, char **argv) {
  if (argc < 2) throw std::invalid_argument("controller mode is required");
  const std::string mode = argv[1];
  Options options{};
  options.safety = simulation_safety_profile();
  int index = 2;
  if (mode == "--sim" || mode == "--sim-no-target-rate-limit" ||
      mode == "--sim-hardware-handover" || mode == "--sim-stack10") {
    options.simulator = true;
    options.interface = "lo";
    options.domain = 1;
    options.limit_target_rate =
        mode != "--sim-no-target-rate-limit" && mode != "--sim-stack10";
    options.initial_interpolation =
        (mode == "--sim-hardware-handover" || mode == "--sim-stack10")
            ? std::chrono::duration_cast<std::chrono::milliseconds>(
                  mode == "--sim-stack10" ? std::chrono::seconds(1)
                                           : kHardwareInterpolation)
            : std::chrono::milliseconds(0);
  } else if (mode == "--hardware") {
    if (argc < 4 || std::string(argv[3]) !=
                        "--acknowledge-b2-low-level-control")
      throw std::invalid_argument("hardware acknowledgement is required");
    options.simulator = false;
    options.interface = argv[2];
    options.domain = 0;
    options.limit_target_rate = true;
    options.initial_interpolation =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            kHardwareInterpolation);
    if (options.interface.empty() || options.interface == "lo")
      throw std::invalid_argument(
          "hardware mode requires a non-loopback interface");
    index = 4;
  } else if (mode == "--hardware-no-target-rate-limit") {
    throw std::invalid_argument(
        "hardware target-rate protection cannot be disabled");
  } else {
    throw std::invalid_argument("unknown controller mode: " + mode);
  }

  while (index < argc) {
    const std::string argument = argv[index++];
    if (index >= argc)
      throw std::invalid_argument("missing value for " + argument);
    const std::string value = argv[index++];
    if (argument == "--diagnostics-csv")
      options.diagnostics_csv = value;
    else if (argument == "--hardware-profile")
      options.hardware_profile = value;
    else if (argument == "--hardware-profile-sha256")
      options.hardware_profile_sha256 = value;
    else if (argument == "--policy-sha256")
      options.policy_sha256 = value;
    else
      throw std::invalid_argument("unknown argument: " + argument);
  }
  if (!options.simulator) {
    if (options.hardware_profile.empty() ||
        !sha256_hex(options.hardware_profile_sha256) ||
        !sha256_hex(options.policy_sha256))
      throw std::invalid_argument(
          "hardware mode requires reviewed profile path, profile SHA-256 and "
          "policy SHA-256");
    options.safety = b2::safety::load_hardware_profile(
        options.hardware_profile, options.policy_sha256);
    for (std::size_t joint = 0; joint < kJoints; ++joint) {
      const float range = kQMax[joint] - kQMin[joint];
      if (options.safety.soft_limit_margin_rad[joint] * 2.0F >= range)
        throw std::invalid_argument(
            "hardware soft-limit margin removes the joint operating range");
    }
  }
  if (options.diagnostics_csv.empty() && !options.simulator)
    throw std::invalid_argument("hardware diagnostics CSV is required");
  return options;
}

void restore_native_motion_if_needed(const Options &options) {
  if (options.simulator) return;
  unitree::robot::b2::MotionSwitcherClient switcher;
  switcher.SetTimeout(10.0F);
  switcher.Init();
  const int selected = switcher.SelectMode("ai");
  if (selected != 0) {
    std::cerr << "MotionSwitcher: failed to restore ai mode: "
              << selected << std::endl;
    return;
  }
  std::cout << "MotionSwitcher: restored ai mode" << std::endl;
}

void release_native_motion_if_needed(const Options &options) {
  if (options.simulator) {
    std::cout << "MotionSwitcher: skipped in simulator domain" << std::endl;
    return;
  }
  unitree::robot::b2::MotionSwitcherClient switcher;
  switcher.SetTimeout(10.0F);
  switcher.Init();
  for (int attempt = 0; attempt < 5; ++attempt) {
    std::string form;
    std::string name;
    const int check = switcher.CheckMode(form, name);
    if (check != 0)
      throw std::runtime_error("MotionSwitcher CheckMode failed: " +
                               std::to_string(check));
    if (name.empty()) {
      std::cout << "MotionSwitcher: native motion service released" << std::endl;
      return;
    }
    std::cout << "MotionSwitcher: releasing form=" << form
              << " mode=" << name << std::endl;
    const int released = switcher.ReleaseMode();
    if (released != 0)
      throw std::runtime_error("MotionSwitcher ReleaseMode failed: " +
                               std::to_string(released));
    std::cout << "MotionSwitcher: ReleaseMode acknowledged; starting LowCmd "
                 "handover without an unpowered verification delay"
              << std::endl;
    return;
  }
  throw std::runtime_error("native motion service remained active");
}

class Gateway {
public:
  explicit Gateway(Options options) : options_(std::move(options)) {
    if (!options_.simulator) {
      lowcmd_lock_fd_ =
          open("/tmp/b2-lowcmd-domain0.lock", O_CREAT | O_RDWR, 0600);
      if (lowcmd_lock_fd_ < 0)
        throw std::runtime_error("cannot open the local LowCmd publisher lock");
      if (flock(lowcmd_lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        close(lowcmd_lock_fd_);
        lowcmd_lock_fd_ = -1;
        throw std::runtime_error(
            "another managed B2 LowCmd publisher owns domain 0");
      }
    }
    command_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    state_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (command_socket_ < 0 || state_socket_ < 0)
      throw std::runtime_error("cannot create loopback UDP sockets");
    const int flags = fcntl(command_socket_, F_GETFL, 0);
    if (flags < 0 || fcntl(command_socket_, F_SETFL, flags | O_NONBLOCK) < 0)
      throw std::runtime_error("cannot make command socket non-blocking");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(kCommandPort);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(command_socket_, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) != 0)
      throw std::runtime_error("cannot bind 127.0.0.1:15001");
    state_destination_.sin_family = AF_INET;
    state_destination_.sin_port = htons(kStatePort);
    state_destination_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  }

  ~Gateway() {
    watchdog_running_.store(false);
    if (watchdog_thread_.joinable()) watchdog_thread_.join();
    if (blackbox_.is_open()) blackbox_.flush();
    if (command_socket_ >= 0) close(command_socket_);
    if (state_socket_ >= 0) close(state_socket_);
    if (lowcmd_lock_fd_ >= 0) {
      flock(lowcmd_lock_fd_, LOCK_UN);
      close(lowcmd_lock_fd_);
    }
  }

  void init() {
    if (!options_.simulator) {
      std::atomic<std::uint64_t> observed_lowcmd{0};
      auto conflict_subscriber = std::make_shared<
          ChannelSubscriber<unitree_go::msg::dds_::LowCmd_>>("rt/lowcmd");
      conflict_subscriber->InitChannel(
          [&observed_lowcmd](const void *) { observed_lowcmd.fetch_add(1); },
          1);
      const auto baseline_started = Clock::now();
      std::this_thread::sleep_for(std::chrono::duration<float>(
          options_.safety.lowcmd_conflict_check_s));
      conflict_subscriber->CloseChannel();
      const double baseline_seconds = std::chrono::duration<double>(
          Clock::now() - baseline_started).count();
      const double baseline_rate = baseline_seconds > 0.0
          ? static_cast<double>(observed_lowcmd.load()) / baseline_seconds
          : 0.0;
      if (baseline_rate < 350.0 || baseline_rate > 650.0)
        throw std::runtime_error(
            "rt/lowcmd baseline is not the expected native 350-650 Hz stream");
      std::cout << "LowCmd native baseline check passed: rate_hz="
                << baseline_rate << " over "
                << options_.safety.lowcmd_conflict_check_s << " s"
                << std::endl;
    }
    publisher_ = std::make_shared<
        ChannelPublisher<unitree_go::msg::dds_::LowCmd_>>("rt/lowcmd");
    publisher_->InitChannel();
    watchdog_running_.store(true);
    watchdog_thread_ = std::thread([this]() {
      while (watchdog_running_.load()) {
        if (publishing_started_.load()) {
          const std::uint64_t last = last_lowcmd_write_ns_.load();
          const std::uint64_t now = monotonic_ns();
          const std::uint64_t timeout_ns = static_cast<std::uint64_t>(
              options_.safety.publisher_timeout_s * 1000000000.0F);
          if (last != 0U && now > last && now - last > timeout_ns) {
            latch_fault("LowCmd publisher heartbeat timeout",
                        "publisher_heartbeat_age", "rt/lowcmd",
                        static_cast<float>(now - last) / 1000000000.0F,
                        options_.safety.publisher_timeout_s,
                        options_.safety.publisher_timeout_s);
            std::lock_guard<std::mutex> lock(command_mutex_);
            set_damping();
            mode_ = "SAFE_DAMPING";
            write();
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
    subscriber_ = std::make_shared<
        ChannelSubscriber<unitree_go::msg::dds_::LowState_>>("rt/lowstate");
    subscriber_->InitChannel(
        [this](const void *message) {
          const auto state =
              *static_cast<const unitree_go::msg::dds_::LowState_ *>(message);
          const auto expected = crc32_core(
              const_cast<std::uint32_t *>(
                  reinterpret_cast<const std::uint32_t *>(&state)),
              (sizeof(state) >> 2U) - 1U);
          if (state.crc() != expected || !valid(state)) {
            if (state.crc() != expected) {
              ++state_crc_errors_;
              latch_fault("LowState CRC mismatch", "crc_error_count",
                          "rt/lowstate",
                          static_cast<float>(state_crc_errors_.load()), 0.0F,
                          0.0F);
            } else {
              ++invalid_states_;
              latch_fault("LowState contains invalid numeric data",
                          "invalid_state_count", "rt/lowstate",
                          static_cast<float>(invalid_states_.load()), 0.0F,
                          0.0F);
            }
            return;
          }
          unitree_go::msg::dds_::WirelessController_ raw_remote{};
          if (decode_lowstate_remote(state, raw_remote)) {
            if (emergency_stop_pressed(raw_remote.keys()))
              latch_emergency_stop("LowState wireless_remote L2+B");
            std::lock_guard<std::mutex> remote_lock(remote_mutex_);
            remote_ = raw_remote;
            remote_received_at_ = Clock::now();
            have_remote_ = true;
            ++remote_samples_;
          }
          std::lock_guard<std::mutex> lock(state_mutex_);
          const auto received_now = Clock::now();
          if (last_state_rate_sample_.time_since_epoch().count() != 0) {
            const float interval_s =
                std::chrono::duration<float>(received_now - last_state_rate_sample_)
                    .count();
            if (interval_s > 0.0F) {
              const float instantaneous_hz = 1.0F / interval_s;
              state_rate_hz_ = state_rate_hz_ == 0.0F
                                   ? instantaneous_hz
                                   : 0.9F * state_rate_hz_ +
                                         0.1F * instantaneous_hz;
            }
          }
          last_state_rate_sample_ = received_now;
          low_state_ = state;
          state_received_at_ = received_now;
          have_state_ = true;
          ++state_samples_;
        }, 1);
    remote_subscriber_ = std::make_shared<
        ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>>(
        "rt/wirelesscontroller");
    remote_subscriber_->InitChannel(
        [this](const void *message) {
          const auto remote = *static_cast<
              const unitree_go::msg::dds_::WirelessController_ *>(message);
          if (!valid(remote)) {
            ++invalid_remote_samples_;
            return;
          }
          if (emergency_stop_pressed(remote.keys()))
            latch_emergency_stop("rt/wirelesscontroller L2+B");
          std::lock_guard<std::mutex> lock(remote_mutex_);
          remote_ = remote;
          remote_received_at_ = Clock::now();
          have_remote_ = true;
          ++remote_samples_;
        }, 1);
    low_command_.head() = {0xFE, 0xEF};
    low_command_.level_flag() = 0xFF;
    low_command_.gpio() = 0;
    for (auto &motor : low_command_.motor_cmd()) {
      motor.mode() = kB2ServoMode;
      motor.q() = kPosStop;
      motor.dq() = kVelStop;
      motor.kp() = 0.0F;
      motor.kd() = 0.0F;
      motor.tau() = 0.0F;
    }
    if (!options_.diagnostics_csv.empty()) {
      diagnostics_.open(options_.diagnostics_csv, std::ios::out | std::ios::trunc);
      if (!diagnostics_)
        throw std::runtime_error("cannot open diagnostics CSV: " +
                                 options_.diagnostics_csv);
      std::ostringstream header;
      header << "elapsed_s,mode,state_samples,policy_monotonic_ns,lowcmd";
      header << ",remote_fresh,remote_keys,remote_lx,remote_ly,remote_rx,"
                "remote_ry,emergency_stop_latched,fault_reason,fault_metric,"
                "fault_joint,fault_value,fault_threshold,fault_duration_s";
      header << ",dds_lowstate_hz,state_crc_errors,invalid_states,"
                "publish_jitter_ms,publish_jitter_max_ms";
      header << ",battery_soc,battery_current_raw,battery_max_temp,"
                "battery_min_cell_mv,battery_max_cell_mv";
      for (const char *name : kJointNames) header << ",target_" << name;
      for (const char *name : kJointNames) header << ",actual_" << name;
      for (const char *name : kJointNames) header << ",dq_" << name;
      for (const char *name : kJointNames) header << ",tau_" << name;
      for (const char *name : kJointNames) header << ",temp_" << name;
      for (const char *name : kJointNames) header << ",lost_" << name;
      for (const char *name : kJointNames) header << ",lowcmd_q_" << name;
      for (const char *name : kJointNames) header << ",lowcmd_dq_" << name;
      for (const char *name : kJointNames) header << ",lowcmd_kp_" << name;
      for (const char *name : kJointNames) header << ",lowcmd_kd_" << name;
      for (const char *name : kJointNames) header << ",lowcmd_tau_" << name;
      header << ",lowcmd_crc";
      diagnostics_header_ = header.str();
      diagnostics_ << diagnostics_header_ << "\n";
    }
  }

  void wait_for_policy_ready(const char *phase, bool reset_command) {
    if (reset_command) have_command_ = false;
    mode_ = "WAIT_FIRST_POLICY";
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    std::uint64_t tick = 0;
    bool simulator_hold_captured = false;
    std::array<float, kJoints> simulator_hold_target{};
    const bool require_simulator_rest =
        options_.simulator && options_.initial_interpolation.count() > 0;
    bool simulator_rest_timing = false;
    Clock::time_point simulator_rest_ready_since{};
    Clock::time_point last_simulator_rest_report{};
    const bool require_handover_pose =
        !options_.simulator && std::string(phase) == "before handover";
    bool handover_pose_timing = false;
    Clock::time_point handover_pose_ready_since{};
    Clock::time_point last_handover_pose_report{};
    while (running.load() && Clock::now() < deadline) {
      receive_command();
      if ((tick % 10U) == 0U &&
          (!require_simulator_rest || simulator_hold_captured))
        send_state();
      ++tick;

      Clock::time_point state_received_at{};
      unitree_go::msg::dds_::LowState_ state{};
      bool have_state = false;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        have_state = have_state_;
        state_received_at = state_received_at_;
        if (have_state) state = low_state_;
      }
      const auto now = Clock::now();
      const bool state_fresh =
          have_state && now - state_received_at <= state_timeout();
      const bool command_fresh =
          have_command_ && now - command_received_at_ <= command_timeout();
      const bool remote_ready =
          !remote_control_required_ || remote_fresh(now);
      if (fault_latched_.load())
        throw std::runtime_error(
            std::string("LowState fault while waiting for policy: ") + phase);

      if (options_.simulator && state_fresh) {
        if (!simulator_hold_captured) {
          if (require_simulator_rest) {
            const HandoverPoseMetrics metrics = handover_pose_metrics(state);
            if (metrics.max_abs_dq > kHandoverMaxDqLimit) {
              simulator_rest_timing = false;
              if (last_simulator_rest_report.time_since_epoch().count() == 0 ||
                  now - last_simulator_rest_report >= std::chrono::seconds(1)) {
                std::cout << "Waiting for stationary simulator LowState: "
                          << "max_abs_dq=" << metrics.max_abs_dq << std::endl;
                last_simulator_rest_report = now;
              }
              std::this_thread::sleep_for(kControlPeriod);
              continue;
            }
            if (!simulator_rest_timing) {
              simulator_rest_timing = true;
              simulator_rest_ready_since = now;
            }
            if (now - simulator_rest_ready_since <
                kHandoverPoseStableDuration) {
              std::this_thread::sleep_for(kControlPeriod);
              continue;
            }
            std::cout << "Simulator LowState stationary for "
                      << (kHandoverPoseStableDuration.count() / 1000.0)
                      << " s before hardware-style handover: max_abs_dq="
                      << metrics.max_abs_dq << std::endl;
          }
          for (std::size_t i = 0; i < kJoints; ++i)
            simulator_hold_target[i] = std::clamp(
                state.motor_state()[i].q(), kQMin[i], kQMax[i]);
          simulator_hold_captured = true;
          std::cout << "Simulator pre-hold captured from stationary LowState; "
                       "waiting for first zero-command policy target"
                    << std::endl;
        }
        {
          std::lock_guard<std::mutex> command_lock(command_mutex_);
          set_position_target(simulator_hold_target);
          write();
        }
        mode_ = "SIM_PREHOLD";
      }

      if (state_fresh && command_fresh && remote_ready) {
        if (options_.simulator && !simulator_hold_captured) {
          std::this_thread::sleep_for(kControlPeriod);
          continue;
        }
        if (require_handover_pose) {
          const HandoverPoseMetrics metrics = handover_pose_metrics(state);
          if (!handover_pose_ready(metrics)) {
            handover_pose_timing = false;
            if (last_handover_pose_report.time_since_epoch().count() == 0 ||
                now - last_handover_pose_report >= std::chrono::seconds(1)) {
              std::cout
                  << "Waiting for policy-ready standing pose: mean_q_error="
                  << metrics.mean_q_error
                  << " max_q_error=" << metrics.max_q_error
                  << " max_abs_dq=" << metrics.max_abs_dq << std::endl;
              last_handover_pose_report = now;
            }
            std::this_thread::sleep_for(kControlPeriod);
            continue;
          }
          if (!handover_pose_timing) {
            handover_pose_timing = true;
            handover_pose_ready_since = now;
          }
          if (now - handover_pose_ready_since < kHandoverPoseStableDuration) {
            std::this_thread::sleep_for(kControlPeriod);
            continue;
          }
          std::cout << "Policy-ready standing pose stable for "
                    << (kHandoverPoseStableDuration.count() / 1000.0)
                    << " s: mean_q_error=" << metrics.mean_q_error
                    << " max_q_error=" << metrics.max_q_error
                    << " max_abs_dq=" << metrics.max_abs_dq << std::endl;
        }
        std::cout << "Policy target ready (" << phase << ")"
                  << (options_.simulator
                          ? "; simulator pre-hold complete"
                          : "; no active LowCmd published yet")
                  << std::endl;
        return;
      }
      if (state_fresh && command_fresh && !remote_ready &&
          (last_remote_wait_report_.time_since_epoch().count() == 0 ||
           now - last_remote_wait_report_ >= std::chrono::seconds(1))) {
        std::cout << "Waiting for fresh rt/wirelesscontroller before handover"
                  << std::endl;
        last_remote_wait_report_ = now;
      }
      std::this_thread::sleep_for(kControlPeriod);
    }
    throw std::runtime_error(
        std::string("timed out waiting for fresh LowState/policy: ") + phase);
  }

  void rearm_policy_watchdog_after_native_release() {
    if (options_.simulator)
      throw std::logic_error(
          "policy watchdog re-arm is only valid for hardware handover");
    std::lock_guard<std::mutex> command_lock(command_mutex_);
    if (!have_command_)
      throw std::runtime_error(
          "cannot re-arm policy watchdog without a validated policy target");
    const auto now = Clock::now();
    const float excluded_age_ms =
        std::chrono::duration<float, std::milli>(now - command_received_at_)
            .count();
    command_received_at_ = now;
    std::cout
        << "Policy watchdog re-armed after MotionSwitcher release: excluded "
        << excluded_age_ms
        << " ms of blocking handover time; retaining validated policy sequence="
        << command_.sequence << std::endl;
  }

  void run() {
    std::cout << "b2_policy_controller: DDS domain=" << options_.domain
              << " interface=" << options_.interface
              << (options_.simulator ? " (SIMULATOR)" : " (HARDWARE)") << "\n"
              << "rt/lowstate -> policy 50 Hz -> rt/lowcmd 500 Hz\n"
              << "active gains Kp[0]=" << options_.safety.position_kp[0]
              << " Kd[0]=" << options_.safety.velocity_kd[0]
              << ", interpolation tracks latest policy target for "
              << (options_.initial_interpolation.count() / 1000.0)
              << " s" << std::endl;
    diagnostics_started_at_ = Clock::now();
    if (diagnostics_.is_open())
      std::cout << "50 Hz controller diagnostics CSV: "
                << options_.diagnostics_csv << std::endl;
    auto next = Clock::now();
    last_publish_cycle_ = next;
    std::uint64_t tick = 0;
    while (running.load()) {
      next += kControlPeriod;
      const auto cycle_now = Clock::now();
      const auto expected_cycle = next - kControlPeriod;
      publish_jitter_ms_ = std::fabs(
          std::chrono::duration<float, std::milli>(cycle_now - expected_cycle)
              .count());
      publish_jitter_max_ms_ =
          std::max(publish_jitter_max_ms_, publish_jitter_ms_);
      if (cycle_now - last_publish_cycle_ > publisher_timeout())
        latch_fault(
            "LowCmd publisher scheduling timeout", "publisher_cycle_age",
            "rt/lowcmd",
            std::chrono::duration<float>(cycle_now - last_publish_cycle_)
                .count(),
            options_.safety.publisher_timeout_s,
            options_.safety.publisher_timeout_s);
      receive_command();
      if ((tick % 10U) == 0U) send_state();
      publish();
      last_publish_cycle_ = Clock::now();
      if ((tick % 10U) == 0U) write_diagnostics();
      if ((tick % 500U) == 0U) report();
      ++tick;
      std::this_thread::sleep_until(next);
    }
    safe_stop();
  }

private:
  void receive_command() {
    CommandPacket candidate{};
    for (;;) {
      const ssize_t count = recv(command_socket_, &candidate, sizeof(candidate), 0);
      if (count < 0) break;
      const auto now_ns = monotonic_ns();
      const bool fresh_timestamp =
          candidate.monotonic_ns <= now_ns &&
          now_ns - candidate.monotonic_ns <= static_cast<std::uint64_t>(
              options_.safety.command_timeout_s * 1000000000.0F);
      if (count == static_cast<ssize_t>(sizeof(candidate)) && valid(candidate) &&
          fresh_timestamp && candidate.monotonic_ns > command_monotonic_ns_) {
        if ((candidate.flags & kCommandEmergencyStop) != 0U)
          latch_emergency_stop("policy process request");
        command_ = candidate;
        command_monotonic_ns_ = candidate.monotonic_ns;
        command_received_at_ = Clock::now();
        have_command_ = true;
        remote_control_required_ =
            (candidate.flags & kCommandRemoteRequired) != 0U;
      }
    }
  }

  void send_state() {
    unitree_go::msg::dds_::LowState_ snapshot{};
    Clock::time_point snapshot_received_at{};
    float snapshot_rate_hz = 0.0F;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!have_state_) return;
      snapshot = low_state_;
      snapshot_received_at = state_received_at_;
      snapshot_rate_hz = state_rate_hz_;
    }
    StatePacket packet{};
    packet.magic = kStateMagic;
    packet.version = kVersion;
    packet.sequence = ++state_sequence_;
    packet.monotonic_ns = monotonic_ns();
    for (std::size_t i = 0; i < kJoints; ++i) {
      packet.q[i] = snapshot.motor_state()[i].q();
      packet.dq[i] = snapshot.motor_state()[i].dq();
      packet.tau_est[i] = snapshot.motor_state()[i].tau_est();
      packet.motor_temperature[i] = snapshot.motor_state()[i].temperature();
      packet.motor_lost[i] = snapshot.motor_state()[i].lost();
    }
    for (std::size_t i = 0; i < 3; ++i)
      packet.gyro[i] = snapshot.imu_state().gyroscope()[i];
    for (std::size_t i = 0; i < 4; ++i)
      packet.quaternion_wxyz[i] = snapshot.imu_state().quaternion()[i];
    {
      std::lock_guard<std::mutex> lock(remote_mutex_);
      packet.remote_axes[0] = remote_.lx();
      packet.remote_axes[1] = remote_.ly();
      packet.remote_axes[2] = remote_.rx();
      packet.remote_axes[3] = remote_.ry();
      packet.remote_keys = remote_.keys();
      if (have_remote_ && Clock::now() - remote_received_at_ <= kRemoteTimeout)
        packet.remote_flags |= kRemoteValid;
    }
    if (emergency_stop_latched_.load())
      packet.remote_flags |= kRemoteEmergencyStopLatched;
    const auto &bms = snapshot.bms_state();
    packet.battery_soc = bms.soc();
    packet.battery_status = bms.status();
    packet.battery_current_raw = bms.current();
    packet.battery_temperature[0] = bms.bq_ntc()[0];
    packet.battery_temperature[1] = bms.bq_ntc()[1];
    packet.battery_temperature[2] = bms.mcu_ntc()[0];
    packet.battery_temperature[3] = bms.mcu_ntc()[1];
    std::copy(bms.cell_vol().begin(), bms.cell_vol().end(),
              packet.battery_cell_mv);
    if (fault_latched_.load()) packet.safety_flags |= kSafetyFaultLatched;
    packet.gateway_mode = gateway_mode_code(mode_);
    packet.state_samples = state_samples_.load();
    packet.state_crc_errors = state_crc_errors_.load();
    packet.invalid_states = invalid_states_.load();
    packet.published_commands = published_commands_.load();
    packet.dds_lowstate_hz = snapshot_rate_hz;
    packet.publish_jitter_ms = publish_jitter_ms_;
    packet.publish_jitter_max_ms = publish_jitter_max_ms_;
    packet.lowstate_age_ms = std::chrono::duration<float, std::milli>(
                                 Clock::now() - snapshot_received_at)
                                 .count();
    {
      std::lock_guard<std::mutex> command_lock(command_mutex_);
      packet.policy_command_sequence = command_.sequence;
      packet.policy_command_age_ms = have_command_
                                         ? std::chrono::duration<float, std::milli>(
                                               Clock::now() - command_received_at_)
                                               .count()
                                         : -1.0F;
    }
    sendto(state_socket_, &packet, sizeof(packet), 0,
           reinterpret_cast<sockaddr *>(&state_destination_),
           sizeof(state_destination_));
  }

  void set_damping() {
    for (std::size_t i = 0; i < kJoints; ++i) {
      auto &motor = low_command_.motor_cmd()[i];
      motor.mode() = kB2ServoMode;
      motor.q() = kPosStop;
      motor.dq() = 0.0F;
      motor.kp() = 0.0F;
      motor.kd() = options_.safety.damping_kd;
      motor.tau() = 0.0F;
    }
  }

  void set_position_target(const std::array<float, kJoints> &target) {
    for (std::size_t i = 0; i < kJoints; ++i) {
      auto &motor = low_command_.motor_cmd()[i];
      motor.mode() = kB2ServoMode;
      motor.q() = target[i];
      motor.dq() = 0.0F;
      const float gain_scale = active_gain_scale(Clock::now());
      motor.kp() = options_.safety.position_kp[i] * gain_scale;
      motor.kd() = options_.safety.velocity_kd[i] * gain_scale;
      motor.tau() = 0.0F;
      last_target_[i] = target[i];
    }
  }

  void begin_interpolation(
      const unitree_go::msg::dds_::LowState_ &state, Clock::time_point now) {
    if (!have_command_)
      throw std::runtime_error("cannot interpolate without a policy target");
    interpolation_started_ = true;
    interpolation_started_at_ = now;
    for (std::size_t i = 0; i < kJoints; ++i) {
      interpolation_start_[i] = state.motor_state()[i].q();
      last_target_[i] = interpolation_start_[i];
    }
    if (options_.initial_interpolation.count() > 0)
      std::cout << "Handover LowState captured; interpolating measured pose to "
                   "the latest learned-hold policy target for "
                << (options_.initial_interpolation.count() / 1000.0)
                << " s"
                << std::endl;
    else
      std::cout
          << "Simulator handover LowState captured; applying the current policy "
             "target "
          << (options_.limit_target_rate
                  ? "with joint-rate limits"
                  : "without joint-rate limits (hard q bounds retained)")
          << std::endl;
  }

  float active_gain_scale(Clock::time_point now) const {
    if (options_.simulator || options_.safety.gain_ramp_s <= 0.0F ||
        !interpolation_started_)
      return 1.0F;
    const float elapsed =
        std::chrono::duration<float>(now - interpolation_started_at_).count();
    return std::clamp(elapsed / options_.safety.gain_ramp_s, 0.0F, 1.0F);
  }

  float lower_target_bound(std::size_t joint) const {
    return kQMin[joint] + options_.safety.soft_limit_margin_rad[joint];
  }

  float upper_target_bound(std::size_t joint) const {
    return kQMax[joint] - options_.safety.soft_limit_margin_rad[joint];
  }

  std::array<float, kJoints> interpolation_target(Clock::time_point now) const {
    const float elapsed =
        std::chrono::duration<float>(now - interpolation_started_at_).count();
    const float interpolation_seconds =
        std::chrono::duration<float>(options_.initial_interpolation).count();
    const float alpha = interpolation_seconds > 0.0F
                            ? std::clamp(elapsed / interpolation_seconds,
                                         0.0F, 1.0F)
                            : 1.0F;
    std::array<float, kJoints> target{};
    constexpr float dt = 0.002F;
    for (std::size_t i = 0; i < kJoints; ++i) {
      const float latest_goal =
          std::clamp(command_.q_desired[i], lower_target_bound(i),
                     upper_target_bound(i));
      const float desired = interpolation_start_[i] +
                            alpha * (latest_goal - interpolation_start_[i]);
      const float maximum_step = options_.safety.target_slew_rad_s[i] * dt;
      target[i] = last_target_[i] +
                  std::clamp(desired - last_target_[i], -maximum_step,
                             maximum_step);
    }
    return target;
  }

  std::array<float, kJoints> limited_policy_target() const {
    std::array<float, kJoints> target{};
    constexpr float dt = 0.002F;
    for (std::size_t i = 0; i < kJoints; ++i) {
      const float bounded =
          std::clamp(command_.q_desired[i], lower_target_bound(i),
                     upper_target_bound(i));
      if (!options_.limit_target_rate) {
        target[i] = bounded;
        continue;
      }
      const float maximum_step = options_.safety.target_slew_rad_s[i] * dt;
      target[i] = last_target_[i] +
                  std::clamp(bounded - last_target_[i], -maximum_step,
                             maximum_step);
    }
    return target;
  }

  void write() {
    low_command_.crc() = crc32_core(
        reinterpret_cast<std::uint32_t *>(&low_command_),
        (sizeof(low_command_) >> 2U) - 1U);
    if (!publisher_->Write(low_command_))
      latch_fault("DDS rt/lowcmd write failed", "dds_write", "rt/lowcmd",
                  1.0F, 0.0F, 0.0F);
    else
      last_lowcmd_write_ns_.store(monotonic_ns());
    publishing_started_.store(true);
    ++published_commands_;
  }

  void publish() {
    unitree_go::msg::dds_::LowState_ state{};
    Clock::time_point state_received_at{};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!have_state_) return;
      state = low_state_;
      state_received_at = state_received_at_;
    }
    const auto now = Clock::now();
    std::lock_guard<std::mutex> command_lock(command_mutex_);
    if (!have_command_)
      throw std::runtime_error("control started without a policy target");
    if (!interpolation_started_) begin_interpolation(state, now);
    evaluate_hardware_safety(state, now);
    if (now - state_received_at > state_timeout())
      latch_fault("rt/lowstate timeout", "state_age", "rt/lowstate",
                  std::chrono::duration<float>(now - state_received_at).count(),
                  options_.safety.state_timeout_s,
                  options_.safety.state_timeout_s);
    if (remote_control_required_ && !remote_fresh(now))
      latch_emergency_stop("rt/wirelesscontroller timeout");

    if (fault_latched_.load()) {
      set_damping();
      mode_ = "SAFE_DAMPING";
    } else if (now - command_received_at_ > policy_timeout()) {
      latch_fault(
          "policy command timeout", "policy_command_age", "policy",
          std::chrono::duration<float>(now - command_received_at_).count(),
          options_.safety.policy_timeout_s,
          options_.safety.policy_timeout_s);
      set_damping();
      mode_ = "SAFE_DAMPING";
    } else if (now - command_received_at_ > command_timeout()) {
      set_position_target(last_target_);
      mode_ = "SAFE_HOLD";
    } else if (now - interpolation_started_at_ <
                   options_.initial_interpolation) {
      set_position_target(interpolation_target(now));
      mode_ = "INTERPOLATE";
    } else {
      set_position_target(limited_policy_target());
      policy_ever_active_ = true;
      mode_ = "POLICY";
    }
    write();
  }

  void safe_stop() {
    if (!interpolation_started_) return;
    std::lock_guard<std::mutex> command_lock(command_mutex_);
    if (fault_latched_.load()) {
      set_damping();
      mode_ = "SAFE_DAMPING";
    } else {
      set_position_target(last_target_);
      mode_ = "SAFE_HOLD_EXIT";
    }
    const int frames = std::max(
        25, static_cast<int>(std::ceil(options_.safety.recovery_timeout_s /
                                       0.002F)));
    for (int i = 0; i < frames; ++i) {
      write();
      std::this_thread::sleep_for(kControlPeriod);
    }
  }

  void write_diagnostics() {
    if (!diagnostics_.is_open()) return;
    unitree_go::msg::dds_::LowState_ state{};
    float dds_rate_hz = 0.0F;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!have_state_) return;
      state = low_state_;
      dds_rate_hz = state_rate_hz_;
    }
    const double elapsed_s =
        std::chrono::duration<double>(Clock::now() - diagnostics_started_at_)
            .count();
    unitree_go::msg::dds_::WirelessController_ remote{};
    {
      std::lock_guard<std::mutex> lock(remote_mutex_);
      remote = remote_;
    }
    const FaultDetail fault = fault_detail();
    std::ostringstream row;
    row << std::fixed << std::setprecision(9) << elapsed_s << ","
        << mode_ << "," << state_samples_.load() << ","
        << command_monotonic_ns_ << "," << published_commands_.load()
        << "," << remote_fresh(Clock::now())
        << "," << remote.keys()
        << "," << remote.lx() << "," << remote.ly()
        << "," << remote.rx() << "," << remote.ry()
        << "," << emergency_stop_latched_.load()
        << "," << csv_quote(fault.reason)
        << "," << csv_quote(fault.metric)
        << "," << csv_quote(fault.joint)
        << "," << fault.value << "," << fault.threshold << ","
        << fault.duration_s << "," << dds_rate_hz << ","
        << state_crc_errors_.load() << "," << invalid_states_.load() << ","
        << publish_jitter_ms_ << "," << publish_jitter_max_ms_;
    const auto &bms = state.bms_state();
    unsigned int maximum_bms_temperature = 0;
    for (std::uint8_t value : bms.bq_ntc())
      maximum_bms_temperature = std::max(
          maximum_bms_temperature, static_cast<unsigned int>(value));
    for (std::uint8_t value : bms.mcu_ntc())
      maximum_bms_temperature = std::max(
          maximum_bms_temperature, static_cast<unsigned int>(value));
    std::uint16_t minimum_cell_mv = 0;
    std::uint16_t maximum_cell_mv = 0;
    for (std::uint16_t value : bms.cell_vol()) {
      if (value == 0U) continue;
      minimum_cell_mv = minimum_cell_mv == 0U
                            ? value
                            : std::min(minimum_cell_mv, value);
      maximum_cell_mv = std::max(maximum_cell_mv, value);
    }
    row << "," << static_cast<unsigned int>(bms.soc()) << ","
        << bms.current() << "," << maximum_bms_temperature << ","
        << minimum_cell_mv << "," << maximum_cell_mv;
    for (float target : last_target_) row << "," << target;
    for (std::size_t i = 0; i < kJoints; ++i)
      row << "," << state.motor_state()[i].q();
    for (std::size_t i = 0; i < kJoints; ++i)
      row << "," << state.motor_state()[i].dq();
    for (std::size_t i = 0; i < kJoints; ++i)
      row << "," << state.motor_state()[i].tau_est();
    for (std::size_t i = 0; i < kJoints; ++i)
      row << "," << static_cast<unsigned int>(
                         state.motor_state()[i].temperature());
    for (std::size_t i = 0; i < kJoints; ++i)
      row << "," << static_cast<unsigned long long>(
                         state.motor_state()[i].lost());
    for (const auto &motor : low_command_.motor_cmd()) row << "," << motor.q();
    for (const auto &motor : low_command_.motor_cmd()) row << "," << motor.dq();
    for (const auto &motor : low_command_.motor_cmd()) row << "," << motor.kp();
    for (const auto &motor : low_command_.motor_cmd()) row << "," << motor.kd();
    for (const auto &motor : low_command_.motor_cmd()) row << "," << motor.tau();
    row << "," << low_command_.crc();
    const std::string serialized = row.str();
    diagnostics_ << serialized << "\n";
    update_blackbox(serialized);
    if ((++diagnostics_rows_ % 10U) == 0U || fault_latched_.load())
      diagnostics_.flush();
  }

  void report() {
    unitree_go::msg::dds_::LowState_ state{};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!have_state_) return;
      state = low_state_;
    }

    std::array<float, 4> tracking_error{};
    std::array<float, 4> absolute_tau{};
    unsigned int maximum_temperature = 0;
    std::uint64_t lost_samples = 0;
    for (std::size_t i = 0; i < kJoints; ++i) {
      const std::size_t leg = i / 3U;
      tracking_error[leg] =
          std::max(tracking_error[leg],
                   std::fabs(last_target_[i] - state.motor_state()[i].q()));
      const float tau = state.motor_state()[i].tau_est();
      if (std::isfinite(tau))
        absolute_tau[leg] = std::max(absolute_tau[leg], std::fabs(tau));
      maximum_temperature =
          std::max(maximum_temperature,
                   static_cast<unsigned int>(
                       state.motor_state()[i].temperature()));
      lost_samples += state.motor_state()[i].lost();
    }

    std::cout << "controller mode=" << mode_
              << " state=" << state_samples_.load()
              << " state_crc_errors=" << state_crc_errors_.load()
              << " invalid_state=" << invalid_states_.load()
              << " remote=" << remote_samples_.load()
              << " invalid_remote=" << invalid_remote_samples_.load()
              << " remote_fresh=" << remote_fresh(Clock::now())
              << " estop=" << emergency_stop_latched_.load()
              << " fault_reason=\"" << fault_reason() << "\""
              << " battery_soc="
              << static_cast<unsigned int>(state.bms_state().soc())
              << " battery_current_raw=" << state.bms_state().current()
              << " policy_sequence=" << command_monotonic_ns_
              << " lowcmd=" << published_commands_.load()
              << " tracking_error_rad_FR_FL_RR_RL=["
              << tracking_error[0] << "," << tracking_error[1] << ","
              << tracking_error[2] << "," << tracking_error[3] << "]"
              << " abs_tau_est_FR_FL_RR_RL=["
              << absolute_tau[0] << "," << absolute_tau[1] << ","
              << absolute_tau[2] << "," << absolute_tau[3] << "]"
              << " max_motor_temp=" << maximum_temperature
              << " motor_lost_sum=" << lost_samples << std::endl;
    std::cout << "joint_diag";
    for (std::size_t i = 0; i < kJoints; ++i) {
      const auto &motor = state.motor_state()[i];
      std::cout << " " << kJointNames[i]
                << "={target:" << last_target_[i]
                << ",actual:" << motor.q()
                << ",error:" << last_target_[i] - motor.q()
                << ",dq:" << motor.dq()
                << ",tau:" << motor.tau_est()
                << ",temp:"
                << static_cast<unsigned int>(motor.temperature())
                << ",lost:" << motor.lost() << "}";
    }
    std::cout << std::endl;
  }

  Options options_;
  std::ofstream diagnostics_;
  std::ofstream blackbox_;
  std::string diagnostics_header_;
  std::deque<std::string> blackbox_pre_rows_;
  bool blackbox_complete_ = false;
  Clock::time_point diagnostics_started_at_{};
  Clock::time_point last_publish_cycle_{};
  std::uint64_t diagnostics_rows_ = 0;
  int command_socket_ = -1;
  int state_socket_ = -1;
  int lowcmd_lock_fd_ = -1;
  sockaddr_in state_destination_{};
  ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> publisher_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> subscriber_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_>
      remote_subscriber_;
  unitree_go::msg::dds_::LowCmd_ low_command_{};
  unitree_go::msg::dds_::LowState_ low_state_{};
  std::mutex state_mutex_;
  bool have_state_ = false;
  Clock::time_point state_received_at_{};
  Clock::time_point last_state_rate_sample_{};
  float state_rate_hz_ = 0.0F;
  unitree_go::msg::dds_::WirelessController_ remote_{};
  mutable std::mutex remote_mutex_;
  bool have_remote_ = false;
  Clock::time_point remote_received_at_{};
  Clock::time_point last_remote_wait_report_{};
  bool have_command_ = false;
  bool remote_control_required_ = false;
  bool policy_ever_active_ = false;
  CommandPacket command_{};
  std::uint64_t state_sequence_ = 0;
  std::uint64_t command_monotonic_ns_ = 0;
  Clock::time_point command_received_at_{};
  bool interpolation_started_ = false;
  Clock::time_point interpolation_started_at_{};
  std::array<float, kJoints> interpolation_start_{};
  std::array<float, kJoints> last_target_{};
  std::array<Clock::time_point, kJoints> dq_hazard_since_{};
  std::array<Clock::time_point, kJoints> torque_hazard_since_{};
  std::array<Clock::time_point, kJoints> temperature_hazard_since_{};
  std::array<Clock::time_point, kJoints> tracking_hazard_since_{};
  std::array<std::uint32_t, kJoints> previous_lost_{};
  bool have_lost_baseline_ = false;
  Clock::time_point tilt_hazard_since_{};
  Clock::time_point angular_rate_hazard_since_{};
  Clock::time_point battery_soc_hazard_since_{};
  Clock::time_point battery_current_hazard_since_{};
  Clock::time_point battery_temperature_hazard_since_{};
  std::atomic<bool> fault_latched_{false};
  std::atomic<bool> emergency_stop_latched_{false};
  std::mutex command_mutex_;
  std::atomic<bool> watchdog_running_{false};
  std::atomic<bool> publishing_started_{false};
  std::atomic<std::uint64_t> last_lowcmd_write_ns_{0};
  std::thread watchdog_thread_;
  mutable std::mutex fault_mutex_;
  std::string fault_reason_;
  FaultDetail fault_detail_{};
  std::atomic<std::uint64_t> fault_latched_monotonic_ns_{0};
  std::atomic<bool> fault_record_written_{false};
  std::atomic<std::uint64_t> state_samples_{0};
  std::atomic<std::uint64_t> state_crc_errors_{0};
  std::atomic<std::uint64_t> invalid_states_{0};
  std::atomic<std::uint64_t> remote_samples_{0};
  std::atomic<std::uint64_t> invalid_remote_samples_{0};
  std::atomic<std::uint64_t> published_commands_{0};
  float publish_jitter_ms_ = 0.0F;
  float publish_jitter_max_ms_ = 0.0F;
  std::string mode_{"WAITING_STATE"};

  bool remote_fresh(Clock::time_point now) const {
    std::lock_guard<std::mutex> lock(remote_mutex_);
    return have_remote_ && now - remote_received_at_ <= kRemoteTimeout;
  }

  std::chrono::duration<float> state_timeout() const {
    return std::chrono::duration<float>(options_.safety.state_timeout_s);
  }

  std::chrono::duration<float> command_timeout() const {
    return std::chrono::duration<float>(options_.safety.command_timeout_s);
  }

  std::chrono::duration<float> policy_timeout() const {
    return std::chrono::duration<float>(options_.safety.policy_timeout_s);
  }

  std::chrono::duration<float> publisher_timeout() const {
    return std::chrono::duration<float>(options_.safety.publisher_timeout_s);
  }

  static bool persisted(bool condition, Clock::time_point now,
                        Clock::time_point &started,
                        std::chrono::duration<float> duration) {
    if (!condition) {
      started = Clock::time_point{};
      return false;
    }
    if (started.time_since_epoch().count() == 0) started = now;
    return now - started >= duration;
  }

  void evaluate_hardware_safety(
      const unitree_go::msg::dds_::LowState_ &state, Clock::time_point now) {
    if (options_.simulator) {
      const float tilt = tilt_radians(state);
      const float limit = std::asin(0.6F);
      if (tilt > limit)
        latch_fault("simulator projected-gravity tilt exceeds 0.6", "tilt",
                    "body", tilt, limit, 0.0F);
      return;
    }
    const auto debounce =
        std::chrono::duration<float>(options_.safety.fault_debounce_s);
    const float body_tilt = tilt_radians(state);
    if (persisted(body_tilt > options_.safety.tilt_trip_rad, now,
                  tilt_hazard_since_, debounce))
      latch_fault("body tilt threshold exceeded", "tilt", "body", body_tilt,
                  options_.safety.tilt_trip_rad, debounce.count());
    float max_abs_gyro = 0.0F;
    for (float value : state.imu_state().gyroscope())
      max_abs_gyro = std::max(max_abs_gyro, std::fabs(value));
    if (persisted(max_abs_gyro > options_.safety.angular_rate_trip_rad_s, now,
                  angular_rate_hazard_since_, debounce))
      latch_fault("body angular-rate threshold exceeded", "angular_rate",
                  "body", max_abs_gyro,
                  options_.safety.angular_rate_trip_rad_s, debounce.count());

    for (std::size_t joint = 0; joint < kJoints; ++joint) {
      const auto &motor = state.motor_state()[joint];
      if (motor.q() < kQMin[joint] || motor.q() > kQMax[joint]) {
        const float threshold =
            motor.q() < kQMin[joint] ? kQMin[joint] : kQMax[joint];
        latch_fault(std::string("hard joint position limit: ") +
                        kJointNames[joint],
                    "position", kJointNames[joint], motor.q(), threshold,
                    0.0F);
      }
      if (persisted(
              std::fabs(motor.dq()) >
                  options_.safety.joint_velocity_trip_rad_s[joint],
              now, dq_hazard_since_[joint], debounce))
        latch_fault(std::string("joint velocity threshold: ") +
                        kJointNames[joint],
                    "velocity", kJointNames[joint], std::fabs(motor.dq()),
                    options_.safety.joint_velocity_trip_rad_s[joint],
                    debounce.count());
      if (persisted(
              std::fabs(motor.tau_est()) > options_.safety.torque_limit[joint],
              now, torque_hazard_since_[joint], debounce))
        latch_fault(std::string("estimated torque threshold: ") +
                        kJointNames[joint],
                    "torque", kJointNames[joint],
                    std::fabs(motor.tau_est()),
                    options_.safety.torque_limit[joint], debounce.count());
      if (persisted(
              static_cast<float>(motor.temperature()) >=
                  options_.safety.motor_temperature_trip_c,
              now, temperature_hazard_since_[joint], debounce))
        latch_fault(std::string("motor temperature threshold: ") +
                        kJointNames[joint],
                    "temperature", kJointNames[joint],
                    static_cast<float>(motor.temperature()),
                    options_.safety.motor_temperature_trip_c,
                    debounce.count());
      if (policy_ever_active_ &&
          persisted(
              std::fabs(last_target_[joint] - motor.q()) >
                  options_.safety.tracking_error_trip_rad[joint],
              now, tracking_hazard_since_[joint],
              std::chrono::duration<float>(
                  options_.safety.tracking_error_duration_s)))
        latch_fault(std::string("sustained joint tracking error: ") +
                        kJointNames[joint],
                    "tracking_error", kJointNames[joint],
                    std::fabs(last_target_[joint] - motor.q()),
                    options_.safety.tracking_error_trip_rad[joint],
                    options_.safety.tracking_error_duration_s);
      if (have_lost_baseline_ && motor.lost() > previous_lost_[joint])
        latch_fault(std::string("motor communication lost increment: ") +
                        kJointNames[joint],
                    "lost", kJointNames[joint],
                    static_cast<float>(motor.lost()),
                    static_cast<float>(previous_lost_[joint]), 0.0F);
      previous_lost_[joint] = motor.lost();
    }
    have_lost_baseline_ = true;

    const auto &bms = state.bms_state();
    if (persisted(
            static_cast<float>(bms.soc()) <
                options_.safety.battery_soc_min_percent,
            now, battery_soc_hazard_since_, debounce))
      latch_fault("battery state of charge below threshold", "battery_soc",
                  "battery", static_cast<float>(bms.soc()),
                  options_.safety.battery_soc_min_percent, debounce.count());
    // The SDK exposes this integer without a stable SI unit in its message
    // type. Compare the reviewed raw threshold rather than assuming mA.
    const float battery_current_raw =
        std::fabs(static_cast<float>(bms.current()));
    if (persisted(
            battery_current_raw >
                options_.safety.battery_current_abs_limit_raw,
            now, battery_current_hazard_since_, debounce))
      latch_fault("battery current threshold exceeded", "battery_current_raw",
                  "battery", battery_current_raw,
                  options_.safety.battery_current_abs_limit_raw,
                  debounce.count());
    unsigned int max_battery_temperature = 0;
    for (std::uint8_t value : bms.bq_ntc())
      max_battery_temperature = std::max(
          max_battery_temperature, static_cast<unsigned int>(value));
    for (std::uint8_t value : bms.mcu_ntc())
      max_battery_temperature = std::max(
          max_battery_temperature, static_cast<unsigned int>(value));
    if (persisted(
            static_cast<float>(max_battery_temperature) >=
                options_.safety.battery_temperature_trip_c,
            now, battery_temperature_hazard_since_, debounce))
      latch_fault("battery temperature threshold exceeded",
                  "battery_temperature", "battery",
                  static_cast<float>(max_battery_temperature),
                  options_.safety.battery_temperature_trip_c,
                  debounce.count());
  }

  std::string fault_reason() const {
    std::lock_guard<std::mutex> lock(fault_mutex_);
    return fault_reason_;
  }

  FaultDetail fault_detail() const {
    std::lock_guard<std::mutex> lock(fault_mutex_);
    return fault_detail_;
  }

  void update_blackbox(const std::string &row) {
    if (!fault_latched_.load()) {
      blackbox_pre_rows_.push_back(row);
      while (blackbox_pre_rows_.size() > kBlackBoxPreRows)
        blackbox_pre_rows_.pop_front();
      return;
    }
    if (blackbox_complete_) return;
    if (!blackbox_.is_open()) {
      blackbox_.open(options_.diagnostics_csv + ".blackbox.csv",
                     std::ios::out | std::ios::trunc);
      if (!blackbox_) return;
      blackbox_ << diagnostics_header_ << "\n";
      for (const auto &prior : blackbox_pre_rows_) blackbox_ << prior << "\n";
      blackbox_pre_rows_.clear();
    }
    blackbox_ << row << "\n";
    const std::uint64_t latched_ns = fault_latched_monotonic_ns_.load();
    const std::uint64_t post_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            kBlackBoxPostDuration)
            .count());
    if (latched_ns != 0U && monotonic_ns() - latched_ns >= post_ns) {
      blackbox_.flush();
      blackbox_.close();
      blackbox_complete_ = true;
      std::ofstream marker(options_.diagnostics_csv + ".blackbox.complete.json",
                           std::ios::out | std::ios::trunc);
      if (marker)
        marker << "{\"pre_seconds\":30,\"post_seconds\":30,"
                  "\"status\":\"complete\"}\n";
    }
  }

  void latch_fault(
      const std::string &reason, const std::string &metric = "",
      const std::string &joint = "",
      float value = std::numeric_limits<float>::quiet_NaN(),
      float threshold = std::numeric_limits<float>::quiet_NaN(),
      float duration_s = 0.0F) {
    if (!fault_latched_.exchange(true)) {
      {
        std::lock_guard<std::mutex> lock(fault_mutex_);
        fault_reason_ = reason;
        fault_detail_ = {reason, metric, joint, value, threshold, duration_s};
      }
      fault_latched_monotonic_ns_.store(monotonic_ns());
      std::cerr << "SAFETY FAULT LATCHED: " << reason
                << "; output is SAFE_DAMPING" << std::endl;
      write_fault_record(fault_detail());
    }
  }

  void write_fault_record(const FaultDetail &fault) {
    if (options_.diagnostics_csv.empty() || fault_record_written_.exchange(true))
      return;
    std::ofstream record(options_.diagnostics_csv + ".fault.json",
                         std::ios::out | std::ios::trunc);
    if (!record) return;
    record << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"monotonic_ns\": " << monotonic_ns() << ",\n"
           << "  \"reason\": \"" << fault.reason << "\",\n"
           << "  \"metric\": \"" << fault.metric << "\",\n"
           << "  \"joint\": \"" << fault.joint << "\",\n"
           << "  \"value\": " << json_number(fault.value) << ",\n"
           << "  \"threshold\": " << json_number(fault.threshold) << ",\n"
           << "  \"duration_s\": " << fault.duration_s << ",\n"
           << "  \"mode_before_fault\": \"" << mode_ << "\",\n"
           << "  \"output_mode\": \"SAFE_DAMPING\",\n"
           << "  \"hardware_profile\": \"" << options_.hardware_profile
           << "\",\n"
           << "  \"hardware_profile_sha256\": \""
           << options_.hardware_profile_sha256 << "\",\n"
           << "  \"policy_sha256\": \"" << options_.policy_sha256
           << "\",\n"
           << "  \"robot_serial\": \"" << options_.safety.robot_serial
           << "\",\n"
           << "  \"firmware_version\": \""
           << options_.safety.firmware_version << "\",\n"
           << "  \"blackbox_csv\": \"" << options_.diagnostics_csv
           << ".blackbox.csv\",\n"
           << "  \"blackbox_post_window_s\": 30,\n"
           << "  \"recovery\": \"SAFE_DAMPING remains latched; onsite "
              "inspection and explicit recovery are required\"\n"
           << "}\n";
  }

  void latch_emergency_stop(const char *reason) {
    const bool first_emergency = !emergency_stop_latched_.exchange(true);
    latch_fault(std::string("emergency stop: ") + reason);
    if (first_emergency)
      std::cerr << "EMERGENCY STOP LATCHED: " << reason
                << "; output is SAFE_DAMPING Kp=0 Kd="
                << options_.safety.damping_kd
                << " tau=0; process restart required" << std::endl;
  }

public:
  bool fault_latched() const { return fault_latched_.load(); }

  bool emergency_stop_latched() const {
    return emergency_stop_latched_.load();
  }
};
} // namespace

int main(int argc, char **argv) {
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);
  Options options{};
  bool options_ready = false;
  bool native_release_attempted = false;
  bool inhibit_native_restore = false;
  try {
    options = parse_options(argc, argv);
    options_ready = true;
    ChannelFactory::Instance()->Init(options.domain, options.interface);
    {
      Gateway gateway(options);
      try {
        gateway.init();
        gateway.wait_for_policy_ready("before handover", false);
        if (!options.simulator) {
          native_release_attempted = true;
          release_native_motion_if_needed(options);
          gateway.rearm_policy_watchdog_after_native_release();
        }
        gateway.run();
      } catch (...) {
        inhibit_native_restore = gateway.fault_latched();
        throw;
      }
      inhibit_native_restore = gateway.fault_latched();
    }
    if (inhibit_native_restore)
      std::cerr << "Native motion restore inhibited by latched safety fault"
                << std::endl;
    else
      restore_native_motion_if_needed(options);
  } catch (const std::exception &error) {
    std::cerr << "b2_policy_controller: " << error.what() << std::endl;
    if (options_ready && native_release_attempted && !inhibit_native_restore) {
      try {
        restore_native_motion_if_needed(options);
      } catch (const std::exception &restore_error) {
        std::cerr << "b2_policy_controller: restore failed: "
                  << restore_error.what() << std::endl;
      }
    }
    return 1;
  }
  return 0;
}
