#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace {
using Clock = std::chrono::steady_clock;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;

constexpr std::uint32_t kStateMagic = 0x54533242;
constexpr std::uint32_t kProtocolVersion = 4;
constexpr std::uint16_t kDryRunPort = 15000;
constexpr double kNativeLowCmdMinHz = 350.0;
constexpr double kNativeLowCmdMaxHz = 650.0;

#pragma pack(push, 1)
struct StatePacket {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t sequence;
  std::uint64_t monotonic_ns;
  float q[12];
  float dq[12];
  float gyro[3];
  float quaternion_wxyz[4];
  float remote_axes[4];
  std::uint32_t remote_keys;
  std::uint32_t remote_flags;
  float tau_est[12];
  std::uint8_t motor_temperature[12];
  std::uint32_t motor_lost[12];
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
#pragma pack(pop)
static_assert(sizeof(StatePacket) == 390);

std::uint64_t monotonic_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch()).count());
}

class Probe {
public:
  Probe() {
    dry_run_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (dry_run_socket_ < 0)
      throw std::runtime_error("cannot create dry-run loopback socket");
    dry_run_destination_.sin_family = AF_INET;
    dry_run_destination_.sin_port = htons(kDryRunPort);
    dry_run_destination_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  }

  ~Probe() {
    if (lowcmd_subscriber_) {
      lowcmd_subscriber_->CloseChannel();
      lowcmd_subscriber_.reset();
    }
    if (remote_subscriber_) {
      remote_subscriber_->CloseChannel();
      remote_subscriber_.reset();
    }
    if (subscriber_) {
      subscriber_->CloseChannel();
      subscriber_.reset();
    }
    if (dry_run_socket_ >= 0) close(dry_run_socket_);
  }

  void init() {
    lowcmd_subscriber_ = std::make_shared<
        ChannelSubscriber<unitree_go::msg::dds_::LowCmd_>>("rt/lowcmd");
    lowcmd_subscriber_->InitChannel(
        [this](const void *) { lowcmd_samples_.fetch_add(1); }, 1);

    subscriber_ = std::make_shared<
        ChannelSubscriber<unitree_go::msg::dds_::LowState_>>("rt/lowstate");
    subscriber_->InitChannel(
        [this](const void *message) {
          const auto &state =
              *static_cast<const unitree_go::msg::dds_::LowState_ *>(message);
          const auto expected = crc32_core(
              const_cast<std::uint32_t *>(
                  reinterpret_cast<const std::uint32_t *>(&state)),
              (sizeof(state) >> 2U) - 1U);
          if (state.crc() != expected) crc_errors_.fetch_add(1);

          bool finite = true;
          for (std::size_t i = 0; i < 12; ++i) {
            finite = finite && std::isfinite(state.motor_state()[i].q()) &&
                     std::isfinite(state.motor_state()[i].dq());
          }
          double norm_squared = 0.0;
          for (float value : state.imu_state().quaternion()) {
            finite = finite && std::isfinite(value);
            norm_squared += static_cast<double>(value) * value;
          }
          const double quaternion_norm = std::sqrt(norm_squared);
          if (!finite || quaternion_norm < 0.5 || quaternion_norm > 1.5)
            invalid_samples_.fetch_add(1);

          {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_ = state;
          }
          const auto sample_number = received_.fetch_add(1) + 1;
          if (sample_number % 10U == 0U) send_dry_run_state(state);
        }, 10);
    remote_subscriber_ = std::make_shared<ChannelSubscriber<
        unitree_go::msg::dds_::WirelessController_>>(
        "rt/wirelesscontroller");
    remote_subscriber_->InitChannel(
        [this](const void *message) {
          const auto remote = *static_cast<const
              unitree_go::msg::dds_::WirelessController_ *>(message);
          if (!std::isfinite(remote.lx()) || !std::isfinite(remote.ly()) ||
              !std::isfinite(remote.rx()) || !std::isfinite(remote.ry()))
            return;
          std::lock_guard<std::mutex> lock(remote_mutex_);
          remote_ = remote;
          remote_received_at_ = Clock::now();
          ++remote_received_;
        }, 1);
  }

  bool report_for(int seconds, bool expect_native_lowcmd) {
    const auto report_started_at = Clock::now();
    std::uint64_t previous = 0;
    for (int elapsed = 1; elapsed <= seconds; ++elapsed) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      const auto count = received_.load();
      unitree_go::msg::dds_::LowState_ snapshot{};
      {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot = latest_;
      }
      if (count == 0) {
        std::cout << "second=" << elapsed << " samples=0" << std::endl;
      } else {
        double quaternion_norm_squared = 0.0;
        for (float value : snapshot.imu_state().quaternion())
          quaternion_norm_squared += static_cast<double>(value) * value;
        std::cout << "second=" << elapsed << " rate_hz=" << count - previous
                  << " crc_errors=" << crc_errors_.load()
                  << " invalid_samples=" << invalid_samples_.load()
                  << " lowcmd_samples=" << lowcmd_samples_.load()
                  << " q0_rad=" << snapshot.motor_state()[0].q()
                  << " dq0_rad_s=" << snapshot.motor_state()[0].dq()
                  << " quat_norm=" << std::sqrt(quaternion_norm_squared)
                  << " remote_topic_samples=" << remote_received_.load();
        const auto &raw = snapshot.wireless_remote();
        std::uint16_t raw_keys = 0;
        float raw_lx = 0.0F;
        float raw_rx = 0.0F;
        float raw_ry = 0.0F;
        float raw_ly = 0.0F;
        std::memcpy(&raw_keys, raw.data() + 2, sizeof(raw_keys));
        std::memcpy(&raw_lx, raw.data() + 4, sizeof(raw_lx));
        std::memcpy(&raw_rx, raw.data() + 8, sizeof(raw_rx));
        std::memcpy(&raw_ry, raw.data() + 12, sizeof(raw_ry));
        std::memcpy(&raw_ly, raw.data() + 20, sizeof(raw_ly));
        std::cout << " raw_remote_head=0x" << std::hex
                  << static_cast<unsigned int>(raw[0])
                  << static_cast<unsigned int>(raw[1]) << std::dec
                  << " raw_keys=" << raw_keys
                  << " raw_axes_lx_ly_rx_ry=[" << raw_lx << "," << raw_ly
                  << "," << raw_rx << "," << raw_ry << "]" << std::endl;
      }
      previous = count;
    }
    const double report_elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - report_started_at).count();
    const auto lowcmd_samples = lowcmd_samples_.load();
    const double lowcmd_rate_hz =
        report_elapsed_seconds > 0.0
            ? static_cast<double>(lowcmd_samples) / report_elapsed_seconds
            : 0.0;
    const bool native_rate_ok =
        lowcmd_rate_hz >= kNativeLowCmdMinHz &&
        lowcmd_rate_hz <= kNativeLowCmdMaxHz;
    const bool lowcmd_publisher_conflict =
        expect_native_lowcmd ? !native_rate_ok : lowcmd_samples > 0;
    const bool passed = received_.load() > 0 && crc_errors_.load() == 0 &&
                        invalid_samples_.load() == 0 &&
                        !lowcmd_publisher_conflict;
    std::cout << (passed ? "PASS" : "FAIL")
              << " samples=" << received_.load()
              << " crc_errors=" << crc_errors_.load()
              << " invalid_samples=" << invalid_samples_.load()
              << " lowcmd_policy="
              << (expect_native_lowcmd ? "native_500hz" : "strict_none")
              << " lowcmd_samples=" << lowcmd_samples
              << " lowcmd_rate_hz=" << std::fixed << std::setprecision(3)
              << lowcmd_rate_hz
              << " lowcmd_native_expected="
              << (expect_native_lowcmd ? 1 : 0)
              << " lowcmd_native_rate_ok=" << (native_rate_ok ? 1 : 0)
              << " lowcmd_publisher_conflict="
              << (lowcmd_publisher_conflict ? 1 : 0) << std::endl;
    return passed;
  }

private:
  void send_dry_run_state(const unitree_go::msg::dds_::LowState_ &state) {
    StatePacket packet{};
    packet.magic = kStateMagic;
    packet.version = kProtocolVersion;
    packet.sequence = ++dry_run_sequence_;
    packet.monotonic_ns = monotonic_ns();
    for (std::size_t i = 0; i < 12; ++i) {
      packet.q[i] = state.motor_state()[i].q();
      packet.dq[i] = state.motor_state()[i].dq();
      packet.tau_est[i] = state.motor_state()[i].tau_est();
      packet.motor_temperature[i] = state.motor_state()[i].temperature();
      packet.motor_lost[i] = state.motor_state()[i].lost();
    }
    for (std::size_t i = 0; i < 3; ++i)
      packet.gyro[i] = state.imu_state().gyroscope()[i];
    for (std::size_t i = 0; i < 4; ++i)
      packet.quaternion_wxyz[i] = state.imu_state().quaternion()[i];
    bool remote_from_topic = false;
    {
      std::lock_guard<std::mutex> lock(remote_mutex_);
      packet.remote_axes[0] = remote_.lx();
      packet.remote_axes[1] = remote_.ly();
      packet.remote_axes[2] = remote_.rx();
      packet.remote_axes[3] = remote_.ry();
      packet.remote_keys = remote_.keys();
      if (remote_received_.load() > 0 &&
          Clock::now() - remote_received_at_ <= std::chrono::milliseconds(250)) {
        packet.remote_flags = 1U;
        remote_from_topic = true;
      }
    }
    if (!remote_from_topic) {
      const auto &raw = state.wireless_remote();
      if (raw[0] == 0x55U && raw[1] == 0x51U) {
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
        const bool finite = std::isfinite(lx) && std::isfinite(ly) &&
                            std::isfinite(rx) && std::isfinite(ry);
        if (finite) {
          packet.remote_axes[0] = lx;
          packet.remote_axes[1] = ly;
          packet.remote_axes[2] = rx;
          packet.remote_axes[3] = ry;
          packet.remote_keys = keys;
          packet.remote_flags = 1U;
        }
      }
    }
    const auto &bms = state.bms_state();
    packet.battery_soc = bms.soc();
    packet.battery_status = bms.status();
    packet.battery_current_raw = bms.current();
    packet.battery_temperature[0] = bms.bq_ntc()[0];
    packet.battery_temperature[1] = bms.bq_ntc()[1];
    packet.battery_temperature[2] = bms.mcu_ntc()[0];
    packet.battery_temperature[3] = bms.mcu_ntc()[1];
    std::copy(bms.cell_vol().begin(), bms.cell_vol().end(),
              packet.battery_cell_mv);
    packet.state_samples = received_.load();
    packet.state_crc_errors = crc_errors_.load();
    packet.invalid_states = invalid_samples_.load();
    sendto(dry_run_socket_, &packet, sizeof(packet), 0,
           reinterpret_cast<sockaddr *>(&dry_run_destination_),
           sizeof(dry_run_destination_));
  }

  ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> subscriber_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_subscriber_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_>
      remote_subscriber_;
  std::mutex mutex_;
  unitree_go::msg::dds_::LowState_ latest_{};
  std::atomic<std::uint64_t> received_{0};
  std::atomic<std::uint64_t> crc_errors_{0};
  std::atomic<std::uint64_t> invalid_samples_{0};
  std::atomic<std::uint64_t> lowcmd_samples_{0};
  std::mutex remote_mutex_;
  unitree_go::msg::dds_::WirelessController_ remote_{};
  Clock::time_point remote_received_at_{};
  std::atomic<std::uint64_t> remote_received_{0};
  int dry_run_socket_ = -1;
  sockaddr_in dry_run_destination_{};
  std::uint64_t dry_run_sequence_ = 0;
};
} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 4) {
    std::cerr << "Usage: b2_hw_probe NETWORK_INTERFACE [SECONDS] "
                 "[--expect-native-lowcmd]"
              << std::endl;
    return 2;
  }
  const std::string interface = argv[1];
  if (interface.empty() || interface == "lo") {
    std::cerr << "A non-loopback robot Ethernet interface is required." << std::endl;
    return 2;
  }
  int seconds = 10;
  bool seconds_set = false;
  bool expect_native_lowcmd = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--expect-native-lowcmd") {
      if (expect_native_lowcmd) {
        std::cerr << "--expect-native-lowcmd may only be specified once."
                  << std::endl;
        return 2;
      }
      expect_native_lowcmd = true;
      continue;
    }
    if (seconds_set) {
      std::cerr << "Usage: b2_hw_probe NETWORK_INTERFACE [SECONDS] "
                   "[--expect-native-lowcmd]"
                << std::endl;
      return 2;
    }
    char *end = nullptr;
    const long parsed = std::strtol(argument.c_str(), &end, 10);
    if (end == argument.c_str() || *end != '\0' || parsed < 1 ||
        parsed > 300) {
      std::cerr << "SECONDS must be an integer from 1 through 300."
                << std::endl;
      return 2;
    }
    seconds = static_cast<int>(parsed);
    seconds_set = true;
  }
  try {
    std::cout << "READ-ONLY: subscribing to rt/lowstate on domain 0 via "
              << interface << " lowcmd_policy="
              << (expect_native_lowcmd ? "native_500hz" : "strict_none")
              << std::endl;
    ChannelFactory::Instance()->Init(0, interface);
    Probe probe;
    probe.init();
    return probe.report_for(seconds, expect_native_lowcmd) ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "b2_hw_probe: " << error.what() << std::endl;
    return 1;
  }
}
