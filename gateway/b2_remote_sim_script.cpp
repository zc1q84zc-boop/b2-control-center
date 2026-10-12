#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>

#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::uint16_t kR2 = 1U << 4U;
constexpr std::uint16_t kL2 = 1U << 5U;
constexpr std::uint16_t kA = 1U << 8U;
constexpr std::uint16_t kB = 1U << 9U;

const char *phase(double elapsed_s) {
  if (elapsed_s < 8.0) return "CENTERED_WAIT";
  if (elapsed_s < 8.3) return "A_POLICY";
  if (elapsed_s < 12.3) return "FORWARD_0.25_M_S";
  if (elapsed_s < 12.6) return "R2_STAND";
  if (elapsed_s < 14.5) return "LEARNED_HOLD";
  return "L2_B_ESTOP";
}
} // namespace

int main() {
  try {
    unitree::robot::ChannelFactory::Instance()->Init(1, "lo");
    auto publisher = std::make_shared<unitree::robot::ChannelPublisher<
        unitree_go::msg::dds_::WirelessController_>>(
        "rt/wirelesscontroller");
    publisher->InitChannel();
    std::cout << "SIM-ONLY remote script on DDS domain 1 / lo" << std::endl;
    const auto started = Clock::now();
    auto next = started;
    const char *previous_phase = "";
    while (true) {
      const double elapsed =
          std::chrono::duration<double>(Clock::now() - started).count();
      if (elapsed >= 15.5) break;
      unitree_go::msg::dds_::WirelessController_ message{};
      if (elapsed >= 8.0 && elapsed < 8.3)
        message.keys() = kA;
      else if (elapsed >= 8.3 && elapsed < 12.3)
        message.ly() = 0.28F;
      else if (elapsed >= 12.3 && elapsed < 12.6)
        message.keys() = kR2;
      else if (elapsed >= 14.5)
        message.keys() = kL2 | kB;
      const char *current_phase = phase(elapsed);
      if (current_phase != previous_phase) {
        std::cout << "remote_script phase=" << current_phase << std::endl;
        previous_phase = current_phase;
      }
      publisher->Write(message);
      next += std::chrono::milliseconds(20);
      std::this_thread::sleep_until(next);
    }
    std::cout << "SIM_REMOTE_SCRIPT_PASS" << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "b2_remote_sim_script: " << error.what() << std::endl;
    return 1;
  }
}
