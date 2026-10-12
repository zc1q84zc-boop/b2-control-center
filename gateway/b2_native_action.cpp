#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/b2/robot_state/robot_state_client.hpp>
#include <unitree/robot/b2/sport/sport_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

namespace {

constexpr float kMaxVx = 1.0F;
constexpr float kMaxVy = 0.8F;
constexpr float kMaxWz = 0.8F;

float parse_bounded_float(const char *text, const std::string &name,
                          float absolute_limit) {
  const std::string input(text);
  std::size_t parsed = 0;
  double value = 0.0;
  try {
    value = std::stod(input, &parsed);
  } catch (const std::exception &) {
    throw std::invalid_argument(name + " must be a finite number");
  }
  if (parsed != input.size() || !std::isfinite(value))
    throw std::invalid_argument(name + " must be a finite number");
  if (value < -absolute_limit || value > absolute_limit)
    throw std::invalid_argument(
        name + " exceeds the allowed range [" +
        std::to_string(-absolute_limit) + ", " +
        std::to_string(absolute_limit) + "]");
  return static_cast<float>(value);
}

int run_action(const std::string &action, float vx = 0.0F, float vy = 0.0F,
               float wz = 0.0F) {
  if (action == "low-power-on" || action == "low-power-off" ||
      action == "low-power-status") {
    unitree::robot::b2::RobotStateClient state;
    state.SetTimeout(5.0F);
    state.Init();
    if (action == "low-power-status") {
      int status = 0;
      const int result = state.LowPowerStatus(status);
      std::cout << "native_action=low-power-status result=" << result
                << " status=" << status << std::endl;
      return result;
    }
    return state.LowPowerSwitch(action == "low-power-on" ? 1 : 0);
  }

  unitree::robot::b2::SportClient sport;
  sport.SetTimeout(25.0F);
  sport.Init();
  if (action == "stop") return sport.StopMove();
  if (action == "stand-up") return sport.StandUp();
  if (action == "stand-down") return sport.StandDown();
  if (action == "recovery-stand") return sport.RecoveryStand();
  if (action == "balance-stand") return sport.BalanceStand();
  if (action == "damping") return sport.Damp();
  if (action == "move") return sport.Move(vx, vy, wz);
  throw std::invalid_argument("unsupported native action: " + action);
}

[[noreturn]] void throw_usage() {
  throw std::invalid_argument(
      "usage: b2_native_action INTERFACE ACTION "
      "--acknowledge-native-robot-action\n"
      "       b2_native_action INTERFACE move --vx VX --vy VY --wz WZ "
      "--acknowledge-native-robot-action\n"
      "move limits: |VX| <= 1.0 m/s, |VY| <= 0.8 m/s, "
      "|WZ| <= 0.8 rad/s");
}

} // namespace

int main(int argc, char **argv) {
  try {
    const bool simple_action =
        argc == 4 &&
        std::string(argv[3]) == "--acknowledge-native-robot-action";
    const bool move_action =
        argc == 10 && std::string(argv[2]) == "move" &&
        std::string(argv[3]) == "--vx" && std::string(argv[5]) == "--vy" &&
        std::string(argv[7]) == "--wz" &&
        std::string(argv[9]) == "--acknowledge-native-robot-action";
    if (!simple_action && !move_action) throw_usage();
    const std::string interface = argv[1];
    const std::string action = argv[2];
    if (action == "move" && !move_action) throw_usage();
    const float vx = move_action ? parse_bounded_float(argv[4], "vx", kMaxVx)
                                 : 0.0F;
    const float vy = move_action ? parse_bounded_float(argv[6], "vy", kMaxVy)
                                 : 0.0F;
    const float wz = move_action ? parse_bounded_float(argv[8], "wz", kMaxWz)
                                 : 0.0F;
    if (interface.empty() || interface == "lo")
      throw std::invalid_argument("native action requires a physical interface");
    unitree::robot::ChannelFactory::Instance()->Init(0, interface);

    unitree::robot::b2::MotionSwitcherClient switcher;
    switcher.SetTimeout(5.0F);
    switcher.Init();
    std::string form;
    std::string name;
    const int mode_result = switcher.CheckMode(form, name);
    if (mode_result != 0)
      throw std::runtime_error("MotionSwitcher CheckMode failed");
    if (action == "restore-ai") {
      if (name == "ai") {
        std::cout << "native_action=restore-ai result=0 mode=ai" << std::endl;
        return 0;
      }
      if (!name.empty())
        throw std::runtime_error(
            "another motion service is active; restore-ai refused");
      const int result = switcher.SelectMode("ai");
      std::cout << "native_action=restore-ai result=" << result << std::endl;
      return result == 0 ? 0 : 1;
    }
    if (name != "ai")
      throw std::runtime_error(
          "native ai locomotion is not active; action refused");

    const int result = run_action(action, vx, vy, wz);
    std::cout << "native_action=" << action << " result=" << result
              << " mode=" << name;
    if (action == "move")
      std::cout << " vx=" << vx << " vy=" << vy << " wz=" << wz;
    std::cout << std::endl;
    return result == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "b2_native_action: " << error.what() << std::endl;
    return 2;
  }
}
