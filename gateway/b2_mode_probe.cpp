#include <iostream>
#include <string>

#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "Usage: b2_mode_probe NETWORK_INTERFACE [--select-ai]"
              << std::endl;
    return 2;
  }
  const std::string interface = argv[1];
  const bool select_ai = argc == 3 && std::string(argv[2]) == "--select-ai";
  if (argc == 3 && !select_ai) {
    std::cerr << "Unknown operation: " << argv[2] << std::endl;
    return 2;
  }
  if (interface.empty() || interface == "lo") {
    std::cerr << "A non-loopback robot Ethernet interface is required." << std::endl;
    return 2;
  }
  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, interface);
    unitree::robot::b2::MotionSwitcherClient client;
    client.SetTimeout(3.0F);
    client.Init();
    if (select_ai) {
      const int32_t select_result = client.SelectMode("ai");
      if (select_result != 0) {
        std::cerr << "SelectMode(ai) failed: " << select_result << std::endl;
        return 1;
      }
      std::cout << "MotionSwitcher: selected native ai mode" << std::endl;
    }
    std::string form;
    std::string name;
    const int32_t result = client.CheckMode(form, name);
    if (result != 0) {
      std::cerr << "CheckMode failed: " << result << std::endl;
      return 1;
    }
    std::cout << "READ-ONLY CheckMode: form=" << form << " name=" << name
              << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "b2_mode_probe: " << error.what() << std::endl;
    return 1;
  }
}
