cmake_minimum_required(VERSION 3.16)
project(b2_sim_gateway LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(unitree_sdk2 2.0.0 EXACT REQUIRED)

add_library(b2_hardware_data STATIC ../src/hardware_adapter.cpp)
target_include_directories(b2_hardware_data PUBLIC ../src)
target_compile_options(b2_hardware_data PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(b2_policy_controller b2_sim_gateway.cpp)
target_link_libraries(b2_policy_controller PRIVATE unitree_sdk2)
target_compile_options(b2_policy_controller PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(b2_hw_probe b2_hw_probe.cpp)
target_link_libraries(b2_hw_probe PRIVATE unitree_sdk2)
target_compile_options(b2_hw_probe PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(b2_mode_probe b2_mode_probe.cpp)
target_link_libraries(b2_mode_probe PRIVATE unitree_sdk2)
target_compile_options(b2_mode_probe PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(b2_native_action b2_native_action.cpp)
target_link_libraries(b2_native_action PRIVATE unitree_sdk2)
target_compile_options(b2_native_action PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(b2_remote_sim_script b2_remote_sim_script.cpp)
target_link_libraries(b2_remote_sim_script PRIVATE unitree_sdk2)
target_compile_options(
  b2_remote_sim_script PRIVATE -Wall -Wextra -Wpedantic -Werror
)

include(CTest)
if(BUILD_TESTING)
  add_executable(b2_control_data_test ../tests/cpp/test_control_data_layer.cpp)
  target_link_libraries(b2_control_data_test PRIVATE b2_hardware_data)
  target_compile_options(
    b2_control_data_test PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
  add_test(NAME b2_control_data_test COMMAND b2_control_data_test)
endif()
