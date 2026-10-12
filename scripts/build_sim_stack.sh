#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_prefix="${UNITREE_SDK2_PREFIX:-${repo_root}/third_party/unitree_sdk2/install}"
deps_prefix="${B2_DEPS_PREFIX:-/usr}"
mujoco_prefix="${MUJOCO_PREFIX:-${repo_root}/third_party/mujoco}"
build_dir="${repo_root}/build/gateway"
sim_dir="${repo_root}/third_party/unitree_mujoco/simulate"

test -f "${sdk_prefix}/lib/cmake/unitree_sdk2/unitree_sdk2Config.cmake" || {
  echo "Missing Unitree SDK2 install: ${sdk_prefix}" >&2
  echo "Set UNITREE_SDK2_PREFIX to its CMake install prefix." >&2
  exit 1
}
test -d "${mujoco_prefix}/lib" || {
  echo "Missing MuJoCo install: ${mujoco_prefix}" >&2
  echo "Set MUJOCO_PREFIX to the directory containing lib/." >&2
  exit 1
}

if ! test -e "${sim_dir}/mujoco"; then
  cmake -E create_symlink "${mujoco_prefix}" "${sim_dir}/mujoco"
fi
cmake -S "${sim_dir}" -B "${sim_dir}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${sdk_prefix};${deps_prefix}" \
  -DBOOST_ROOT="${deps_prefix}" \
  -DCMAKE_CXX_FLAGS="-I${deps_prefix}/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${deps_prefix}/lib/x86_64-linux-gnu -Wl,-rpath,${deps_prefix}/lib/x86_64-linux-gnu"
cmake --build "${sim_dir}/build" --target unitree_mujoco -j2

cmake -S "${repo_root}/gateway" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${sdk_prefix}"
cmake --build "${build_dir}" -j2
echo "Built official B2 MuJoCo and all gateway/probe targets"
