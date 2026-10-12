#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_prefix="${UNITREE_SDK2_PREFIX:-${repo_root}/third_party/unitree_sdk2/install}"
gateway="${repo_root}/build/gateway/b2_policy_controller"
test -x "${gateway}" || {
  echo "Run scripts/build_sim_stack.sh first." >&2
  exit 1
}
exec env LD_LIBRARY_PATH="${sdk_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${gateway}" --sim
