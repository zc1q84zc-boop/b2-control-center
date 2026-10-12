#!/usr/bin/env bash
set -euo pipefail

if test "$#" -lt 1 || test "$#" -gt 2; then
  echo "Usage: scripts/run_hw_probe.sh NETWORK_INTERFACE [SECONDS]" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_prefix="${UNITREE_SDK2_PREFIX:-${repo_root}/third_party/unitree_sdk2/install}"
probe="${repo_root}/build/gateway/b2_hw_probe"
test -x "${probe}" || {
  echo "Run scripts/build_sim_stack.sh first." >&2
  exit 1
}
exec env LD_LIBRARY_PATH="${sdk_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${probe}" "$@"
