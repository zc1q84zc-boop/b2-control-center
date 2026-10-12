#!/usr/bin/env bash
set -euo pipefail

if test "$#" -ne 2; then
  echo "Usage: scripts/run_hardware_readonly_acceptance.sh NETWORK_INTERFACE POLICY_BUNDLE" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
interface="$1"
bundle="$2"
python_bin="${B2_PYTHON:-${repo_root}/.venv/bin/python}"
sdk_prefix="${UNITREE_SDK2_PREFIX:-${repo_root}/third_party/unitree_sdk2/install}"
sdk_lib="${sdk_prefix}/lib"

test "${interface}" != "lo" || {
  echo "A non-loopback robot Ethernet interface is required." >&2
  exit 2
}
test -d "${bundle}" || {
  echo "Policy bundle does not exist: ${bundle}" >&2
  exit 2
}
test -x "${python_bin}" || {
  echo "Python environment does not exist: ${python_bin}" >&2
  exit 2
}

run_shadow() {
  local label="$1"
  shift
  echo "=== ${label} ==="
  env PYTHONPATH="${repo_root}/src" "${python_bin}"     -m b2_control.cli_hw_dryrun "${bundle}" --duration 10 "$@" &
  local policy_pid=$!
  if ! "${repo_root}/scripts/run_hw_probe.sh" "${interface}" 12; then
    kill "${policy_pid}" 2>/dev/null || true
    wait "${policy_pid}" 2>/dev/null || true
    return 1
  fi
  wait "${policy_pid}"
}

echo "=== native mode ==="
env LD_LIBRARY_PATH="${sdk_lib}"   "${repo_root}/build/gateway/b2_mode_probe" "${interface}"
echo "=== state gate ==="
"${repo_root}/scripts/run_hw_probe.sh" "${interface}" 3
run_shadow "learned hold shadow" --hold
run_shadow "0.25 m/s forward shadow" --speed 0.25
echo "READ-ONLY ACCEPTANCE PASS: no LowCmd publisher was created."
