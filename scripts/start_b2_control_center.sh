#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if test -n "${B2_PYTHON:-}"; then
  python_bin="${B2_PYTHON}"
elif test -x "${repo_root}/.venv/bin/python"; then
  python_bin="${repo_root}/.venv/bin/python"
else
  python_bin="$(command -v python3 || true)"
fi
lock_file="${XDG_RUNTIME_DIR:-/tmp}/b2-control-center-${UID}.lock"

test -x "${python_bin}" || {
  echo "B2 Control Center Python is unavailable. Create .venv or set B2_PYTHON." >&2
  exit 1
}

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "B2 Control Center is already running."
  exit 1
fi

cd "${repo_root}"
exec env PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" -m b2_control.cli_control_center "$@"
