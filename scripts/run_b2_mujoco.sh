#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim_dir="${repo_root}/third_party/unitree_mujoco/simulate"
deps_prefix="${B2_DEPS_PREFIX:-/usr}"
sdk_prefix="${UNITREE_SDK2_PREFIX:-${repo_root}/third_party/unitree_sdk2/install}"
mujoco_prefix="${MUJOCO_PREFIX:-${repo_root}/third_party/mujoco}"
deps="${deps_prefix}/lib/x86_64-linux-gnu"
sdk_lib="${sdk_prefix}/lib"
mujoco_lib="${mujoco_prefix}/lib"
ffmpeg_bin="${B2_FFMPEG:-$(command -v ffmpeg || true)}"
display_name="${DISPLAY:-:1}"
scene_name="${B2_SIM_SCENE:-scene.xml}"
video_path="${B2_SIM_VIDEO:-${repo_root}/logs/videos/sim-$(date +%Y%m%d-%H%M%S).mp4}"
case "${video_path}" in
  /*) ;;
  *) video_path="${repo_root}/${video_path}" ;;
esac
recording_path="${video_path}.recording.mp4"

test -x "${sim_dir}/build/unitree_mujoco" || {
  echo "Missing official Unitree MuJoCo binary: ${sim_dir}/build/unitree_mujoco" >&2
  exit 1
}
test -x "${ffmpeg_bin}" || {
  echo "Missing ffmpeg. Install it or set B2_FFMPEG to an executable." >&2
  exit 1
}
command -v xdpyinfo >/dev/null 2>&1 || {
  echo "Missing xdpyinfo required for mandatory simulation recording." >&2
  exit 1
}

dimensions="$(xdpyinfo -display "${display_name}" 2>/dev/null | sed -n 's/^[[:space:]]*dimensions:[[:space:]]*\([0-9][0-9]*x[0-9][0-9]*\).*/\1/p' | head -n 1)"
test -n "${dimensions}" || {
  echo "Cannot read X11 dimensions for mandatory recording on ${display_name}." >&2
  exit 1
}
mkdir -p "$(dirname "${video_path}")"

recorder_pid=""
simulator_pid=""
cleaned_up=0
cleanup() {
  if test "${cleaned_up}" -eq 1; then
    return
  fi
  cleaned_up=1
  if test -n "${simulator_pid}"; then
    kill -TERM "${simulator_pid}" 2>/dev/null || true
    wait "${simulator_pid}" 2>/dev/null || true
  fi
  if test -n "${recorder_pid}"; then
    kill -INT "${recorder_pid}" 2>/dev/null || true
    wait "${recorder_pid}" 2>/dev/null || true
  fi
  if test ! -s "${recording_path}"; then
    echo "Mandatory simulation recording was not created: ${recording_path}" >&2
  elif "${ffmpeg_bin}" -hide_banner -loglevel error -y \
      -fflags +discardcorrupt -err_detect ignore_err \
      -i "${recording_path}" -map 0:v:0 \
      -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p \
      -movflags +faststart "${video_path}"; then
    rm -f -- "${recording_path}"
    echo "Simulation video saved: ${video_path}"
  else
    echo "Mandatory simulation video finalization failed; raw recording kept: ${recording_path}" >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${ffmpeg_bin}" -hide_banner -loglevel warning -y \
  -f x11grab -framerate 20 -video_size "${dimensions}" \
  -i "${display_name}+0,0" -vf scale=1280:720 \
  -c:v libx264 -preset ultrafast -crf 24 -pix_fmt yuv420p \
  -movflags +frag_keyframe+empty_moov+default_base_moof \
  "${recording_path}" &
recorder_pid=$!
sleep 1
if ! kill -0 "${recorder_pid}" 2>/dev/null; then
  echo "Mandatory simulation recorder failed to start." >&2
  exit 1
fi

echo "Mandatory simulation video: ${video_path}"
echo "Official Unitree MuJoCo scene: ${scene_name}"
cd "${sim_dir}"
env LD_LIBRARY_PATH="${deps}:${sdk_lib}:${mujoco_lib}" \
  ./build/unitree_mujoco -r b2 -s "${scene_name}" &
simulator_pid=$!
wait "${simulator_pid}"
status=$?
simulator_pid=""
exit "${status}"
