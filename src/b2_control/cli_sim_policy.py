from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

from .bundle import open_verified_bundle
from .core import B2PolicyCore
from .control_lifecycle import ControlLifecycle, lifecycle_for_gateway
from .hardware_mode_manager import GatewayStatus, HardwareModeManager
from .interactive_command import InteractiveVelocityCommand, TerminalKeyboard
from .onnx_policy import OnnxPolicy
from .operator_ui import OperatorWebUI
from .policy_math import JOINT_NAMES
from .stack10_adapter import Stack10ObservationAdapter
from .remote_control import (
    RemoteMode,
    RemoteModeController,
    RemoteVelocityMapper,
)
from .runtime_metrics import RuntimeMetricsSampler
from .sim_protocol import (
    COMMAND_PORT,
    GATEWAY_MODE_NAMES,
    STATE_PORT,
    STATE_STRUCT,
    decode_state,
    encode_command,
    quaternion_wxyz_to_rotation,
    yaw_from_rotation,
)
from .timed_test import TimedTestRunner


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the B2 policy through the local DDS gateway"
    )
    parser.add_argument("bundle", help="verified b2_policy_deployment_handoff directory")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.25,
        help="forward speed in m/s; trained range is 0.25..1.0 (default: 0.25)",
    )
    parser.add_argument(
        "--heading-offset",
        type=float,
        default=0.0,
        help="target yaw offset in radians from startup heading",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="stop after this many seconds of policy control",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="run learned zero-velocity hold instead of a forward command",
    )
    parser.add_argument(
        "--interactive-keyboard",
        action="store_true",
        help="control policy command [vx,vy,wz] from this terminal",
    )
    parser.add_argument(
        "--operator-ui",
        "--operator-gui",
        action="store_true",
        help="open the loopback browser control panel",
    )
    parser.add_argument(
        "--operator-ui-port",
        type=int,
        default=8765,
        help="loopback operator UI port (default: 8765)",
    )
    parser.add_argument(
        "--sim-keyboard-port",
        type=int,
        help="receive MuJoCo-window key press/release events on loopback UDP",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="serve the operator UI without opening another browser window",
    )
    parser.add_argument(
        "--hardware-console",
        metavar="INTERFACE",
        help="let the persistent browser console manage the real B2 gateway",
    )
    parser.add_argument(
        "--acknowledge-b2-low-level-control",
        action="store_true",
        help="required acknowledgement for --hardware-console",
    )
    parser.add_argument(
        "--gateway-executable",
        type=Path,
        default=PROJECT_ROOT / "build/gateway/b2_policy_controller",
        help="managed C++ gateway executable",
    )
    parser.add_argument(
        "--mode-probe-executable",
        type=Path,
        default=PROJECT_ROOT / "build/gateway/b2_mode_probe",
        help="MotionSwitcher verification executable",
    )
    parser.add_argument(
        "--sdk-library-path",
        type=Path,
        default=Path(
            os.environ.get(
                "UNITREE_SDK2_LIB",
                str(PROJECT_ROOT / "third_party/unitree_sdk2/install/lib"),
            )
        ),
        help="directory containing Unitree SDK shared libraries",
    )
    parser.add_argument(
        "--managed-gateway-csv",
        type=Path,
        default=PROJECT_ROOT / "logs/operator-console-gateway.csv",
        help="diagnostics CSV for the console-managed hardware gateway",
    )
    parser.add_argument(
        "--managed-gateway-log",
        type=Path,
        default=PROJECT_ROOT / "logs/operator-console-gateway.log",
        help="text log for the console-managed hardware gateway",
    )
    parser.add_argument(
        "--managed-gateway-pid",
        type=Path,
        default=PROJECT_ROOT / "logs/operator-console-gateway.pid",
        help="PID record used only by the explicit onsite fault-recovery workflow",
    )
    parser.add_argument(
        "--hardware-profile",
        type=Path,
        default=PROJECT_ROOT / "configs/b2-reference-profile.json",
        help="single reviewed hardware safety profile used for real LowCmd activation",
    )
    parser.add_argument(
        "--remote-control",
        action="store_true",
        help="use the Unitree hand controller: A policy, R2 stand, L2+B E-stop",
    )
    parser.add_argument(
        "--remote-deadzone",
        type=float,
        default=0.10,
        help="hand-controller stick deadzone (default: 0.10)",
    )
    parser.add_argument(
        "--yaw-step",
        type=float,
        default=0.10,
        help="Q/E yaw-rate increment in rad/s (default: 0.10)",
    )
    parser.add_argument(
        "--max-linear-speed",
        type=float,
        default=0.90,
        help="maximum W/S or remote forward speed in m/s (default: 0.90)",
    )
    parser.add_argument(
        "--initial-linear-speed",
        type=float,
        help="initial W/S speed before a 1..9 preset is selected (default: maximum)",
    )
    parser.add_argument(
        "--max-lateral-speed",
        type=float,
        default=0.25,
        help="fixed A/D speed in m/s (default: 0.25)",
    )
    parser.add_argument(
        "--max-yaw-rate",
        type=float,
        default=0.40,
        help="keyboard wz absolute limit in rad/s (default: 0.40)",
    )
    parser.add_argument(
        "--warmup-hold-duration",
        type=float,
        default=2.0,
        help="learned-hold seconds before a nonzero motion command (default: 2)",
    )
    parser.add_argument(
        "--target-slew-rad-s",
        type=float,
        default=8.0,
        help="policy target slew limit in rad/s (default: 8.0)",
    )
    parser.add_argument(
        "--disable-target-slew",
        action="store_true",
        help="disable the policy-side q-target slew limiter",
    )
    parser.add_argument(
        "--action-clip-abs",
        type=float,
        help="clip applied actor output to [-VALUE,+VALUE]; raw feedback is preserved",
    )
    parser.add_argument(
        "--diagnostics-csv",
        help="write every 50 Hz policy/state sample to this CSV path",
    )
    parser.add_argument(
        "--operator-event-log",
        type=Path,
        help="append browser/keyboard/mode/test operations as JSONL",
    )
    parser.add_argument(
        "--test-results",
        type=Path,
        help="append completed timed-test results as JSONL",
    )
    parser.add_argument(
        "--keep-hold-after-duration",
        action="store_true",
        help="after bounded motion, keep learned-hold running until interrupted",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.interactive_keyboard and (args.operator_ui or args.remote_control):
        raise SystemExit(
            "terminal keyboard is mutually exclusive with browser UI and hand remote"
        )
    operator_modes = any(
        (args.interactive_keyboard, args.operator_ui, args.remote_control)
    )
    if args.hold and operator_modes:
        raise SystemExit(
            "--hold and interactive operator control are mutually exclusive"
        )
    keyboard_values = np.asarray(
        (
            args.yaw_step,
            args.max_linear_speed,
            args.max_lateral_speed,
            args.max_yaw_rate,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(keyboard_values)) or np.any(keyboard_values <= 0.0):
        raise SystemExit("keyboard steps and limits must be positive finite values")
    if args.yaw_step > args.max_yaw_rate:
        raise SystemExit("--yaw-step cannot exceed --max-yaw-rate")
    if args.initial_linear_speed is not None and (
        not np.isfinite(args.initial_linear_speed)
        or not 0.05 <= args.initial_linear_speed <= args.max_linear_speed
    ):
        raise SystemExit(
            "--initial-linear-speed must be in [0.05, --max-linear-speed]"
        )
    if not 0 <= args.operator_ui_port <= 65535:
        raise SystemExit("--operator-ui-port must be 0..65535")
    if args.sim_keyboard_port is not None and not 0 <= args.sim_keyboard_port <= 65535:
        raise SystemExit("--sim-keyboard-port must be 0..65535")
    if args.sim_keyboard_port is not None and not args.operator_ui:
        raise SystemExit("--sim-keyboard-port requires --operator-ui")
    if args.hardware_console:
        if not args.operator_ui:
            raise SystemExit("--hardware-console requires --operator-ui")
        if args.hardware_console == "lo":
            raise SystemExit("--hardware-console requires a physical interface")
        if not args.acknowledge_b2_low_level_control:
            raise SystemExit(
                "--hardware-console requires --acknowledge-b2-low-level-control"
            )
        if not args.hardware_profile.is_file():
            raise SystemExit(
                f"--hardware-profile is unavailable: {args.hardware_profile}"
            )
    if not np.isfinite(args.remote_deadzone) or not 0.01 <= args.remote_deadzone < 1.0:
        raise SystemExit("--remote-deadzone must be in [0.01,1.0)")
    if not 0.25 <= args.speed <= 1.0:
        raise SystemExit("--speed must remain inside the trained range 0.25..1.0")
    if args.duration is not None and not 1 <= args.duration <= 300:
        raise SystemExit("--duration must be 1..300 seconds")
    if not 0.0 <= args.warmup_hold_duration <= 30.0:
        raise SystemExit("--warmup-hold-duration must be 0..30 seconds")
    if not args.disable_target_slew and not 0.1 <= args.target_slew_rad_s <= 23.0:
        raise SystemExit("--target-slew-rad-s must be 0.1..23.0")
    if args.action_clip_abs is not None and not 0.1 <= args.action_clip_abs <= 20.0:
        raise SystemExit("--action-clip-abs must be 0.1..20.0")
    if args.keep_hold_after_duration and (args.duration is None or args.hold):
        raise SystemExit(
            "--keep-hold-after-duration requires bounded non-hold motion"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

    bundle = open_verified_bundle(args.bundle)
    observation_adapter = (
        Stack10ObservationAdapter() if bundle.adapter == "b2_stack10_v1" else None
    )
    policy = OnnxPolicy(
        bundle.policy_path,
        input_dim=bundle.input_dim,
        output_dim=bundle.output_dim,
    )
    warmup_started = time.perf_counter_ns()
    policy.warmup()
    warmup_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000.0
    print(f"ONNX session warmup complete in {warmup_ms:.3f} ms")
    core = B2PolicyCore(
        policy,
        observation_adapter=observation_adapter,
        target_slew_rad_s=(
            None if args.disable_target_slew else args.target_slew_rad_s
        ),
        action_clip_abs=args.action_clip_abs,
    )
    keyboard = TerminalKeyboard()
    velocity_command = InteractiveVelocityCommand(
        yaw_step=args.yaw_step,
        max_linear_speed=args.max_linear_speed,
        max_yaw_rate=args.max_yaw_rate,
        max_lateral_speed=args.max_lateral_speed,
        initial_linear_speed=args.initial_linear_speed,
    )
    operator_ui = OperatorWebUI(
        velocity_command,
        port=args.operator_ui_port,
        mode_switch_enabled=args.hardware_console is not None,
        input_source=(
            "UNITREE HAND REMOTE"
            if args.remote_control
            else (
                "KEYBOARD (WEB + MUJOCO)"
                if args.sim_keyboard_port is not None
                else "WEB KEYBOARD"
            )
        ),
        native_keyboard_port=args.sim_keyboard_port,
        event_log_path=args.operator_event_log,
    )
    timed_test = TimedTestRunner(args.test_results)
    runtime_metrics = RuntimeMetricsSampler(args.hardware_console)
    hardware_manager: HardwareModeManager | None = None
    gateway_status = GatewayStatus("SIMULATION", "仿真/外部 gateway 模式")
    active_input_source = "remote" if args.remote_control else "keyboard"
    if args.hardware_console:
        policy_sha256 = hashlib.sha256(bundle.policy_path.read_bytes()).hexdigest()
        hardware_manager = HardwareModeManager(
            interface=args.hardware_console,
            gateway_executable=args.gateway_executable,
            mode_probe_executable=args.mode_probe_executable,
            sdk_library_path=args.sdk_library_path,
            diagnostics_csv=args.managed_gateway_csv,
            gateway_log=args.managed_gateway_log,
            gateway_pid=args.managed_gateway_pid,
            hardware_profile=args.hardware_profile,
            policy_sha256=policy_sha256,
        )
        gateway_status = hardware_manager.check_native()
        activation_blockers = hardware_manager.activation_blockers()
        operator_ui.update(
            control_mode=gateway_status.control_mode,
            lifecycle_state=gateway_status.lifecycle_state,
            output_mode=gateway_status.output_mode,
            mode_detail=(
                gateway_status.detail
                if not activation_blockers
                else gateway_status.detail
                + "；自研接管锁定："
                + "; ".join(activation_blockers)
            ),
            motion_commands_enabled=False,
        )
    remote_mapper = RemoteVelocityMapper(
        max_forward_speed=args.max_linear_speed,
        max_lateral_speed=args.max_lateral_speed,
        max_yaw_rate=args.max_yaw_rate,
        deadzone=args.remote_deadzone,
    )
    remote_mode_controller = RemoteModeController()
    remote_mode = RemoteMode.STAND
    previous_remote_mode = remote_mode
    interactive_control = (
        args.interactive_keyboard or args.operator_ui or args.remote_control
    )
    effective_warmup_hold_duration = (
        max(args.warmup_hold_duration, 5.0)
        if args.remote_control or args.hardware_console
        else args.warmup_hold_duration
    )
    diagnostics_file = None
    diagnostics_writer = None
    if args.diagnostics_csv:
        diagnostics_path = Path(args.diagnostics_csv)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_file = diagnostics_path.open("w", newline="", encoding="utf-8")
        diagnostics_writer = csv.writer(diagnostics_file)
        header = [
            "elapsed_s",
            "phase",
            "state_sequence",
            "command_vx",
            "command_vy",
            "command_wz",
        ]
        for prefix in ("raw", "applied", "target", "actual", "dq"):
            header.extend(f"{prefix}_{name.removesuffix('_joint')}" for name in JOINT_NAMES)
        header.extend(f"observation_{index}" for index in range(policy.input_dim))
        diagnostics_writer.writerow(header)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.bind(("127.0.0.1", STATE_PORT))
    receiver.settimeout(0.1 if hardware_manager is not None else 1.0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_heading: float | None = None
    run_started: float | None = None
    command_sequence = 0
    last_state_sequence = 0
    deadline: float | None = None
    warmup_deadline: float | None = None
    settle_deadline: float | None = None
    settling = False
    motion_started = False
    motion_just_started = False
    motion_action_min = np.full(12, np.inf, dtype=np.float64)
    motion_action_max = np.full(12, -np.inf, dtype=np.float64)
    motion_applied_min = np.full(12, np.inf, dtype=np.float64)
    motion_applied_max = np.full(12, -np.inf, dtype=np.float64)
    motion_clip_count = np.zeros(12, dtype=np.int64)
    motion_target_min = np.full(12, np.inf, dtype=np.float64)
    motion_target_max = np.full(12, -np.inf, dtype=np.float64)
    motion_samples = 0
    samples = 0
    max_inference_ms = 0.0
    final_q: np.ndarray | None = None
    final_action: np.ndarray | None = None
    final_q_desired: np.ndarray | None = None
    final_command: np.ndarray | None = None
    pending_native = False
    stationary_since: float | None = None
    last_gateway_poll = 0.0
    emergency_stop_seen = False
    emergency_stop_reported = False
    safety_fault_seen = False
    pending_test_request: str | None = None
    state_packet_gaps = 0
    runtime_snapshot: dict[str, float | int] = runtime_metrics.sample()
    last_runtime_sample = 0.0
    last_test_result: dict[str, object] | None = None

    print(
        "Waiting for B2 gateway state on 127.0.0.1:15000 ... "
        "target_slew="
        + (
            "disabled"
            if args.disable_target_slew
            else f"{args.target_slew_rad_s:.1f} rad/s"
        )
        + (
            " action_clip=disabled"
            if args.action_clip_abs is None
            else f" action_clip=[-{args.action_clip_abs:g},+{args.action_clip_abs:g}]"
        )
    )
    try:
        if args.interactive_keyboard:
            keyboard.open()
            print(
                "Terminal keyboard: 1..9 select W/S speed 0.1..0.9 m/s, "
                "W/S set selected vx, A/D set fixed vy, Q/E step wz, "
                "Space/Z/R/X/Esc ZERO/HOLD, Ctrl-C exit; "
                "use --operator-ui for release-to-stop controls"
            )
        if args.operator_ui:
            ui_url = operator_ui.open(open_browser=not args.no_open_browser)
            print(f"B2 operator UI: {ui_url}")
            if args.sim_keyboard_port is not None:
                print(
                    "MuJoCo window keyboard: "
                    f"127.0.0.1:{operator_ui.native_keyboard_port}"
                )
            if hardware_manager is not None:
                print(
                    "Hardware console starts in verified native mode; "
                    "use the browser button to request custom control"
                )
        if args.remote_control:
            print(
                "Hand remote: original Unitree mapping retained; "
                "A=POLICY, R2=STAND/HOLD, L2+B=latched E-STOP; "
                "left stick ly/lx=vx/-vy, right stick rx=-wz"
            )
        while True:
            loop_now = time.monotonic()
            if args.operator_ui:
                if operator_ui.consume_test_cancel():
                    timed_test.abort(now=loop_now, reason="operator cancelled")
                    pending_test_request = None
                    velocity_command.zero()
                    operator_ui.update(
                        phase="TEST CANCELLED · ZERO/HOLD",
                        lifecycle_state=ControlLifecycle.HOLD,
                    )
                requested_test = operator_ui.consume_test_request()
                if requested_test is not None:
                    pending_test_request = requested_test
            if hardware_manager is not None:
                source_request = operator_ui.consume_input_source_request()
                if source_request is not None:
                    velocity_command.zero()
                    if gateway_status.control_mode != "NATIVE":
                        operator_ui.update(
                            mode_detail=(
                                "命令源切换被拒绝：请先切回原生 locomotion"
                            ),
                            motion_commands_enabled=False,
                            lifecycle_state=gateway_status.lifecycle_state,
                            output_mode=gateway_status.output_mode,
                        )
                    else:
                        active_input_source = source_request
                        remote_mode_controller = RemoteModeController()
                        remote_mode = RemoteMode.STAND
                        previous_remote_mode = remote_mode
                        operator_ui.update(
                            input_source=(
                                "UNITREE HAND REMOTE"
                                if active_input_source == "remote"
                                else "WEB KEYBOARD"
                            ),
                            mode_detail=(
                                "命令源已切换；仍处于原生 locomotion，"
                                "下次自研接管从零命令开始"
                            ),
                            motion_commands_enabled=False,
                            lifecycle_state=ControlLifecycle.NATIVE_CONFIRMED,
                            output_mode="NATIVE",
                        )
                mode_request = operator_ui.consume_mode_request()
                if mode_request == "CUSTOM":
                    if gateway_status.control_mode != "NATIVE":
                        gateway_status = (
                            hardware_manager.recheck_native_after_stopped_gateway()
                        )
                    if gateway_status.control_mode != "NATIVE":
                        operator_ui.update(
                            mode_transition=False,
                            motion_commands_enabled=False,
                            lifecycle_state=gateway_status.lifecycle_state,
                            output_mode=gateway_status.output_mode,
                            mode_detail=(
                                "拒绝自研接管：必须先确认原生 ai，当前为 "
                                + gateway_status.control_mode
                            ),
                        )
                    else:
                        velocity_command.zero()
                        core.reset()
                        target_heading = None
                        run_started = None
                        last_state_sequence = 0
                        warmup_deadline = None
                        motion_started = False
                        settling = False
                        gateway_status = hardware_manager.start_custom()
                        operator_ui.update(
                            phase="CUSTOM STARTING",
                            gateway_connected=False,
                            control_mode=gateway_status.control_mode,
                            mode_detail=gateway_status.detail,
                            mode_transition=(
                                gateway_status.control_mode == "CUSTOM_STARTING"
                            ),
                            motion_commands_enabled=False,
                            lifecycle_state=gateway_status.lifecycle_state,
                            output_mode=gateway_status.output_mode,
                        )
                elif mode_request == "NATIVE":
                    velocity_command.zero()
                    if gateway_status.control_mode == "NATIVE":
                        operator_ui.update(
                            phase="NATIVE ACTIVE",
                            control_mode="NATIVE",
                            mode_detail="已经处于原生 locomotion",
                            mode_transition=False,
                            motion_commands_enabled=False,
                            lifecycle_state=ControlLifecycle.NATIVE,
                            output_mode="NATIVE",
                        )
                    elif (
                        gateway_status.control_mode == "CUSTOM_STARTING"
                        and not gateway_status.gateway_mode
                    ):
                        try:
                            gateway_status = hardware_manager.stop_to_native()
                            operator_ui.update(
                                phase="NATIVE ACTIVE",
                                gateway_connected=False,
                                control_mode=gateway_status.control_mode,
                                mode_detail=gateway_status.detail,
                                mode_transition=False,
                                motion_commands_enabled=False,
                                lifecycle_state=ControlLifecycle.NATIVE,
                                output_mode="NATIVE",
                            )
                        except RuntimeError as error:
                            operator_ui.update(
                                control_mode="FAULT",
                                mode_detail=str(error),
                                mode_transition=False,
                                motion_commands_enabled=False,
                                lifecycle_state=ControlLifecycle.FAULT,
                                output_mode="UNKNOWN",
                            )
                    elif gateway_status.control_mode == "CUSTOM":
                        pending_native = True
                        stationary_since = None
                        operator_ui.update(
                            phase="SETTLING FOR NATIVE",
                            mode_detail="速度已归零，等待关节静止 0.5 秒",
                            mode_transition=True,
                            motion_commands_enabled=False,
                            lifecycle_state=ControlLifecycle.STOPPING,
                            output_mode="HOLD",
                        )
                    else:
                        operator_ui.update(
                            mode_detail=(
                                "拒绝自动切换：当前状态 "
                                + gateway_status.control_mode
                            ),
                            mode_transition=False,
                            motion_commands_enabled=False,
                        )

                if loop_now - last_gateway_poll >= 0.2:
                    gateway_status = hardware_manager.poll()
                    last_gateway_poll = loop_now
                    if not pending_native:
                        custom_ready = (
                            gateway_status.control_mode == "CUSTOM"
                            and gateway_status.gateway_mode == "POLICY"
                            and motion_started
                            and (
                                warmup_deadline is None
                                or loop_now >= warmup_deadline
                            )
                        )
                        operator_ui.update(
                            control_mode=gateway_status.control_mode,
                            mode_detail=gateway_status.detail,
                            mode_transition=(
                                gateway_status.control_mode == "CUSTOM_STARTING"
                            ),
                            motion_commands_enabled=custom_ready,
                            lifecycle_state=gateway_status.lifecycle_state,
                            output_mode=gateway_status.output_mode,
                            fault=gateway_status.fault,
                        )
            try:
                packet, _ = receiver.recvfrom(STATE_STRUCT.size)
            except socket.timeout:
                if args.operator_ui:
                    if hardware_manager is None:
                        operator_ui.update(
                            phase="WAITING LOWSTATE",
                            gateway_connected=False,
                        )
                    elif gateway_status.control_mode == "NATIVE":
                        operator_ui.update(
                            phase="NATIVE ACTIVE",
                            gateway_connected=False,
                            motion_commands_enabled=False,
                            lifecycle_state=ControlLifecycle.NATIVE,
                            output_mode="NATIVE",
                        )
                    else:
                        operator_ui.update(
                            phase="WAITING LOWSTATE",
                            gateway_connected=False,
                            motion_commands_enabled=False,
                            lifecycle_state=gateway_status.lifecycle_state,
                            output_mode=gateway_status.output_mode,
                        )
                continue
            state = decode_state(packet)
            if state.sequence <= last_state_sequence:
                continue
            if last_state_sequence and state.sequence > last_state_sequence + 1:
                state_packet_gaps += state.sequence - last_state_sequence - 1
            last_state_sequence = state.sequence
            emergency_stop_seen = (
                emergency_stop_seen or state.emergency_stop_latched
            )
            safety_fault_seen = safety_fault_seen or state.safety_fault_latched
            if state.safety_fault_latched or state.emergency_stop_latched:
                velocity_command.zero()
                pending_test_request = None
                timed_test.abort(
                    now=time.monotonic(), reason="gateway safety fault"
                )
            complete_native_after_command = False
            if pending_native:
                velocity_command.zero()
                if state.emergency_stop_latched or state.safety_fault_latched:
                    pending_native = False
                    stationary_since = None
                    operator_ui.update(
                        control_mode="FAULT",
                        mode_detail=(
                            "急停/安全故障已锁存，禁止自动恢复 ai；需要现场确认"
                        ),
                        mode_transition=False,
                        motion_commands_enabled=False,
                        lifecycle_state=ControlLifecycle.FAULT,
                        output_mode="DAMPING",
                    )
                else:
                    max_abs_dq = float(np.max(np.abs(state.dq)))
                    if max_abs_dq <= 0.20:
                        if stationary_since is None:
                            stationary_since = time.monotonic()
                        complete_native_after_command = (
                            time.monotonic() - stationary_since >= 0.5
                        )
                    else:
                        stationary_since = None
                        operator_ui.update(
                            mode_detail=(
                                "等待静止：max |dq|="
                                f"{max_abs_dq:.3f} rad/s（要求 ≤0.20）"
                            )
                        )
            rotation = quaternion_wxyz_to_rotation(state.quaternion_wxyz)
            current_heading = yaw_from_rotation(rotation)
            if target_heading is None:
                target_heading = current_heading + args.heading_offset
                now = time.monotonic()
                run_started = now
                if args.hold:
                    if args.duration is not None:
                        deadline = now + args.duration
                    print(
                        "Policy active at 50 Hz: learned-hold, "
                        f"heading={target_heading:.3f} rad"
                    )
                elif effective_warmup_hold_duration > 0.0:
                    warmup_deadline = now + effective_warmup_hold_duration
                    print(
                        "Policy active at 50 Hz: learned-hold warmup "
                        f"for {effective_warmup_hold_duration:.1f} s, "
                        f"heading={target_heading:.3f} rad"
                    )
                else:
                    motion_started = True
                    if args.duration is not None:
                        deadline = now + args.duration
                    if interactive_control:
                        print("Policy active at 50 Hz: operator command=[0,0,0]")
                    else:
                        print(
                            f"Policy active at 50 Hz: speed={args.speed:.2f} m/s, "
                            f"heading={target_heading:.3f} rad"
                        )
            started = time.perf_counter_ns()
            now = time.monotonic()
            warming_up = (
                not args.hold
                and warmup_deadline is not None
                and now < warmup_deadline
            )
            motion_just_started = False
            if (
                not args.hold
                and not motion_started
                and warmup_deadline is not None
                and now >= warmup_deadline
            ):
                motion_started = True
                motion_just_started = True
                if args.duration is not None:
                    deadline = now + args.duration
                if interactive_control:
                    print("Warmup complete; operator control enabled at zero command")
                else:
                    print(f"Warmup complete; commanding speed={args.speed:.2f} m/s")
            if (
                deadline is not None
                and not args.hold
                and not settling
                and now >= deadline
            ):
                settling = True
                if args.keep_hold_after_duration:
                    settle_deadline = None
                    print(
                        "Command duration complete; continuing learned-hold "
                        "until interrupted"
                    )
                else:
                    settle_deadline = now + 2.0
                    print(
                        "Command duration complete; settling with "
                        "learned-hold for 2.0 s"
                    )
            test_ready = (
                not warming_up
                and not settling
                and not pending_native
                and active_input_source == "keyboard"
                and (
                    hardware_manager is None
                    or (
                        gateway_status.control_mode == "CUSTOM"
                        and gateway_status.gateway_mode == "POLICY"
                    )
                )
            )
            if pending_test_request is not None and test_ready:
                test_command = timed_test.start(
                    pending_test_request,
                    now=now,
                    selected_speed_m_s=velocity_command.selected_linear_speed,
                    initial_lost=int(np.sum(state.motor_lost)),
                )
                velocity_command.set_velocity(*test_command.tolist())
                operator_ui.update(
                    phase=f"TEST {pending_test_request}",
                    lifecycle_state=ControlLifecycle.HOLD,
                )
                pending_test_request = None
            if interactive_control:
                if warming_up or settling:
                    velocity_command.zero()
                elif args.interactive_keyboard:
                    characters = keyboard.poll()
                    for character in characters:
                        event = velocity_command.apply_key(character)
                        if event is not None:
                            label, command = event
                            print(
                                f"keyboard {label}: command="
                                + np.array2string(
                                    command, precision=3, separator=","
                                )
                            )
            if timed_test.active is not None:
                velocity_command.set_velocity(*timed_test.command().tolist())
                test_fault = (
                    gateway_status.detail
                    if state.safety_fault_latched or state.emergency_stop_latched
                    else ""
                )
                completed_test = timed_test.finish_if_due(
                    now=now, fault=test_fault
                )
                if completed_test is not None:
                    velocity_command.zero()
                    last_test_result = completed_test
                    operator_ui.update(
                        phase=(
                            f"TEST {completed_test['verdict']} · ZERO/HOLD"
                        ),
                        lifecycle_state=(
                            ControlLifecycle.FAULT
                            if completed_test["verdict"] == "FAIL"
                            else ControlLifecycle.HOLD
                        ),
                    )
            remote_emergency_stop = state.emergency_stop_latched
            remote_velocity: np.ndarray | None = None
            if active_input_source == "remote":
                remote_velocity = np.zeros(3, dtype=np.float32)
                if state.remote_valid and not (warming_up or settling):
                    remote_mode = remote_mode_controller.update(
                        state.remote_keys,
                        sticks_centered=remote_mapper.sticks_centered(
                            state.remote_axes
                        ),
                    )
                    if remote_mode is RemoteMode.POLICY and not (
                        warming_up or settling
                    ):
                        remote_velocity = remote_mapper.map_axes(
                            state.remote_axes
                        )
                    if remote_mode is RemoteMode.EMERGENCY_STOP:
                        remote_emergency_stop = True
                    if remote_mode is not previous_remote_mode:
                        print(f"Hand remote mode: {remote_mode.value.upper()}")
                        previous_remote_mode = remote_mode
                if remote_emergency_stop:
                    remote_mode = RemoteMode.EMERGENCY_STOP
                    remote_velocity.fill(0.0)
            output = core.step(
                omega_body_rad_s=state.gyro,
                rotation_body_to_world=rotation,
                joint_position_rad=state.q,
                joint_velocity_rad_s=state.dq,
                target_yaw_rad=target_heading,
                current_yaw_rad=current_heading,
                nominal_speed_m_s=args.speed,
                hold=(
                    args.hold
                    or warming_up
                    or settling
                    or (
                        active_input_source == "remote"
                        and remote_mode is not RemoteMode.POLICY
                    )
                ),
                velocity_command=(
                    remote_velocity
                    if active_input_source == "remote"
                    else (
                        velocity_command.snapshot()
                        if interactive_control
                        else None
                    )
                ),
            )
            if motion_just_started:
                print(
                    "sim_motion_start_command="
                    + np.array2string(output.command, precision=4, separator=",")
                )
                print(
                    "sim_motion_start_action="
                    + np.array2string(output.raw_action, precision=4, separator=",")
                )
                print(
                    "sim_motion_start_applied_action="
                    + np.array2string(
                        output.applied_action, precision=4, separator=","
                    )
                )
                print(
                    "sim_motion_start_q_desired_rad="
                    + np.array2string(output.q_desired_rad, precision=4, separator=",")
                )
            if motion_started and not settling and not warming_up:
                motion_action_min = np.minimum(motion_action_min, output.raw_action)
                motion_action_max = np.maximum(motion_action_max, output.raw_action)
                motion_applied_min = np.minimum(
                    motion_applied_min, output.applied_action
                )
                motion_applied_max = np.maximum(
                    motion_applied_max, output.applied_action
                )
                if args.action_clip_abs is not None:
                    motion_clip_count += (
                        np.abs(output.raw_action) > args.action_clip_abs
                    )
                motion_target_min = np.minimum(motion_target_min, output.q_desired_rad)
                motion_target_max = np.maximum(motion_target_max, output.q_desired_rad)
                motion_samples += 1
            inference_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            max_inference_ms = max(max_inference_ms, inference_ms)
            tracking_error_rad = float(
                np.max(np.abs(output.q_desired_rad - state.q))
            )
            maximum_temperature_c = int(np.max(state.motor_temperature_c))
            total_motor_lost = int(np.sum(state.motor_lost))
            timed_test.observe(
                tracking_error_rad=tracking_error_rad,
                temperature_c=maximum_temperature_c,
                lost=total_motor_lost,
            )
            if now - last_runtime_sample >= 0.5:
                runtime_snapshot = runtime_metrics.sample()
                last_runtime_sample = now
            if args.operator_ui:
                positive_cells = state.battery_cell_mv[
                    state.battery_cell_mv > 0
                ]
                if pending_native:
                    ui_phase = "SETTLING FOR NATIVE"
                elif args.hold:
                    ui_phase = "LEARNED HOLD"
                elif warming_up:
                    ui_phase = "WARMUP HOLD"
                elif settling:
                    ui_phase = "SETTLING HOLD"
                elif np.allclose(output.command, 0.0):
                    ui_phase = "ZERO / HOLD"
                else:
                    ui_phase = "POLICY ACTIVE"
                command_is_zero = bool(np.allclose(output.command, 0.0))
                packet_gateway_mode = GATEWAY_MODE_NAMES.get(
                    state.gateway_mode_code, "UNKNOWN"
                )
                effective_gateway_mode = (
                    gateway_status.gateway_mode
                    if hardware_manager is not None and gateway_status.gateway_mode
                    else packet_gateway_mode
                )
                if state.safety_fault_latched or state.emergency_stop_latched:
                    lifecycle_state = ControlLifecycle.FAULT
                    output_mode = "DAMPING"
                elif pending_native:
                    lifecycle_state = ControlLifecycle.STOPPING
                    output_mode = "HOLD"
                elif hardware_manager is not None and gateway_status.control_mode == "NATIVE":
                    lifecycle_state = ControlLifecycle.NATIVE
                    output_mode = "NATIVE"
                else:
                    lifecycle_state = lifecycle_for_gateway(
                        effective_gateway_mode,
                        command_is_zero=command_is_zero,
                    )
                    output_mode = "HOLD" if command_is_zero else "LOWCMD"
                policy_e2e_ms = max(
                    0.0, (time.monotonic_ns() - state.monotonic_ns) / 1_000_000.0
                )
                packet_total = max(1, state.sequence + state_packet_gaps)
                operator_ui.update(
                    phase=ui_phase,
                    lifecycle_state=lifecycle_state,
                    output_mode=output_mode,
                    gateway_connected=True,
                    state_sequence=state.sequence,
                    inference_ms=inference_ms,
                    tracking_error_rad=tracking_error_rad,
                    fault=(
                        gateway_status.fault
                        if hardware_manager is not None
                        else {
                            "reason": (
                                "gateway safety fault"
                                if state.safety_fault_latched
                                else ""
                            )
                        }
                    ),
                    display_command=(
                        remote_velocity
                        if active_input_source == "remote"
                        else None
                    ),
                    telemetry={
                        "safety_fault_latched": state.safety_fault_latched,
                        "emergency_stop_latched": state.emergency_stop_latched,
                        "battery_soc_percent": state.battery_soc_percent,
                        "battery_current_raw": state.battery_current_raw,
                        "battery_max_temperature_c": int(
                            np.max(state.battery_temperature_c)
                        ),
                        "battery_min_cell_mv": (
                            int(np.min(positive_cells))
                            if positive_cells.size
                            else 0
                        ),
                        "battery_max_cell_mv": int(
                            np.max(state.battery_cell_mv)
                        ),
                        "max_abs_tau_est": float(np.max(np.abs(state.tau_est))),
                        "max_abs_dq": float(np.max(np.abs(state.dq))),
                        "max_motor_temperature_c": maximum_temperature_c,
                        "motor_lost_sum": total_motor_lost,
                        "crc_errors": state.state_crc_errors,
                        "invalid_states": state.invalid_states,
                        "dds_hz": state.dds_lowstate_hz,
                        "control_jitter_ms": state.publish_jitter_ms,
                        "control_jitter_max_ms": state.publish_jitter_max_ms,
                        "lowstate_age_ms": state.lowstate_age_ms,
                        "policy_command_age_ms": state.policy_command_age_ms,
                        "policy_e2e_ms": policy_e2e_ms,
                        "cpu_percent": runtime_snapshot["policy_cpu_percent"],
                        "network_packet_loss_percent": runtime_snapshot[
                            "network_packet_loss_percent"
                        ],
                        "network_rx_dropped": runtime_snapshot[
                            "network_rx_dropped"
                        ],
                        "network_tx_dropped": runtime_snapshot[
                            "network_tx_dropped"
                        ],
                        "policy_udp_gaps": state_packet_gaps,
                        "policy_udp_loss_percent": 100.0
                        * state_packet_gaps
                        / packet_total,
                        "gateway_mode": effective_gateway_mode,
                        "blackbox_complete": (
                            gateway_status.blackbox_complete
                            if hardware_manager is not None
                            else False
                        ),
                        "last_test_result": last_test_result or {},
                        "joints": [
                            {
                                "name": name.removesuffix("_joint"),
                                "target": float(output.q_desired_rad[index]),
                                "actual": float(state.q[index]),
                                "dq": float(state.dq[index]),
                                "tau": float(state.tau_est[index]),
                                "temperature": int(
                                    state.motor_temperature_c[index]
                                ),
                                "lost": int(state.motor_lost[index]),
                            }
                            for index, name in enumerate(JOINT_NAMES)
                        ],
                    },
                    motion_commands_enabled=(
                        active_input_source != "remote"
                        and timed_test.active is None
                        and pending_test_request is None
                        and not state.safety_fault_latched
                        and not state.emergency_stop_latched
                        and (
                            hardware_manager is None
                            or (
                                gateway_status.control_mode == "CUSTOM"
                                and gateway_status.gateway_mode == "POLICY"
                                and not warming_up
                                and not pending_native
                            )
                        )
                    ),
                )
            samples += 1
            final_q = state.q.copy()
            final_action = output.raw_action.copy()
            final_q_desired = output.q_desired_rad.copy()
            final_command = output.command.copy()
            if diagnostics_writer is not None:
                assert run_started is not None
                if args.hold:
                    phase = "hold"
                elif warming_up:
                    phase = "warmup"
                elif settling:
                    phase = "settle"
                else:
                    phase = "motion"
                diagnostics_writer.writerow(
                    [
                        f"{now - run_started:.9f}",
                        phase,
                        state.sequence,
                        *output.command.tolist(),
                        *output.raw_action.tolist(),
                        *output.applied_action.tolist(),
                        *output.q_desired_rad.tolist(),
                        *state.q.tolist(),
                        *state.dq.tolist(),
                        *output.observation.reshape(-1).tolist(),
                    ]
                )
            if samples == 1:
                print(
                    "sim_first_q_rad="
                    + np.array2string(state.q, precision=4, separator=",")
                )
                print(
                    "sim_first_command="
                    + np.array2string(output.command, precision=4, separator=",")
                )
                print(
                    "sim_first_action="
                    + np.array2string(output.raw_action, precision=4, separator=",")
                )
                print(
                    "sim_first_q_desired_rad="
                    + np.array2string(output.q_desired_rad, precision=4, separator=",")
                )
            command_sequence += 1
            sender.sendto(
                encode_command(
                    sequence=command_sequence,
                    monotonic_ns=time.monotonic_ns(),
                    q_desired=output.q_desired_rad,
                    emergency_stop=remote_emergency_stop,
                    remote_required=active_input_source == "remote",
                ),
                ("127.0.0.1", COMMAND_PORT),
            )
            if complete_native_after_command and hardware_manager is not None:
                try:
                    gateway_status = hardware_manager.stop_to_native()
                    pending_native = False
                    stationary_since = None
                    velocity_command.zero()
                    core.reset()
                    target_heading = None
                    last_state_sequence = 0
                    warmup_deadline = None
                    motion_started = False
                    operator_ui.update(
                        phase="NATIVE ACTIVE",
                        gateway_connected=False,
                        control_mode=gateway_status.control_mode,
                        mode_detail=gateway_status.detail,
                        mode_transition=False,
                        motion_commands_enabled=False,
                        lifecycle_state=ControlLifecycle.NATIVE,
                        output_mode="NATIVE",
                    )
                    continue
                except RuntimeError as error:
                    pending_native = False
                    stationary_since = None
                    operator_ui.update(
                        control_mode="FAULT",
                        mode_detail=str(error),
                        mode_transition=False,
                        motion_commands_enabled=False,
                        lifecycle_state=ControlLifecycle.FAULT,
                        output_mode="UNKNOWN",
                    )
            if remote_emergency_stop:
                velocity_command.zero()
                if not emergency_stop_reported:
                    print(
                        "EMERGENCY STOP LATCHED: command forced to zero; "
                        "gateway remains in SAFE_DAMPING while the 30 s "
                        "post-fault black-box window is captured"
                    )
                    emergency_stop_reported = True
            if args.hold and deadline is not None and now >= deadline:
                break
            if settling and settle_deadline is not None and now >= settle_deadline:
                break
    except KeyboardInterrupt:
        print("Policy console stopped.")
    finally:
        velocity_command.zero()
        timed_test.abort(now=time.monotonic(), reason="policy console stopped")
        keyboard.close()
        if hardware_manager is not None:
            if emergency_stop_seen or safety_fault_seen:
                print(
                    "Managed gateway left running because a safety fault is latched; "
                    "native ai restore requires an explicit onsite recovery"
                )
            else:
                try:
                    native = hardware_manager.close()
                    print(native.detail)
                except RuntimeError as error:
                    print(f"Failed to restore native locomotion: {error}")
        operator_ui.close()
        receiver.close()
        sender.close()
        if diagnostics_file is not None:
            diagnostics_file.close()
    if samples:
        assert final_q is not None
        assert final_action is not None
        assert final_q_desired is not None
        assert final_command is not None
        print("sim_final_q_rad=" + np.array2string(final_q, precision=4, separator=","))
        print(
            "sim_final_command="
            + np.array2string(final_command, precision=4, separator=",")
        )
        print(
            "sim_final_action="
            + np.array2string(final_action, precision=4, separator=",")
        )
        print(
            "sim_final_q_desired_rad="
            + np.array2string(final_q_desired, precision=4, separator=",")
        )
        if motion_samples:
            print(f"sim_motion_samples={motion_samples}")
            print(
                "sim_motion_action_peak_to_peak="
                + np.array2string(
                    motion_action_max - motion_action_min,
                    precision=4,
                    separator=",",
                )
            )
            print(
                "sim_motion_action_min="
                + np.array2string(
                    motion_action_min, precision=4, separator=","
                )
            )
            print(
                "sim_motion_action_max="
                + np.array2string(
                    motion_action_max, precision=4, separator=","
                )
            )
            print(
                "sim_motion_applied_peak_to_peak="
                + np.array2string(
                    motion_applied_max - motion_applied_min,
                    precision=4,
                    separator=",",
                )
            )
            if args.action_clip_abs is not None:
                print(
                    "sim_motion_action_clip_count="
                    + np.array2string(
                        motion_clip_count, separator=","
                    )
                )
            print(
                "sim_motion_target_peak_to_peak_rad="
                + np.array2string(
                    motion_target_max - motion_target_min,
                    precision=4,
                    separator=",",
                )
            )
        print(
            f"SIM_PASS samples={samples} max_inference_ms={max_inference_ms:.3f}"
        )
    if hardware_manager is not None and (emergency_stop_seen or safety_fault_seen):
        return 2
    return 0


def gui_main() -> int:
    return main([*sys.argv[1:], "--operator-ui"])


if __name__ == "__main__":
    raise SystemExit(main())
