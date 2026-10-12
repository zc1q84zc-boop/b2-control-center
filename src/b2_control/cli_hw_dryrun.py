from __future__ import annotations

import argparse
import csv
from pathlib import Path
import socket
import time

import numpy as np

from .bundle import open_verified_bundle
from .core import B2PolicyCore
from .handover_math import model_joint_limit_margin_rad
from .onnx_policy import OnnxPolicy
from .policy_math import JOINT_NAMES
from .stack10_adapter import Stack10ObservationAdapter
from .sim_protocol import (
    STATE_PORT,
    STATE_STRUCT,
    decode_state,
    quaternion_wxyz_to_rotation,
    yaw_from_rotation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the B2 policy on real LowState without publishing commands"
    )
    parser.add_argument("bundle", help="verified policy bundle directory")
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument(
        "--hold", action="store_true", help="use the learned zero-command hold"
    )
    parser.add_argument(
        "--diagnostics-csv",
        help="write every 50 Hz state, policy output and 45-D observation",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.duration <= 300:
        raise SystemExit("--duration must be 1..300 seconds")
    if not 0.25 <= args.speed <= 1.0:
        raise SystemExit("--speed must remain inside the trained range 0.25..1.0")

    bundle = open_verified_bundle(args.bundle)
    observation_adapter = (
        Stack10ObservationAdapter() if bundle.adapter == "b2_stack10_v1" else None
    )
    policy = OnnxPolicy(
        bundle.policy_path,
        input_dim=bundle.input_dim,
        output_dim=bundle.output_dim,
    )
    policy.warmup()
    core = B2PolicyCore(
        policy, target_slew_rad_s=8.0, observation_adapter=observation_adapter
    )
    diagnostics_file = None
    diagnostics_writer = None
    if args.diagnostics_csv:
        diagnostics_path = Path(args.diagnostics_csv)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_file = diagnostics_path.open("w", newline="", encoding="utf-8")
        diagnostics_writer = csv.writer(diagnostics_file)
        header = ["elapsed_s", "state_sequence", "command_vx", "command_vy", "command_wz"]
        for prefix in ("raw", "applied", "target", "actual", "dq"):
            header.extend(
                f"{prefix}_{name.removesuffix('_joint')}" for name in JOINT_NAMES
            )
        header.extend(f"observation_{index}" for index in range(policy.input_dim))
        diagnostics_writer.writerow(header)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.bind(("127.0.0.1", STATE_PORT))
    receiver.settimeout(2.0)
    deadline = time.monotonic() + args.duration
    run_started = time.monotonic()
    target_heading: float | None = None
    samples = 0
    max_inference_ms = 0.0
    max_abs_action = 0.0
    max_target_delta_rad = 0.0
    min_model_limit_margin_rad = float("inf")
    mode = "hold" if args.hold else f"forward {args.speed:.2f} m/s"
    print(
        "READ-ONLY POLICY DRY-RUN: no LowCmd publisher exists in this process; "
        f"mode={mode}"
    )
    try:
        while time.monotonic() < deadline:
            try:
                packet, _ = receiver.recvfrom(STATE_STRUCT.size)
            except socket.timeout:
                continue
            state = decode_state(packet)
            rotation = quaternion_wxyz_to_rotation(state.quaternion_wxyz)
            current_heading = yaw_from_rotation(rotation)
            if target_heading is None:
                target_heading = current_heading
            started = time.perf_counter_ns()
            output = core.step(
                omega_body_rad_s=state.gyro,
                rotation_body_to_world=rotation,
                joint_position_rad=state.q,
                joint_velocity_rad_s=state.dq,
                target_yaw_rad=target_heading,
                current_yaw_rad=current_heading,
                nominal_speed_m_s=args.speed,
                hold=args.hold,
            )
            if diagnostics_writer is not None:
                diagnostics_writer.writerow(
                    [
                        f"{time.monotonic() - run_started:.9f}",
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
            if samples == 0:
                projected_gravity = rotation.T @ np.array((0.0, 0.0, -1.0))
                print(
                    "first_quaternion_wxyz="
                    + np.array2string(
                        state.quaternion_wxyz, precision=5, separator=","
                    )
                )
                print(
                    "first_gyro_rad_s="
                    + np.array2string(state.gyro, precision=5, separator=",")
                )
                print(
                    "first_projected_gravity="
                    + np.array2string(projected_gravity, precision=5, separator=",")
                )
                print(
                    "first_q_rad="
                    + np.array2string(state.q, precision=4, separator=",")
                )
                print(
                    "first_action="
                    + np.array2string(output.raw_action, precision=4, separator=",")
                )
                print(
                    "first_q_desired_rad="
                    + np.array2string(
                        output.q_desired_rad, precision=4, separator=","
                    )
                )
            inference_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            samples += 1
            max_inference_ms = max(max_inference_ms, inference_ms)
            max_abs_action = max(
                max_abs_action, float(np.max(np.abs(output.raw_action)))
            )
            max_target_delta_rad = max(
                max_target_delta_rad,
                float(np.max(np.abs(output.q_desired_rad - state.q))),
            )
            min_model_limit_margin_rad = min(
                min_model_limit_margin_rad,
                float(np.min(model_joint_limit_margin_rad(output.q_desired_rad))),
            )
            if samples % 50 == 0:
                print(
                    f"samples={samples} inference_ms={inference_ms:.3f} "
                    f"action_abs_max={max_abs_action:.3f} "
                    f"target_delta_max_rad={max_target_delta_rad:.3f} "
                    f"model_limit_margin_min_rad={min_model_limit_margin_rad:.3f}"
                )
    finally:
        receiver.close()
        if diagnostics_file is not None:
            diagnostics_file.close()

    if samples == 0:
        print("FAIL samples=0")
        return 1
    print(
        f"PASS samples={samples} max_inference_ms={max_inference_ms:.3f} "
        f"action_abs_max={max_abs_action:.3f} "
        f"target_delta_max_rad={max_target_delta_rad:.3f} "
        f"model_limit_margin_min_rad={min_model_limit_margin_rad:.3f}"
    )
    print("PASS means inference/data validity only; it does not approve motor enable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
