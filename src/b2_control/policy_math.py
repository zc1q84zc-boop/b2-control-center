from __future__ import annotations

from typing import Any

import numpy as np


JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
Q_DEFAULT = np.array(
    (
        -0.1,
        0.8,
        -1.5,
        0.1,
        0.8,
        -1.5,
        -0.1,
        1.0,
        -1.5,
        0.1,
        1.0,
        -1.5,
    ),
    dtype=np.float64,
)
ACTION_SCALE = np.array((0.125, 0.25, 0.25) * 4, dtype=np.float64)
Q_DEFAULT.setflags(write=False)
ACTION_SCALE.setflags(write=False)
OBSERVATION_DIM = 45
ACTION_DIM = 12


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric vector") from exc
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _scalar(value: Any, name: str) -> float:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def build_observation(
    *,
    omega_body: Any,
    rotation_body_to_world: Any,
    command: Any,
    q: Any,
    dq: Any,
    previous_action: Any,
) -> np.ndarray:
    omega = _vector(omega_body, 3, "omega_body")
    cmd = _vector(command, 3, "command")
    joint_position = _vector(q, ACTION_DIM, "q")
    joint_velocity = _vector(dq, ACTION_DIM, "dq")
    last_action = _vector(previous_action, ACTION_DIM, "previous_action")

    try:
        rotation = np.asarray(rotation_body_to_world, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "rotation_body_to_world must be a finite numeric matrix"
        ) from exc
    if rotation.shape != (3, 3):
        raise ValueError(
            f"rotation_body_to_world must have shape (3, 3), got {rotation.shape}"
        )
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_body_to_world must contain only finite values")
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError("rotation_body_to_world must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("rotation_body_to_world must have determinant +1")

    projected_gravity = rotation.T @ np.array((0.0, 0.0, -1.0))
    raw_observation = np.concatenate(
        (
            0.25 * omega,
            projected_gravity,
            cmd,
            joint_position - Q_DEFAULT,
            0.05 * joint_velocity,
            last_action,
        )
    )
    with np.errstate(over="ignore", invalid="ignore"):
        observation = raw_observation.astype(np.float32, copy=False)
    if not np.all(np.isfinite(observation)):
        raise ValueError("observation cannot be represented as finite float32")
    return np.ascontiguousarray(observation.reshape(1, OBSERVATION_DIM))


def heading_first_command(
    target_heading: float,
    current_heading: float,
    nominal_speed: float,
    *,
    hold: bool = False,
) -> tuple[np.ndarray, float, float]:
    target = _scalar(target_heading, "target_heading")
    current = _scalar(current_heading, "current_heading")
    speed = _scalar(nominal_speed, "nominal_speed")
    if speed < 0.0:
        raise ValueError("nominal_speed must be non-negative")

    full_turn = 2.0 * np.pi
    reduced_delta = target % full_turn - current % full_turn
    error = float((reduced_delta + np.pi) % full_turn - np.pi)
    gate = float(np.exp(-((error / 0.55) ** 2)))
    if hold:
        command = np.zeros(3, dtype=np.float32)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            command = np.array(
                (speed * gate, 0.0, np.clip(error, -1.0, 1.0)),
                dtype=np.float32,
            )
        if not np.all(np.isfinite(command)):
            raise ValueError("command cannot be represented as finite float32")
    return command, error, gate


def direct_velocity_command(value: Any) -> np.ndarray:
    """Validate a direct policy command ordered as ``[vx, vy, wz]``."""

    command = _vector(value, 3, "velocity_command")
    with np.errstate(over="ignore", invalid="ignore"):
        result = command.astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError("velocity_command cannot be represented as finite float32")
    return np.ascontiguousarray(result)


def joint_targets(action: Any) -> np.ndarray:
    return Q_DEFAULT + ACTION_SCALE * _vector(action, ACTION_DIM, "action")
