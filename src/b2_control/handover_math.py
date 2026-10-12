from __future__ import annotations

from typing import Any

import numpy as np

from .policy_math import ACTION_DIM


MODEL_JOINT_MIN_RAD = np.array(
    (-0.87, -0.94, -2.82) * 4, dtype=np.float64
)
MODEL_JOINT_MAX_RAD = np.array(
    (0.87, 4.69, -0.43) * 4, dtype=np.float64
)
MODEL_JOINT_MIN_RAD.setflags(write=False)
MODEL_JOINT_MAX_RAD.setflags(write=False)


def _finite_vector(value: Any, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite 12-vector") from exc
    if vector.shape != (ACTION_DIM,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 12-vector")
    return vector


def validate_model_joint_range(target_rad: Any) -> np.ndarray:
    target = _finite_vector(target_rad, "target_rad")
    violations = np.flatnonzero(
        (target < MODEL_JOINT_MIN_RAD) | (target > MODEL_JOINT_MAX_RAD)
    )
    if violations.size:
        index = int(violations[0])
        raise ValueError(
            f"target_rad[{index}]={target[index]} outside model range "
            f"[{MODEL_JOINT_MIN_RAD[index]}, {MODEL_JOINT_MAX_RAD[index]}]"
        )
    return target.copy()


def model_joint_limit_margin_rad(target_rad: Any) -> np.ndarray:
    target = validate_model_joint_range(target_rad)
    return np.minimum(
        target - MODEL_JOINT_MIN_RAD, MODEL_JOINT_MAX_RAD - target
    )


def slew_limited_step(
    *, previous_target_rad: Any, requested_target_rad: Any,
    max_rate_rad_s: Any, dt_s: float
) -> np.ndarray:
    previous = validate_model_joint_range(previous_target_rad)
    requested = validate_model_joint_range(requested_target_rad)
    rates = _finite_vector(max_rate_rad_s, "max_rate_rad_s")
    if np.any(rates <= 0.0):
        raise ValueError("max_rate_rad_s values must be positive")
    try:
        dt = float(dt_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt_s must be a finite positive scalar") from exc
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be a finite positive scalar")
    maximum_step = rates * dt
    next_target = previous + np.clip(
        requested - previous, -maximum_step, maximum_step
    )
    return validate_model_joint_range(next_target)
