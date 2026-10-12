from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


KEY_R2 = 1 << 4
KEY_L2 = 1 << 5
KEY_A = 1 << 8
KEY_B = 1 << 9


def emergency_stop_pressed(keys: int) -> bool:
    return (int(keys) & (KEY_L2 | KEY_B)) == (KEY_L2 | KEY_B)


def apply_deadzone(value: float, deadzone: float) -> float:
    if not np.isfinite(value):
        raise ValueError("remote axis must be finite")
    bounded = float(np.clip(value, -1.0, 1.0))
    return 0.0 if abs(bounded) < deadzone else bounded


@dataclass(frozen=True)
class RemoteVelocityMapper:
    max_forward_speed: float = 0.90
    max_lateral_speed: float = 0.25
    max_yaw_rate: float = 0.40
    deadzone: float = 0.10

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.max_forward_speed,
                self.max_lateral_speed,
                self.max_yaw_rate,
                self.deadzone,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("remote limits and deadzone must be positive finite")
        if self.deadzone >= 1.0:
            raise ValueError("remote deadzone must be less than 1")

    def map_axes(self, axes: Any) -> np.ndarray:
        values = np.asarray(axes, dtype=np.float64)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("remote axes must be a finite [lx,ly,rx,ry] vector")
        lx, ly, rx, _ = (
            apply_deadzone(float(value), self.deadzone) for value in values
        )
        # Preserve Unitree's official B2 controller mapping.
        return np.asarray(
            (
                ly * self.max_forward_speed,
                -lx * self.max_lateral_speed,
                -rx * self.max_yaw_rate,
            ),
            dtype=np.float32,
        )

    def sticks_centered(self, axes: Any) -> bool:
        values = np.asarray(axes, dtype=np.float64)
        return bool(
            values.shape == (4,)
            and np.all(np.isfinite(values))
            and np.max(np.abs(values[[0, 1, 2]])) < self.deadzone
        )


class RemoteMode(Enum):
    STAND = "stand"
    POLICY = "policy"
    EMERGENCY_STOP = "emergency_stop"


class RemoteModeController:
    """Keeps Unitree's R2/A/L2+B state transitions and latches E-stop."""

    def __init__(self) -> None:
        self.mode = RemoteMode.STAND
        self._previous_keys = 0

    def update(self, keys: int, *, sticks_centered: bool) -> RemoteMode:
        keys = int(keys) & 0xFFFF
        if self.mode is RemoteMode.EMERGENCY_STOP:
            self._previous_keys = keys
            return self.mode
        if emergency_stop_pressed(keys):
            self.mode = RemoteMode.EMERGENCY_STOP
        else:
            rising = keys & ~self._previous_keys
            if rising & KEY_R2:
                self.mode = RemoteMode.STAND
            elif rising & KEY_A and sticks_centered:
                self.mode = RemoteMode.POLICY
        self._previous_keys = keys
        return self.mode
