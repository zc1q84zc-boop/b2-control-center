"""Terminal keyboard velocity commands shared by simulation and hardware runs.

The velocity-state semantics are adapted from StevenLiudw/B2_sim2real.  This
module only produces the policy command vector ``[vx, vy, wz]``; it never
publishes DDS or motor commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import select
import sys
import termios
import threading
import tty
from typing import Any

import numpy as np


@dataclass
class InteractiveVelocityCommand:
    """Apply hold-to-move translation and stepped yaw commands."""

    yaw_step: float
    max_linear_speed: float
    max_yaw_rate: float
    max_lateral_speed: float | None = None
    initial_linear_speed: float | None = None
    selected_linear_speed: float = field(init=False)
    values: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _held_translation_keys: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        scalars = np.asarray(
            (
                self.yaw_step,
                self.max_linear_speed,
                self.max_yaw_rate,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0.0):
            raise ValueError("interactive command steps and limits must be positive")
        if self.max_lateral_speed is None:
            self.max_lateral_speed = self.max_linear_speed
        elif (
            not np.isfinite(self.max_lateral_speed)
            or self.max_lateral_speed <= 0.0
        ):
            raise ValueError("max_lateral_speed must be positive and finite")
        if self.initial_linear_speed is None:
            self.selected_linear_speed = float(self.max_linear_speed)
        elif (
            not np.isfinite(self.initial_linear_speed)
            or self.initial_linear_speed <= 0.0
            or self.initial_linear_speed > self.max_linear_speed
        ):
            raise ValueError(
                "initial_linear_speed must be positive and no greater than max_linear_speed"
            )
        else:
            self.selected_linear_speed = float(self.initial_linear_speed)
        initial = np.asarray(self.values, dtype=np.float32)
        if initial.shape != (3,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial interactive command must be a finite 3-vector")
        self.values = initial.copy()
        self._clip()

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self.values.copy()

    def apply_key(self, character: str) -> tuple[str, np.ndarray] | None:
        """Apply a key-down event and return its label and command."""

        if len(character) != 1:
            raise ValueError("character must contain exactly one key")
        # Keep uppercase characters inert.  Terminal arrow sequences end in
        # uppercase A/B/C/D; accepting them as WASD would create motion while
        # the operator was only trying to navigate terminal history.
        key = character
        with self._lock:
            label: str | None = None
            if key == "w":
                self._held_translation_keys.add(key)
                self._update_translation()
                label = "forward held"
            elif key == "s":
                self._held_translation_keys.add(key)
                self._update_translation()
                label = "backward held"
            elif key == "a":
                self._held_translation_keys.add(key)
                self._update_translation()
                label = "left held"
            elif key == "d":
                self._held_translation_keys.add(key)
                self._update_translation()
                label = "right held"
            elif key == "q":
                self.values[2] += self.yaw_step
                label = "yaw left +"
            elif key == "e":
                self.values[2] -= self.yaw_step
                label = "yaw right -"
            elif key in "123456789":
                self.selected_linear_speed = min(
                    int(key) / 10.0, self.max_linear_speed
                )
                self._update_translation()
                label = f"speed preset {self.selected_linear_speed:.1f} m/s"
            elif key in (" ", "z", "r", "x", "\x1b"):
                self.values[:] = 0.0
                self._held_translation_keys.clear()
                label = "ZERO/HOLD"
            else:
                return None
            self._clip()
            return label, self.values.copy()

    def release_key(self, character: str) -> tuple[str, np.ndarray] | None:
        """Apply a key-up event; releasing W/S/A/D stops that axis."""

        if len(character) != 1:
            raise ValueError("character must contain exactly one key")
        if character not in "wasd":
            return None
        with self._lock:
            if character not in self._held_translation_keys:
                return f"{character.upper()} release ignored", self.values.copy()
            self._held_translation_keys.discard(character)
            self._update_translation()
            return f"{character.upper()} released", self.values.copy()

    def zero(self) -> np.ndarray:
        with self._lock:
            self.values[:] = 0.0
            self._held_translation_keys.clear()
            return self.values.copy()

    def set_velocity(self, vx: float, vy: float, wz: float) -> np.ndarray:
        """Set a finite programmatic command using the same configured limits."""

        requested = np.asarray((vx, vy, wz), dtype=np.float64)
        if requested.shape != (3,) or not np.all(np.isfinite(requested)):
            raise ValueError("velocity command must contain three finite values")
        with self._lock:
            self.values[:] = requested
            # Programmatic test execution owns translation until the operator
            # presses a key again; stale browser key-up events must not revive it.
            self._held_translation_keys.clear()
            self._clip()
            return self.values.copy()

    def _update_translation(self) -> None:
        self.values[0] = self.selected_linear_speed * (
            int("w" in self._held_translation_keys)
            - int("s" in self._held_translation_keys)
        )
        self.values[1] = self.max_lateral_speed * (
            int("a" in self._held_translation_keys)
            - int("d" in self._held_translation_keys)
        )

    def _clip(self) -> None:
        self.values[0] = np.clip(
            self.values[0], -self.max_linear_speed, self.max_linear_speed
        )
        self.values[1] = np.clip(
            self.values[1], -self.max_lateral_speed, self.max_lateral_speed
        )
        self.values[2] = np.clip(
            self.values[2], -self.max_yaw_rate, self.max_yaw_rate
        )


class TerminalKeyboard:
    """Nonblocking single-character reader that restores terminal settings."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._original: list[Any] | None = None

    @property
    def active(self) -> bool:
        return self._fd is not None

    def open(self) -> None:
        if self.active:
            return
        if not sys.stdin.isatty():
            raise RuntimeError("--interactive-keyboard requires a TTY")
        self._fd = sys.stdin.fileno()
        self._original = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def poll(self) -> list[str]:
        if self._fd is None:
            return []
        characters: list[str] = []
        while select.select([self._fd], [], [], 0.0)[0]:
            data = os.read(self._fd, 32)
            if not data:
                break
            characters.extend(data.decode("utf-8", errors="ignore"))
        return characters

    def close(self) -> None:
        if self._fd is not None and self._original is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original)
        self._fd = None
        self._original = None


__all__ = ["InteractiveVelocityCommand", "TerminalKeyboard"]
