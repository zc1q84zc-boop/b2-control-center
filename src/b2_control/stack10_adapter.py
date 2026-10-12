"""Reference Stack10 deployment adapter for B2 model_31997.

The production controller may use another language or runtime, but it must
reproduce this module's values and sequencing exactly.
"""

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
    (-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5),
    dtype=np.float64,
)
ACTION_SCALE = np.array((0.125, 0.25, 0.25) * 4, dtype=np.float64)

FRAME_DIM = 45
ACTION_DIM = 12
HISTORY_SAMPLES = 10
STACKED_DIM = 423


def _finite_vector(
    name: str,
    value: Any,
    size: int,
    *,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    vector = np.asarray(value, dtype=dtype)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def projected_gravity_from_rotation(rotation_body_to_world: Any) -> np.ndarray:
    """Project world down into the body frame from a body-to-world rotation."""
    rotation = np.asarray(rotation_body_to_world, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(
            "rotation_body_to_world must have shape (3, 3), "
            f"got {rotation.shape}."
        )
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_body_to_world contains non-finite values.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError("rotation_body_to_world is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("rotation_body_to_world must have determinant +1.")
    return np.asarray(rotation.T @ (0.0, 0.0, -1.0), dtype=np.float64)


def build_current_frame(
    body_angular_velocity: Any,
    projected_gravity: Any,
    command: Any,
    joint_position: Any,
    joint_velocity: Any,
    previous_action: Any,
) -> np.ndarray:
    """Build the baseline 45-value frame before selective history stacking."""
    omega = _finite_vector("body_angular_velocity", body_angular_velocity, 3)
    gravity = _finite_vector("projected_gravity", projected_gravity, 3)
    velocity_command = _finite_vector("command", command, 3)
    q = _finite_vector("joint_position", joint_position, ACTION_DIM)
    dq = _finite_vector("joint_velocity", joint_velocity, ACTION_DIM)
    last_action = _finite_vector("previous_action", previous_action, ACTION_DIM)

    frame = np.concatenate(
        (
            0.25 * omega,
            gravity,
            velocity_command,
            q - Q_DEFAULT,
            0.05 * dq,
            last_action,
        )
    ).astype(np.float32, copy=False)
    if frame.shape != (FRAME_DIM,):
        raise RuntimeError(f"Internal frame shape is {frame.shape}, expected (45,).")
    return np.ascontiguousarray(frame)


def assemble_stack10(frames: Any) -> np.ndarray:
    """Assemble one to ten chronological frames into `[1, 423]` term-major form."""
    values = np.asarray(frames, dtype=np.float32)
    if values.shape == (FRAME_DIM,):
        values = values.reshape(1, FRAME_DIM)
    if (
        values.ndim != 2
        or values.shape[1] != FRAME_DIM
        or not 1 <= values.shape[0] <= HISTORY_SAMPLES
    ):
        raise ValueError(
            "frames must have shape (45,) or (N, 45), 1 <= N <= 10; "
            f"got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("frames contain non-finite values.")
    if values.shape[0] < HISTORY_SAMPLES:
        padding = np.repeat(
            values[0:1], HISTORY_SAMPLES - values.shape[0], axis=0
        )
        values = np.concatenate((padding, values), axis=0)

    stacked = np.concatenate(
        (
            values[:, 0:3].reshape(-1),
            values[:, 3:6].reshape(-1),
            values[-1, 6:9],
            values[:, 9:21].reshape(-1),
            values[:, 21:33].reshape(-1),
            values[:, 33:45].reshape(-1),
        )
    ).astype(np.float32, copy=False)
    if stacked.shape != (STACKED_DIM,):
        raise RuntimeError(
            f"Internal Stack10 shape is {stacked.shape}, expected (423,)."
        )
    return np.ascontiguousarray(stacked.reshape(1, STACKED_DIM))


def action_to_joint_targets(raw_action: Any) -> np.ndarray:
    """Convert the raw 12-action actor output to absolute joint targets."""
    action = _finite_vector("raw_action", raw_action, ACTION_DIM)
    return Q_DEFAULT + ACTION_SCALE * action


class Stack10ObservationAdapter:
    """Own the history and previous-action state for one robot/controller."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Start a new episode/mode; the first future frame will fill history."""
        self._frames: list[np.ndarray] = []
        self._previous_action = np.zeros(ACTION_DIM, dtype=np.float32)

    @property
    def previous_action(self) -> np.ndarray:
        return self._previous_action.copy()

    def build_input(
        self,
        body_angular_velocity: Any,
        projected_gravity: Any,
        command: Any,
        joint_position: Any,
        joint_velocity: Any,
    ) -> np.ndarray:
        """Append one 50 Hz frame and return the actor's `[1, 423]` input."""
        frame = build_current_frame(
            body_angular_velocity,
            projected_gravity,
            command,
            joint_position,
            joint_velocity,
            self._previous_action,
        )
        if not self._frames:
            self._frames = [frame.copy() for _ in range(HISTORY_SAMPLES)]
        else:
            self._frames.append(frame)
            self._frames = self._frames[-HISTORY_SAMPLES:]
        return assemble_stack10(self._frames)

    def commit_action(self, raw_action: Any) -> None:
        """Save a successful raw actor output for the next policy tick."""
        self._previous_action = _finite_vector(
            "raw_action", raw_action, ACTION_DIM, dtype=np.float32
        ).copy()


def self_test() -> None:
    """Check term-major offsets, reset filling, and action transform."""
    frames = np.arange(3 * FRAME_DIM, dtype=np.float32).reshape(3, FRAME_DIM)
    padded = np.concatenate((np.repeat(frames[0:1], 7, axis=0), frames), axis=0)
    expected = np.concatenate(
        (
            padded[:, 0:3].reshape(-1),
            padded[:, 3:6].reshape(-1),
            padded[-1, 6:9],
            padded[:, 9:21].reshape(-1),
            padded[:, 21:33].reshape(-1),
            padded[:, 33:45].reshape(-1),
        )
    ).reshape(1, STACKED_DIM)
    np.testing.assert_array_equal(assemble_stack10(frames), expected)

    adapter = Stack10ObservationAdapter()
    first = adapter.build_input(
        np.zeros(3),
        (0.0, 0.0, -1.0),
        np.zeros(3),
        Q_DEFAULT,
        np.zeros(ACTION_DIM),
    )
    expected_frame = np.zeros(FRAME_DIM, dtype=np.float32)
    expected_frame[5] = -1.0
    np.testing.assert_array_equal(
        first,
        assemble_stack10(np.repeat(expected_frame[None, :], 10, axis=0)),
    )
    np.testing.assert_array_equal(
        action_to_joint_targets(np.zeros(ACTION_DIM)), Q_DEFAULT
    )


if __name__ == "__main__":
    self_test()
    print("PASS: Stack10 adapter self-test")
