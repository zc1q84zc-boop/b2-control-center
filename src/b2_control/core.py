from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .policy_math import (
    ACTION_DIM,
    build_observation,
    direct_velocity_command,
    heading_first_command,
    joint_targets,
)


class Policy(Protocol):
    def infer(self, observation: Any) -> np.ndarray: ...


class ObservationAdapter(Protocol):
    def reset(self) -> None: ...

    def build_input(
        self,
        body_angular_velocity: Any,
        projected_gravity: Any,
        command: Any,
        joint_position: Any,
        joint_velocity: Any,
    ) -> np.ndarray: ...

    def commit_action(self, raw_action: Any) -> None: ...


@dataclass(frozen=True)
class PolicyOutput:
    q_desired_rad: np.ndarray
    raw_action: np.ndarray
    applied_action: np.ndarray
    observation: np.ndarray
    command: np.ndarray
    heading_error_rad: float
    alignment_gate: float


class B2PolicyCore:
    def __init__(
        self,
        policy: Policy,
        *,
        target_slew_rad_s: float | None = None,
        action_clip_abs: float | None = None,
        control_dt_s: float = 0.02,
        observation_adapter: ObservationAdapter | None = None,
    ):
        if target_slew_rad_s is not None and (
            not np.isfinite(target_slew_rad_s) or target_slew_rad_s <= 0.0
        ):
            raise ValueError("target_slew_rad_s must be positive and finite")
        if not np.isfinite(control_dt_s) or control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive and finite")
        if action_clip_abs is not None and (
            not np.isfinite(action_clip_abs) or action_clip_abs <= 0.0
        ):
            raise ValueError("action_clip_abs must be positive and finite")
        self.policy = policy
        self.target_slew_rad_s = target_slew_rad_s
        self.action_clip_abs = action_clip_abs
        self.control_dt_s = float(control_dt_s)
        self.observation_adapter = observation_adapter
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._previous_target: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_action.fill(0.0)
        self._previous_target = None
        if self.observation_adapter is not None:
            self.observation_adapter.reset()

    def step(
        self,
        *,
        omega_body_rad_s: Any,
        rotation_body_to_world: Any,
        joint_position_rad: Any,
        joint_velocity_rad_s: Any,
        target_yaw_rad: float,
        current_yaw_rad: float,
        nominal_speed_m_s: float,
        hold: bool,
        velocity_command: Any | None = None,
    ) -> PolicyOutput:
        if velocity_command is None:
            command, error, gate = heading_first_command(
                target_yaw_rad,
                current_yaw_rad,
                nominal_speed_m_s,
                hold=hold,
            )
        else:
            command = (
                np.zeros(3, dtype=np.float32)
                if hold
                else direct_velocity_command(velocity_command)
            )
            error = 0.0
            gate = 1.0
        if self.observation_adapter is None:
            observation = build_observation(
                omega_body=omega_body_rad_s,
                rotation_body_to_world=rotation_body_to_world,
                command=command,
                q=joint_position_rad,
                dq=joint_velocity_rad_s,
                previous_action=self.previous_action,
            )
        else:
            projected_gravity = (
                np.asarray(rotation_body_to_world, dtype=np.float64).T
                @ np.array((0.0, 0.0, -1.0), dtype=np.float64)
            )
            observation = self.observation_adapter.build_input(
                omega_body_rad_s,
                projected_gravity,
                command,
                joint_position_rad,
                joint_velocity_rad_s,
            )
        action = np.asarray(self.policy.infer(observation), dtype=np.float32)
        if self.action_clip_abs is None:
            applied_action = action
        else:
            applied_action = np.clip(
                action, -self.action_clip_abs, self.action_clip_abs
            )
        raw_desired = joint_targets(applied_action)
        if self.target_slew_rad_s is None:
            desired = raw_desired
        else:
            if self._previous_target is None:
                previous_target = np.asarray(
                    joint_position_rad, dtype=np.float64
                ).copy()
            else:
                previous_target = self._previous_target
            maximum_step = self.target_slew_rad_s * self.control_dt_s
            desired = previous_target + np.clip(
                raw_desired - previous_target, -maximum_step, maximum_step
            )
            self._previous_target = desired.copy()
        self.previous_action[:] = action
        if self.observation_adapter is not None:
            self.observation_adapter.commit_action(action)
        return PolicyOutput(
            q_desired_rad=desired.copy(),
            raw_action=action.copy(),
            applied_action=applied_action.copy(),
            observation=observation.copy(),
            command=command.copy(),
            heading_error_rad=error,
            alignment_gate=gate,
        )
