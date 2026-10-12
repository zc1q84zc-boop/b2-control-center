from __future__ import annotations

import numpy as np

from b2_control.remote_control import (
    KEY_A,
    KEY_B,
    KEY_L2,
    KEY_R2,
    RemoteMode,
    RemoteModeController,
    RemoteVelocityMapper,
    emergency_stop_pressed,
)


def test_preserves_official_unitree_axis_mapping() -> None:
    mapper = RemoteVelocityMapper(
        max_forward_speed=0.9,
        max_lateral_speed=0.25,
        max_yaw_rate=0.4,
    )
    np.testing.assert_allclose(
        mapper.map_axes([0.4, 0.5, -0.25, 0.9]),
        [0.45, -0.1, 0.1],
    )


def test_deadzone_zeros_centered_sticks() -> None:
    mapper = RemoteVelocityMapper(deadzone=0.1)
    np.testing.assert_array_equal(
        mapper.map_axes([0.09, -0.09, 0.09, 1.0]),
        np.zeros(3, dtype=np.float32),
    )
    assert mapper.sticks_centered([0.09, -0.09, 0.09, 1.0])


def test_a_enters_policy_and_r2_returns_stand() -> None:
    controller = RemoteModeController()
    assert controller.update(KEY_A, sticks_centered=True) is RemoteMode.POLICY
    controller.update(0, sticks_centered=True)
    assert controller.update(KEY_R2, sticks_centered=True) is RemoteMode.STAND


def test_a_requires_centered_sticks() -> None:
    controller = RemoteModeController()
    assert controller.update(KEY_A, sticks_centered=False) is RemoteMode.STAND


def test_l2_b_emergency_stop_is_latched() -> None:
    controller = RemoteModeController()
    assert emergency_stop_pressed(KEY_L2 | KEY_B)
    assert (
        controller.update(KEY_L2 | KEY_B, sticks_centered=True)
        is RemoteMode.EMERGENCY_STOP
    )
    assert controller.update(0, sticks_centered=True) is RemoteMode.EMERGENCY_STOP
