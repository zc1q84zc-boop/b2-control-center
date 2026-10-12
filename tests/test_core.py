import numpy as np

from b2_control.core import B2PolicyCore
from b2_control.policy_math import ACTION_SCALE, Q_DEFAULT


class FakePolicy:
    def __init__(self):
        self.observations = []

    def infer(self, observation):
        self.observations.append(np.asarray(observation).copy())
        return np.full(12, len(self.observations), dtype=np.float32)


def _step(core):
    return core.step(
        omega_body_rad_s=np.zeros(3),
        rotation_body_to_world=np.eye(3),
        joint_position_rad=Q_DEFAULT,
        joint_velocity_rad_s=np.zeros(12),
        target_yaw_rad=0.0,
        current_yaw_rad=0.0,
        nominal_speed_m_s=0.25,
        hold=True,
    )


def test_previous_raw_action_is_fed_back_then_reset():
    actor = FakePolicy()
    core = B2PolicyCore(actor)
    _step(core)
    _step(core)
    np.testing.assert_array_equal(actor.observations[0][0, 33:45], np.zeros(12))
    np.testing.assert_array_equal(actor.observations[1][0, 33:45], np.ones(12))
    core.reset()
    _step(core)
    np.testing.assert_array_equal(actor.observations[2][0, 33:45], np.zeros(12))


def test_hold_command_is_exact_zero():
    output = _step(B2PolicyCore(FakePolicy()))
    np.testing.assert_array_equal(output.command, np.zeros(3, dtype=np.float32))


def test_direct_velocity_command_is_inserted_into_observation():
    actor = FakePolicy()
    core = B2PolicyCore(actor)
    output = core.step(
        omega_body_rad_s=np.zeros(3),
        rotation_body_to_world=np.eye(3),
        joint_position_rad=Q_DEFAULT,
        joint_velocity_rad_s=np.zeros(12),
        target_yaw_rad=0.0,
        current_yaw_rad=0.0,
        nominal_speed_m_s=0.25,
        hold=False,
        velocity_command=np.array([0.20, -0.05, 0.30]),
    )
    np.testing.assert_allclose(output.command, [0.20, -0.05, 0.30])
    np.testing.assert_allclose(actor.observations[0][0, 6:9], output.command)
    assert output.heading_error_rad == 0.0
    assert output.alignment_gate == 1.0


def test_hold_overrides_direct_velocity_command():
    actor = FakePolicy()
    core = B2PolicyCore(actor)
    output = core.step(
        omega_body_rad_s=np.zeros(3),
        rotation_body_to_world=np.eye(3),
        joint_position_rad=Q_DEFAULT,
        joint_velocity_rad_s=np.zeros(12),
        target_yaw_rad=0.0,
        current_yaw_rad=0.0,
        nominal_speed_m_s=0.25,
        hold=True,
        velocity_command=np.array([0.20, -0.05, 0.30]),
    )
    np.testing.assert_array_equal(output.command, np.zeros(3))


def test_direct_velocity_command_rejects_invalid_vectors():
    core = B2PolicyCore(FakePolicy())
    for command in ([0.1, 0.2], [0.1, np.nan, 0.2]):
        try:
            core.step(
                omega_body_rad_s=np.zeros(3),
                rotation_body_to_world=np.eye(3),
                joint_position_rad=Q_DEFAULT,
                joint_velocity_rad_s=np.zeros(12),
                target_yaw_rad=0.0,
                current_yaw_rad=0.0,
                nominal_speed_m_s=0.25,
                hold=False,
                velocity_command=command,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid velocity command {command}")


def test_step_returns_raw_action_and_scaled_joint_targets():
    output = _step(B2PolicyCore(FakePolicy()))
    np.testing.assert_array_equal(output.raw_action, np.ones(12, dtype=np.float32))
    np.testing.assert_array_equal(
        output.applied_action, np.ones(12, dtype=np.float32)
    )
    np.testing.assert_allclose(output.q_desired_rad, Q_DEFAULT + ACTION_SCALE)
    assert output.observation.shape == (1, 45)
    assert output.heading_error_rad == 0.0
    assert output.alignment_gate == 1.0


def test_target_slew_starts_at_measured_q_and_feedbacks_raw_action():
    actor = FakePolicy()
    core = B2PolicyCore(actor, target_slew_rad_s=5.0, control_dt_s=0.02)

    first = _step(core)
    np.testing.assert_allclose(first.q_desired_rad, Q_DEFAULT + 0.1)
    second = _step(core)
    np.testing.assert_array_equal(
        actor.observations[1][0, 33:45], np.ones(12, dtype=np.float32)
    )
    np.testing.assert_allclose(second.q_desired_rad, Q_DEFAULT + 0.2)

    core.reset()
    reset = _step(core)
    np.testing.assert_allclose(reset.q_desired_rad, Q_DEFAULT + 0.1)


def test_target_slew_configuration_rejects_invalid_values():
    actor = FakePolicy()
    for value in (0.0, -1.0, float("inf"), float("nan")):
        try:
            B2PolicyCore(actor, target_slew_rad_s=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid target slew {value}")


def test_action_clip_changes_target_but_preserves_raw_action_feedback():
    actor = FakePolicy()
    core = B2PolicyCore(actor, action_clip_abs=0.5)
    first = _step(core)
    np.testing.assert_array_equal(first.raw_action, np.ones(12))
    np.testing.assert_array_equal(first.applied_action, np.full(12, 0.5))
    np.testing.assert_allclose(
        first.q_desired_rad, Q_DEFAULT + 0.5 * ACTION_SCALE
    )
    _step(core)
    np.testing.assert_array_equal(actor.observations[1][0, 33:45], np.ones(12))


def test_action_clip_configuration_rejects_invalid_values():
    actor = FakePolicy()
    for value in (0.0, -1.0, float("inf"), float("nan")):
        try:
            B2PolicyCore(actor, action_clip_abs=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid action clip {value}")
