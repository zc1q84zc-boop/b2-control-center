import numpy as np
import pytest

from b2_control.policy_math import (
    ACTION_DIM,
    ACTION_SCALE,
    JOINT_NAMES,
    OBSERVATION_DIM,
    Q_DEFAULT,
    build_observation,
    heading_first_command,
    joint_targets,
)

def _observation_inputs() -> dict[str, np.ndarray]:
    return {
        "omega_body": np.zeros(3),
        "rotation_body_to_world": np.eye(3),
        "command": np.zeros(3),
        "q": Q_DEFAULT.copy(),
        "dq": np.zeros(12),
        "previous_action": np.zeros(12),
    }

def test_public_constants_match_policy_contract():
    assert JOINT_NAMES == (
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
    np.testing.assert_array_equal(
        Q_DEFAULT,
        np.array(
            (-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5),
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(
        ACTION_SCALE,
        np.array((0.125, 0.25, 0.25) * 4, dtype=np.float64),
    )
    assert Q_DEFAULT.dtype == np.dtype(np.float64)
    assert ACTION_SCALE.dtype == np.dtype(np.float64)
    assert not Q_DEFAULT.flags.writeable
    assert not ACTION_SCALE.flags.writeable
    assert OBSERVATION_DIM == 45
    assert ACTION_DIM == 12

@pytest.mark.parametrize("constant", (Q_DEFAULT, ACTION_SCALE))
def test_public_array_constants_reject_mutation(constant):
    with pytest.raises(ValueError, match="read-only"):
        constant[0] = 0.0

def test_level_default_hold_observation_is_zero_except_gravity():
    observation = build_observation(
        omega_body=np.zeros(3),
        rotation_body_to_world=np.eye(3),
        command=np.zeros(3),
        q=Q_DEFAULT,
        dq=np.zeros(12),
        previous_action=np.zeros(12),
    )
    expected = np.zeros((1, 45), dtype=np.float32)
    expected[0, 3:6] = [0, 0, -1]
    np.testing.assert_array_equal(observation, expected)

def test_observation_has_exact_order_scaling_dtype_and_layout():
    omega = np.array((4.0, 8.0, 12.0))
    rotation = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    command = np.array((0.3, -0.2, 0.7))
    q_offset = np.arange(12, dtype=np.float64) / 10.0
    dq = np.arange(12, dtype=np.float64) + 1.0
    previous_action = -(np.arange(12, dtype=np.float64) + 1.0)

    observation = build_observation(
        omega_body=omega,
        rotation_body_to_world=rotation,
        command=command,
        q=Q_DEFAULT + q_offset,
        dq=dq,
        previous_action=previous_action,
    )
    expected = np.concatenate(
        (
            0.25 * omega,
            rotation.T @ np.array((0.0, 0.0, -1.0)),
            command,
            q_offset,
            0.05 * dq,
            previous_action,
        )
    ).astype(np.float32)[None, :]

    assert observation.shape == (1, OBSERVATION_DIM)
    assert observation.dtype == np.float32
    assert observation.flags.c_contiguous
    np.testing.assert_array_equal(observation, expected)

@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("omega_body", np.zeros((1, 3))),
        ("command", np.zeros(2)),
        ("q", np.zeros(11)),
        ("dq", np.zeros(13)),
        ("previous_action", 0.0),
    ),
)
def test_observation_rejects_wrong_vector_shapes(field, value):
    inputs = _observation_inputs()
    inputs[field] = value
    with pytest.raises(ValueError, match=field):
        build_observation(**inputs)

@pytest.mark.parametrize(
    ("field", "size"),
    (
        ("omega_body", 3),
        ("command", 3),
        ("q", 12),
        ("dq", 12),
        ("previous_action", 12),
    ),
)
def test_observation_rejects_nonfinite_vectors(field, size):
    inputs = _observation_inputs()
    value = np.zeros(size)
    value[-1] = np.nan
    inputs[field] = value
    with pytest.raises(ValueError, match=field):
        build_observation(**inputs)

@pytest.mark.parametrize("field", ("omega_body", "previous_action"))
def test_observation_rejects_float32_overflow(field):
    inputs = _observation_inputs()
    value = np.zeros(3 if field == "omega_body" else 12)
    value[0] = np.finfo(np.float64).max
    inputs[field] = value
    with pytest.raises(ValueError, match="observation.*float32"):
        build_observation(**inputs)

def test_observation_rejects_integer_conversion_overflow_with_field_name():
    inputs = _observation_inputs()
    inputs["command"] = [10**1000, 0, 0]
    with pytest.raises(ValueError, match="command"):
        build_observation(**inputs)

def test_observation_rejects_unconvertible_vector_with_field_name():
    inputs = _observation_inputs()
    inputs["command"] = ["not-a-number", 0.0, 0.0]
    with pytest.raises(ValueError, match="command"):
        build_observation(**inputs)

@pytest.mark.parametrize(
    ("rotation", "message"),
    (
        (np.eye(2), "shape"),
        (np.array(((1.0, 0.0, 0.0), (0.0, np.inf, 0.0), (0.0, 0.0, 1.0))), "finite"),
        (np.diag((1.0, 1.0, 2.0)), "orthonormal"),
        (np.diag((1.0, 1.0, -1.0)), "determinant"),
    ),
)
def test_observation_rejects_invalid_rotation(rotation, message):
    inputs = _observation_inputs()
    inputs["rotation_body_to_world"] = rotation
    with pytest.raises(ValueError, match=f"rotation_body_to_world.*{message}"):
        build_observation(**inputs)

def test_observation_rejects_unconvertible_rotation_with_field_name():
    inputs = _observation_inputs()
    inputs["rotation_body_to_world"] = "not-a-matrix"
    with pytest.raises(ValueError, match="rotation_body_to_world"):
        build_observation(**inputs)

def test_observation_rejects_rotation_conversion_overflow_with_field_name():
    inputs = _observation_inputs()
    inputs["rotation_body_to_world"] = ((10**1000, 0, 0), (0, 1, 0), (0, 0, 1))
    with pytest.raises(ValueError, match="rotation_body_to_world"):
        build_observation(**inputs)

def test_heading_command_holds_exact_zero():
    command, error, gate = heading_first_command(1.0, 0.0, 0.25, hold=True)
    np.testing.assert_array_equal(command, np.zeros(3, dtype=np.float32))
    assert error == pytest.approx(1.0)
    assert 0.0 < gate < 1.0

def test_negative_nominal_speed_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        heading_first_command(0.0, 0.0, -0.25)

@pytest.mark.parametrize(
    ("target_heading", "current_heading", "nominal_speed"),
    (
        (np.nan, 0.0, 0.25),
        (0.0, np.inf, 0.25),
        (0.0, 0.0, np.nan),
    ),
)
def test_heading_command_rejects_nonfinite_scalars(
    target_heading, current_heading, nominal_speed
):
    with pytest.raises(ValueError, match="finite"):
        heading_first_command(target_heading, current_heading, nominal_speed)

@pytest.mark.parametrize("value", ([0.0], np.array([0.0])))
def test_heading_command_rejects_non_scalar_input(value):
    with pytest.raises(ValueError, match="target_heading.*scalar"):
        heading_first_command(value, 0.0, 0.25)

def test_heading_command_rejects_scalar_conversion_overflow_with_field_name():
    with pytest.raises(ValueError, match="target_heading"):
        heading_first_command(10**1000, 0.0, 0.25)

def test_heading_command_rejects_float32_command_overflow():
    with pytest.raises(ValueError, match="command.*float32"):
        heading_first_command(0.0, 0.0, np.finfo(np.float64).max)

def test_heading_command_has_exact_output_contract_and_antipodal_convention():
    command, error, _ = heading_first_command(np.pi, 0.0, 0.25)
    assert command.shape == (3,)
    assert command.dtype == np.float32
    assert command.flags.c_contiguous
    assert error == pytest.approx(-np.pi)

def test_heading_command_wraps_error_and_clips_yaw_rate():
    command, error, gate = heading_first_command(3.0, -3.0, 0.5)
    expected_error = 6.0 - 2.0 * np.pi
    assert error == pytest.approx(expected_error)
    assert gate == pytest.approx(np.exp(-((expected_error / 0.55) ** 2)))
    np.testing.assert_array_equal(
        command,
        np.array((0.5 * gate, 0.0, expected_error), dtype=np.float32),
    )

    clipped, clipped_error, _ = heading_first_command(2.0, 0.0, 0.5)
    assert clipped_error == pytest.approx(2.0)
    assert clipped[2] == np.float32(1.0)

def test_heading_command_wraps_extreme_finite_headings_without_overflow():
    maximum = np.finfo(np.float64).max
    command, error, gate = heading_first_command(maximum, -maximum, 0.25)
    reduced_delta = maximum % (2.0 * np.pi) - (-maximum % (2.0 * np.pi))
    expected_error = (reduced_delta + np.pi) % (2.0 * np.pi) - np.pi

    assert error == pytest.approx(expected_error)
    assert np.isfinite(gate)
    assert np.all(np.isfinite(command))

def test_joint_targets_use_exact_scale():
    action = np.ones(12)
    targets = joint_targets(action)
    assert targets.shape == (ACTION_DIM,)
    assert targets.dtype == np.float64
    assert targets.flags.c_contiguous
    np.testing.assert_allclose(targets, Q_DEFAULT + ACTION_SCALE)

@pytest.mark.parametrize("action", (np.zeros(11), np.zeros((1, 12))))
def test_joint_targets_reject_wrong_shape(action):
    with pytest.raises(ValueError, match="action"):
        joint_targets(action)

def test_joint_targets_reject_nonfinite_action():
    action = np.zeros(12)
    action[0] = np.inf
    with pytest.raises(ValueError, match="action"):
        joint_targets(action)
