import numpy as np
import pytest

from b2_control.interactive_command import InteractiveVelocityCommand


def _command() -> InteractiveVelocityCommand:
    return InteractiveVelocityCommand(
        yaw_step=0.10,
        max_linear_speed=0.20,
        max_yaw_rate=0.40,
    )


def test_translation_is_active_only_while_key_is_held() -> None:
    command = _command()
    for key in "waq":
        command.apply_key(key)
    np.testing.assert_allclose(command.snapshot(), [0.20, 0.20, 0.10])
    command.release_key("w")
    np.testing.assert_allclose(command.snapshot(), [0.0, 0.20, 0.10])
    command.release_key("a")
    np.testing.assert_allclose(command.snapshot(), [0.0, 0.0, 0.10])


def test_yaw_steps_clip_to_configured_limit() -> None:
    command = _command()
    for _ in range(100):
        command.apply_key("q")
    np.testing.assert_allclose(command.snapshot(), [0.0, 0.0, 0.40])


@pytest.mark.parametrize("key", [" ", "z", "r", "x", "\x1b"])
def test_stop_keys_zero_immediately(key: str) -> None:
    command = _command()
    command.apply_key("w")
    command.apply_key("q")
    event = command.apply_key(key)
    assert event is not None
    np.testing.assert_array_equal(command.snapshot(), np.zeros(3))


def test_speed_digits_select_w_s_speed_without_starting_motion() -> None:
    command = InteractiveVelocityCommand(0.1, 0.9, 0.4, initial_linear_speed=0.6)
    for digit in "123456789":
        event = command.apply_key(digit)
        assert event is not None
        assert command.selected_linear_speed == pytest.approx(int(digit) / 10)
        np.testing.assert_array_equal(command.snapshot(), np.zeros(3))
    command.apply_key("4")
    command.apply_key("w")
    np.testing.assert_allclose(command.snapshot(), [0.4, 0.0, 0.0])
    command.apply_key("7")
    np.testing.assert_allclose(command.snapshot(), [0.7, 0.0, 0.0])
    assert command.apply_key("0") is None
    assert command.apply_key("?") is None


def test_forward_and_lateral_limits_are_independent() -> None:
    command = InteractiveVelocityCommand(
        0.10,
        0.90,
        0.40,
        max_lateral_speed=0.25,
    )
    command.apply_key("w")
    command.apply_key("a")
    np.testing.assert_allclose(command.snapshot(), [0.90, 0.25, 0.0])


def test_programmatic_velocity_uses_limits_and_clears_held_keys() -> None:
    command = InteractiveVelocityCommand(
        0.10,
        0.90,
        0.40,
        max_lateral_speed=0.25,
    )
    command.apply_key("w")
    result = command.set_velocity(4.0, -2.0, 1.5)
    np.testing.assert_allclose(result, [0.90, -0.25, 0.40])
    command.release_key("w")
    np.testing.assert_allclose(command.snapshot(), [0.90, -0.25, 0.40])


@pytest.mark.parametrize(
    "values",
    [
        (float("nan"), 0.0, 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, 0.0, float("-inf")),
    ],
)
def test_programmatic_velocity_rejects_non_finite(values: tuple[float, ...]) -> None:
    command = _command()
    with pytest.raises(ValueError):
        command.set_velocity(*values)
    np.testing.assert_array_equal(command.snapshot(), np.zeros(3))


def test_opposite_held_keys_cancel_until_one_is_released() -> None:
    command = _command()
    command.apply_key("w")
    command.apply_key("s")
    np.testing.assert_allclose(command.snapshot(), [0.0, 0.0, 0.0])
    command.release_key("s")
    np.testing.assert_allclose(command.snapshot(), [0.20, 0.0, 0.0])
    command.release_key("w")
    np.testing.assert_allclose(command.snapshot(), [0.0, 0.0, 0.0])


def test_terminal_arrow_suffixes_and_shifted_letters_are_inert() -> None:
    command = _command()
    for key in "ABCDWQ":
        assert command.apply_key(key) is None
    np.testing.assert_array_equal(command.snapshot(), np.zeros(3))


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_step_or_limit_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        InteractiveVelocityCommand(value, 0.2, 0.4)
