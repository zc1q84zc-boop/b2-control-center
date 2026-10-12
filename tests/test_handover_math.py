from __future__ import annotations

import numpy as np
import pytest

from b2_control.handover_math import (
    MODEL_JOINT_MAX_RAD,
    MODEL_JOINT_MIN_RAD,
    model_joint_limit_margin_rad,
    slew_limited_step,
    validate_model_joint_range,
)


def test_model_ranges_follow_policy_joint_order() -> None:
    np.testing.assert_array_equal(MODEL_JOINT_MIN_RAD, (-0.87, -0.94, -2.82) * 4)
    np.testing.assert_array_equal(MODEL_JOINT_MAX_RAD, (0.87, 4.69, -0.43) * 4)


def test_validate_range_accepts_boundaries() -> None:
    np.testing.assert_array_equal(
        validate_model_joint_range(MODEL_JOINT_MIN_RAD), MODEL_JOINT_MIN_RAD
    )
    np.testing.assert_array_equal(
        validate_model_joint_range(MODEL_JOINT_MAX_RAD), MODEL_JOINT_MAX_RAD
    )


@pytest.mark.parametrize("index", range(12))
def test_validate_range_rejects_each_joint(index: int) -> None:
    target = (MODEL_JOINT_MIN_RAD + MODEL_JOINT_MAX_RAD) / 2
    target[index] = MODEL_JOINT_MAX_RAD[index] + 0.01
    with pytest.raises(ValueError, match=rf"target_rad\[{index}\]"):
        validate_model_joint_range(target)


def test_margin_is_distance_to_nearest_limit() -> None:
    center = (MODEL_JOINT_MIN_RAD + MODEL_JOINT_MAX_RAD) / 2
    np.testing.assert_allclose(
        model_joint_limit_margin_rad(center),
        (MODEL_JOINT_MAX_RAD - MODEL_JOINT_MIN_RAD) / 2,
    )


def test_slew_step_limits_each_joint_in_both_directions() -> None:
    previous = (MODEL_JOINT_MIN_RAD + MODEL_JOINT_MAX_RAD) / 2
    requested = previous + np.array((-0.5, 0.5) * 6)
    result = slew_limited_step(
        previous_target_rad=previous,
        requested_target_rad=requested,
        max_rate_rad_s=np.full(12, 0.5),
        dt_s=0.02,
    )
    np.testing.assert_allclose(result - previous, np.array((-0.01, 0.01) * 6))


@pytest.mark.parametrize("rate", [0.0, -1.0, np.nan])
def test_slew_step_rejects_invalid_rate(rate: float) -> None:
    center = (MODEL_JOINT_MIN_RAD + MODEL_JOINT_MAX_RAD) / 2
    with pytest.raises(ValueError, match="max_rate_rad_s"):
        slew_limited_step(
            previous_target_rad=center,
            requested_target_rad=center,
            max_rate_rad_s=np.full(12, rate),
            dt_s=0.02,
        )


@pytest.mark.parametrize("dt", [0.0, -0.01, np.nan])
def test_slew_step_rejects_invalid_dt(dt: float) -> None:
    center = (MODEL_JOINT_MIN_RAD + MODEL_JOINT_MAX_RAD) / 2
    with pytest.raises(ValueError, match="dt_s"):
        slew_limited_step(
            previous_target_rad=center,
            requested_target_rad=center,
            max_rate_rad_s=np.ones(12),
            dt_s=dt,
        )
