from __future__ import annotations

import json
import math
from pathlib import Path


PROFILE = Path(__file__).parents[1] / "config" / "b2-reference-profile.json"


def test_hardware_data_profile_matches_current_hardware_comparison() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert profile["schema_version"] == 1
    assert profile["status"] == "hardware_hold_candidate_pending_validation"
    assert profile["activation_permitted"] is False
    assert profile["fallback_params"] == {
        "damping_kp": 0.0,
        "damping_kd": 5.0,
        "approved_for_hardware": False,
    }
    assert profile["hardware_gains"]["kp"] == [160.0] * 12
    assert profile["hardware_gains"]["kd"] == [5.0] * 12
    assert profile["hardware_gains"]["approved_for_hardware"] is False
    limits = profile["safety_limits"]
    assert limits["dq_max"] == [23.0, 23.0, 14.0] * 4
    assert limits["max_angle_step_rad"] == [0.046, 0.046, 0.028] * 4


def test_hardware_data_profile_limits_are_finite_ordered_12_vectors() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    limits = profile["safety_limits"]
    for key in ("q_min", "q_max", "dq_max", "max_angle_step_rad"):
        assert len(limits[key]) == 12
        assert all(math.isfinite(value) for value in limits[key])
    assert all(
        lower < upper
        for lower, upper in zip(limits["q_min"], limits["q_max"], strict=True)
    )
