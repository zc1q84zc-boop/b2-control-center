from __future__ import annotations

import json
from pathlib import Path

import pytest

from b2_control.parameter_profile import (
    REQUIRED_HARDWARE_APPROVAL_FIELDS,
    hardware_activation_blockers,
    hardware_policy_blockers,
    load_parameter_profile,
    require_hardware_activation_approval,
)


PROFILE_PATH = Path(__file__).parents[1] / "configs" / "b2-reference-profile.json"


def test_reference_profile_keeps_hardware_activation_blocked() -> None:
    profile = load_parameter_profile(PROFILE_PATH)
    blockers = hardware_activation_blockers(profile)
    assert "status is not approved" in blockers
    assert "activation_permitted is not true" in blockers
    assert all(
        f"hardware_approval.{field} is missing" in blockers
        for field in REQUIRED_HARDWARE_APPROVAL_FIELDS
    )
    with pytest.raises(ValueError, match="hardware activation blocked"):
        require_hardware_activation_approval(profile)


def test_reference_profile_preserves_known_policy_values() -> None:
    profile = load_parameter_profile(PROFILE_PATH)
    assert profile["policy"]["frequency_hz"] == 50
    assert profile["policy"]["training_kp"] == [160] * 12
    assert profile["policy"]["training_kd"] == [5] * 12
    assert profile["policy"]["forward_speed_range_m_s"] == [0.25, 1.0]
    assert profile["vendor_example_reference"]["usable_as_policy_hardware_defaults"] is False
    candidates = profile["engineering_review_candidates"]
    assert candidates["position_kp"]["value"] == [160] * 12
    assert candidates["position_kp"]["approved_for_hardware"] is False
    assert candidates["velocity_kd"]["value"] == [5] * 12
    assert candidates["joint_velocity_ceiling_rad_s"]["usable_as_target_slew"] is False
    assert candidates["joint_effort_ceiling"]["usable_as_runtime_torque_limit"] is False
    assert candidates["low_command_period_s"]["usable_as_watchdog_timeout"] is False


def test_complete_approved_profile_passes_gate() -> None:
    profile = load_parameter_profile(PROFILE_PATH)
    profile["status"] = "approved"
    profile["activation_permitted"] = True
    approval = profile["hardware_approval"]
    for field in (
        "position_kp",
        "velocity_kd",
        "target_slew_rad_s",
        "soft_limit_margin_rad",
        "torque_limit",
        "current_limit",
        "joint_velocity_trip_rad_s",
        "tracking_error_trip_rad",
    ):
        approval[field] = [1.0] * 12
    for field in (
        "gain_ramp_s",
        "damping_kd",
        "tracking_error_duration_s",
        "battery_soc_min_percent",
        "battery_current_abs_limit_raw",
        "battery_temperature_trip_c",
        "battery_temperature_reset_c",
        "tilt_trip_rad",
        "angular_rate_trip_rad_s",
        "state_timeout_s",
        "policy_timeout_s",
        "command_timeout_s",
        "publisher_timeout_s",
        "recovery_timeout_s",
        "lowcmd_conflict_check_s",
        "fault_debounce_s",
    ):
        approval[field] = 1.0
    approval["motor_temperature_trip_c"] = 80.0
    approval["motor_temperature_reset_c"] = 70.0
    approval["battery_temperature_trip_c"] = 70.0
    approval["battery_temperature_reset_c"] = 60.0
    approval["policy_sha256"] = "a" * 64
    approval["robot_serial"] = "B2-TEST"
    approval["firmware_version"] = "test-firmware"
    approval["reviewer"] = "hardware engineer"
    approval["reviewed_at"] = "2026-08-11T00:00:00+08:00"
    assert hardware_activation_blockers(profile) == ()
    require_hardware_activation_approval(profile)


def test_operator_authorized_simulation_parameters_pass_only_for_bound_policy() -> None:
    policy_sha = "a" * 64
    profile = {
        "schema_version": 1,
        "status": "operator_authorized_experimental",
        "activation_permitted": True,
        "parameter_source": "compiled_simulation_parameters",
        "policy_sha256": policy_sha,
    }

    assert hardware_policy_blockers(profile, policy_sha) == ()
    assert "does not match" in hardware_policy_blockers(profile, "b" * 64)[0]


def test_non_null_but_invalid_values_still_block_activation() -> None:
    profile = load_parameter_profile(PROFILE_PATH)
    profile["status"] = "approved"
    profile["activation_permitted"] = True
    approval = profile["hardware_approval"]
    for field in REQUIRED_HARDWARE_APPROVAL_FIELDS:
        approval[field] = "not-a-reviewed-number"
    blockers = hardware_activation_blockers(profile)
    assert "hardware_approval.position_kp must be a positive 12-vector" in blockers
    assert "hardware_approval.gain_ramp_s must be positive and finite" in blockers
    assert "hardware_approval.motor_temperature_trip_c must be finite" in blockers


@pytest.mark.parametrize("value", ([], {"schema_version": 2}))
def test_loader_rejects_wrong_shape_or_version(tmp_path: Path, value: object) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_parameter_profile(path)
