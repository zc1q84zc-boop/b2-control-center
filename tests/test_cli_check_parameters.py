from __future__ import annotations

import json
from pathlib import Path

from b2_control.cli_check_parameters import main
from b2_control.parameter_profile import REQUIRED_HARDWARE_APPROVAL_FIELDS


PROFILE_PATH = Path(__file__).parents[1] / "configs" / "b2-reference-profile.json"


def test_reference_profile_reports_every_blocker(capsys) -> None:
    result = main([str(PROFILE_PATH)])
    output = capsys.readouterr().out
    assert result == 2
    assert "BLOCKED:" in output
    assert "status is not approved" in output
    assert "hardware_approval.position_kp is missing" in output


def test_invalid_file_is_blocked(tmp_path: Path, capsys) -> None:
    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")
    assert main([str(path)]) == 2
    assert "BLOCKED: cannot load parameter profile" in capsys.readouterr().out


def test_complete_profile_passes(tmp_path: Path, capsys) -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    profile["status"] = "approved"
    profile["activation_permitted"] = True
    approval = profile["hardware_approval"]
    vector_fields = {
        "position_kp",
        "velocity_kd",
        "target_slew_rad_s",
        "soft_limit_margin_rad",
        "torque_limit",
        "current_limit",
        "joint_velocity_trip_rad_s",
        "tracking_error_trip_rad",
    }
    for field in REQUIRED_HARDWARE_APPROVAL_FIELDS:
        approval[field] = [1.0] * 12 if field in vector_fields else 1.0
    approval["motor_temperature_trip_c"] = 2.0
    approval["motor_temperature_reset_c"] = 1.0
    approval["battery_temperature_trip_c"] = 2.0
    approval["battery_temperature_reset_c"] = 1.0
    approval["policy_sha256"] = "a" * 64
    approval["robot_serial"] = "B2-TEST"
    approval["firmware_version"] = "test-firmware"
    approval["reviewer"] = "hardware engineer"
    approval["reviewed_at"] = "review timestamp"
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    assert main([str(path)]) == 0
    assert capsys.readouterr().out == (
        "PASS: hardware parameter profile is complete and explicitly approved\n"
    )
