from pathlib import Path
import json
import subprocess
from unittest.mock import Mock, patch

from b2_control.hardware_mode_manager import HardwareModeManager


def _manager(tmp_path: Path) -> HardwareModeManager:
    gateway = tmp_path / "gateway"
    probe = tmp_path / "probe"
    sdk = tmp_path / "sdk"
    gateway.touch(mode=0o755)
    probe.touch(mode=0o755)
    sdk.mkdir()
    source = (
        Path(__file__).parents[1] / "configs" / "b2-reference-profile.json"
    )
    profile = json.loads(source.read_text(encoding="utf-8"))
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
    for field in approval:
        approval[field] = [1.0] * 12 if field in vector_fields else 1.0
    approval["motor_temperature_trip_c"] = 2.0
    approval["motor_temperature_reset_c"] = 1.0
    approval["battery_temperature_trip_c"] = 2.0
    approval["battery_temperature_reset_c"] = 1.0
    approval["policy_sha256"] = "a" * 64
    approval["robot_serial"] = "B2-TEST"
    approval["firmware_version"] = "test-firmware"
    approval["reviewer"] = "test reviewer"
    approval["reviewed_at"] = "2026-08-14T00:00:00+08:00"
    profile_path = tmp_path / "approved-profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return HardwareModeManager(
        interface="enp8s0",
        gateway_executable=gateway,
        mode_probe_executable=probe,
        sdk_library_path=sdk,
        diagnostics_csv=tmp_path / "gateway.csv",
        gateway_log=tmp_path / "gateway.log",
        gateway_pid=tmp_path / "gateway.pid",
        hardware_profile=profile_path,
        policy_sha256="a" * 64,
    )


def test_check_native_requires_confirmed_ai_output(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="READ-ONLY CheckMode: form=0 name=ai\n", stderr=""
    )
    with patch("b2_control.hardware_mode_manager.subprocess.run", return_value=completed):
        status = manager.check_native()
    assert status.control_mode == "NATIVE"


def test_stopped_pre_handover_failure_can_recheck_confirmed_native(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    manager.process = None
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="READ-ONLY CheckMode: form=0 name=ai\n", stderr=""
    )

    with patch("b2_control.hardware_mode_manager.subprocess.run", return_value=completed):
        status = manager.recheck_native_after_stopped_gateway()

    assert status.control_mode == "NATIVE"


def test_live_gateway_fault_cannot_be_hidden_by_native_recheck(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    process = Mock()
    process.poll.return_value = None
    manager.process = process
    original = manager._status

    with patch.object(manager, "check_native") as check_native:
        status = manager.recheck_native_after_stopped_gateway()

    assert status is original
    check_native.assert_not_called()


def test_gateway_lifecycle_reports_custom_then_restores_native(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    process = Mock()
    process.poll.side_effect = [None, None, None, 0]
    process.wait.return_value = 0
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="READ-ONLY CheckMode: form=0 name=ai\n", stderr=""
    )
    with (
        patch("b2_control.hardware_mode_manager.subprocess.Popen", return_value=process),
        patch("b2_control.hardware_mode_manager.subprocess.run", return_value=completed),
    ):
        starting = manager.start_custom()
        assert starting.control_mode == "CUSTOM_STARTING"
        manager.diagnostics_csv.write_text(
            "elapsed_s,mode\n0.0,POLICY\n", encoding="utf-8"
        )
        custom = manager.poll()
        assert custom.control_mode == "CUSTOM"
        assert custom.gateway_mode == "POLICY"
        native = manager.stop_to_native()
    process.terminate.assert_called_once_with()
    assert native.control_mode == "NATIVE"


def test_gateway_status_exposes_metrics_and_structured_fault(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.process = Mock()
    manager.process.poll.return_value = None
    manager.diagnostics_csv.write_text(
        "mode,fault_reason,fault_metric,fault_joint,fault_value,"
        "fault_threshold,fault_duration_s,dds_lowstate_hz,state_crc_errors,"
        "invalid_states,publish_jitter_ms,publish_jitter_max_ms,lowcmd\n"
        "SAFE_DAMPING,estimated torque threshold,torque,RR_calf,310,300,0.2,"
        "499.4,2,1,0.08,0.7,1000\n",
        encoding="utf-8",
    )

    status = manager.poll()

    assert status.control_mode == "FAULT"
    assert status.output_mode == "DAMPING"
    assert status.metrics["dds_lowstate_hz"] == 499.4
    assert status.metrics["crc_errors"] == 2
    assert status.fault["joint"] == "RR_calf"
    assert status.fault["threshold"] == 300.0
