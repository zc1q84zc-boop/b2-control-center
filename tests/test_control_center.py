import json
from pathlib import Path
from types import SimpleNamespace
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import b2_control.control_center as control_center_module
from b2_control.control_center import (
    ControlCenterPaths,
    ControlCenterSupervisor,
    ControlCenterWeb,
)
from b2_control.policy_registry import PolicyProfile, PolicyRegistry


def _paths(tmp_path: Path) -> ControlCenterPaths:
    return ControlCenterPaths(
        root=tmp_path,
        simulator_script=tmp_path / "sim",
        gateway=tmp_path / "gateway",
        hardware_probe=tmp_path / "probe",
        mode_probe=tmp_path / "mode",
        native_action=tmp_path / "native-action",
        sdk_library=tmp_path / "lib",
        robot_model_dir=tmp_path / "model",
        logs=tmp_path / "logs",
    )


def _static_policy_registry(policy_sha: str = "a" * 64) -> SimpleNamespace:
    policy = PolicyProfile(
        id="current-policy",
        name="Current Policy",
        bundle="/verified/policy-bundle",
        adapter="b2_45x12_v1",
        input_dim=45,
        output_dim=12,
        policy_rate_hz=50,
        sha256=policy_sha,
    )
    return SimpleNamespace(profiles=lambda: [policy], policy=policy)


def _write_profile_choices(tmp_path: Path, policy_sha: str = "a" * 64) -> None:
    profiles = tmp_path / "configs" / "hardware-profiles"
    profiles.mkdir(parents=True)
    (tmp_path / "configs" / "b2-reference-profile.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "reference_only",
                "activation_permitted": False,
            }
        ),
        encoding="utf-8",
    )
    (profiles / "operator-authorized-simulation-runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "operator_authorized_experimental",
                "activation_permitted": True,
                "parameter_source": "compiled_simulation_parameters",
                "policy_sha256": policy_sha,
            }
        ),
        encoding="utf-8",
    )


def _native_lowcmd_probe_output(
    rate_hz: float, *, crc_errors: int = 0, invalid_samples: int = 0
) -> str:
    stable = 350.0 <= rate_hz <= 650.0 and not crc_errors and not invalid_samples
    result = "PASS" if stable else "FAIL"
    return (
        f"{result} samples=1500 crc_errors={crc_errors} "
        f"invalid_samples={invalid_samples} lowcmd_policy=native_500hz "
        f"lowcmd_samples={max(0, int(rate_hz * 3))} "
        f"lowcmd_rate_hz={rate_hz:.3f} lowcmd_native_expected=1 "
        f"lowcmd_native_rate_ok={int(stable)} "
        f"lowcmd_publisher_conflict={int(not stable)}"
    )


def test_passive_startup_does_not_create_processes(tmp_path: Path) -> None:
    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    status = supervisor.snapshot()
    assert status["state"] == "IDLE"
    assert status["hardware_lifecycle"] == "DISCONNECTED"
    assert status["processes"] == {}
    assert status["operator_url"] == ""
    assert status["hardware_capabilities"]["readonly_connected"] is False
    assert status["hardware_capabilities"]["native_actions"] is False
    assert status["hardware_capabilities"]["custom_takeover"] is False

    supervisor.state = "HARDWARE_CONNECTED"
    supervisor.operator_url = "http://127.0.0.1:9999/token/"
    connected = supervisor.snapshot()["hardware_capabilities"]
    assert connected["readonly_connected"] is True
    assert connected["operator_panel"] is True
    assert connected["native_actions"] is True
    assert connected["input_source_switch"] is True
    assert connected["custom_takeover"] is False


def test_multiple_repository_hardware_profiles_are_selectable(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    configs = tmp_path / "configs"
    extra = configs / "hardware-profiles"
    extra.mkdir(parents=True)
    payload = {"schema_version": 1, "status": "reference_only"}
    (configs / "b2-reference-profile.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (extra / "stairs.json").write_text(json.dumps(payload), encoding="utf-8")
    supervisor = ControlCenterSupervisor(
        paths, PolicyRegistry(tmp_path / "policies.json")
    )

    status = supervisor.snapshot()

    assert [profile["id"] for profile in status["hardware_profiles"]] == [
        "b2-reference-profile.json",
        "hardware-profiles/stairs.json",
    ]
    selected_id, selected_path = supervisor._resolve_hardware_profile(
        "hardware-profiles/stairs.json"
    )
    assert selected_id == "hardware-profiles/stairs.json"
    assert selected_path == (extra / "stairs.json").resolve()
    with pytest.raises(ValueError, match="unknown hardware safety profile"):
        supervisor._resolve_hardware_profile("../../outside.json")


def test_restart_and_refresh_default_to_current_policy_approved_profile(
    tmp_path: Path,
) -> None:
    _write_profile_choices(tmp_path)
    paths = _paths(tmp_path)
    registry = _static_policy_registry()

    first_supervisor = ControlCenterSupervisor(paths, registry)  # type: ignore[arg-type]
    first = first_supervisor.snapshot()
    refreshed = first_supervisor.snapshot()
    restarted = ControlCenterSupervisor(paths, registry).snapshot()  # type: ignore[arg-type]

    expected = "hardware-profiles/operator-authorized-simulation-runtime.json"
    assert first["policy_id"] == "current-policy"
    assert first["hardware_profile_id"] == expected
    assert refreshed["hardware_profile_id"] == expected
    assert restarted["hardware_profile_id"] == expected
    assert first["hardware_safety"]["approved"] is True
    assert first["hardware_safety"]["blockers"] == []
    selected = next(
        profile for profile in first["hardware_profiles"] if profile["id"] == expected
    )
    assert selected["policy_approvals"]["current-policy"]["allowed"] is True
    resolved_id, _ = first_supervisor._resolve_hardware_profile(
        "", policy=registry.policy
    )
    assert resolved_id == expected


def test_default_profile_keeps_safe_reference_fallback_without_approval(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "configs" / "b2-reference-profile.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "reference_only",
                "activation_permitted": False,
            }
        ),
        encoding="utf-8",
    )
    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), _static_policy_registry()  # type: ignore[arg-type]
    )

    status = supervisor.snapshot()

    assert status["hardware_profile_id"] == "b2-reference-profile.json"
    assert status["hardware_safety"]["approved"] is False
    assert status["hardware_safety"]["blockers"]


def test_explicit_hardware_profile_selection_is_not_overwritten(
    tmp_path: Path,
) -> None:
    _write_profile_choices(tmp_path)
    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), _static_policy_registry()  # type: ignore[arg-type]
    )
    supervisor.hardware_profile_id = "b2-reference-profile.json"

    first = supervisor.snapshot()
    refreshed = supervisor.snapshot()

    assert first["hardware_profile_id"] == "b2-reference-profile.json"
    assert refreshed["hardware_profile_id"] == "b2-reference-profile.json"
    assert first["hardware_safety"]["approved"] is False


def test_successful_readonly_hardware_connection_unlocks_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    profile = tmp_path / "configs" / "b2-reference-profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        json.dumps({"schema_version": 1, "status": "reference_only"}),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    readonly_commands: list[list[str]] = []

    class FakeProcess:
        def __init__(self, _name: str, command: list[str], **_kwargs: object) -> None:
            self.command = command
            commands.append(command)

        def start(self) -> None:
            return

        def wait_for_text(self, _prefix: str, _timeout: float) -> str:
            return "B2 operator UI: http://127.0.0.1:9999/token/"

        def snapshot(self) -> dict[str, object]:
            return {"running": True, "command": self.command}

    monkeypatch.setattr(control_center_module, "ManagedProcess", FakeProcess)
    supervisor = ControlCenterSupervisor(
        paths, PolicyRegistry(tmp_path / "policies.json")
    )
    def fake_readonly_probe(command: list[str], **_kwargs: object) -> str:
        readonly_commands.append(command)
        if command[0] == str(paths.mode_probe):
            return "name=ai"
        return _native_lowcmd_probe_output(499.3)

    supervisor._run_readonly_probe = fake_readonly_probe  # type: ignore[method-assign]

    supervisor._connect_hardware_worker(
        SimpleNamespace(bundle=str(tmp_path / "policy-bundle")),  # type: ignore[arg-type]
        profile,
        "enp8s0",
        "keyboard",
        0.6,
        0.25,
        0.1,
    )

    status = supervisor.snapshot()
    assert status["state"] == "HARDWARE_CONNECTED"
    assert status["hardware_lifecycle"] == "NATIVE_CONFIRMED"
    assert status["operator_url"] == "http://127.0.0.1:9999/token/"
    assert status["hardware_capabilities"]["native_actions"] is True
    assert status["hardware_capabilities"]["input_source_switch"] is True
    assert "--hardware-profile" in commands[0]
    assert str(profile) in commands[0]
    assert "--operator-event-log" in commands[0]
    assert "--test-results" in commands[0]
    assert readonly_commands[0] == [str(paths.mode_probe), "enp8s0"]
    assert readonly_commands[1] == [
        str(paths.hardware_probe),
        "enp8s0",
        "3",
        "--expect-native-lowcmd",
    ]


def test_mode_must_confirm_ai_before_native_lowcmd_probe(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    profile = tmp_path / "configs" / "b2-reference-profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        json.dumps({"schema_version": 1, "status": "reference_only"}),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    supervisor = ControlCenterSupervisor(
        paths, PolicyRegistry(tmp_path / "policies.json")
    )

    def reject_mode(command: list[str], **_kwargs: object) -> str:
        commands.append(command)
        return "name=custom"

    supervisor._run_readonly_probe = reject_mode  # type: ignore[method-assign]
    supervisor._connect_hardware_worker(
        SimpleNamespace(bundle=str(tmp_path / "policy-bundle")),  # type: ignore[arg-type]
        profile,
        "enp8s0",
        "keyboard",
        0.6,
        0.25,
        0.1,
    )

    assert commands == [[str(paths.mode_probe), "enp8s0"]]
    assert supervisor.state == "FAULT"
    assert supervisor.hardware_lifecycle.value == "FAULT"


def test_native_lowcmd_rate_conflict_is_rejected_after_ai_confirmation(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    profile = tmp_path / "configs" / "b2-reference-profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        json.dumps({"schema_version": 1, "status": "reference_only"}),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    supervisor = ControlCenterSupervisor(
        paths, PolicyRegistry(tmp_path / "policies.json")
    )

    def conflict_probe(command: list[str], **_kwargs: object) -> str:
        commands.append(command)
        if command[0] == str(paths.mode_probe):
            return "name=ai"
        return _native_lowcmd_probe_output(800.0)

    supervisor._run_readonly_probe = conflict_probe  # type: ignore[method-assign]
    supervisor._connect_hardware_worker(
        SimpleNamespace(bundle=str(tmp_path / "policy-bundle")),  # type: ignore[arg-type]
        profile,
        "enp8s0",
        "keyboard",
        0.6,
        0.25,
        0.1,
    )

    assert len(commands) == 2
    assert commands[1][-1] == "--expect-native-lowcmd"
    assert supervisor.state == "FAULT"
    assert "exceeds the 650 Hz ceiling" in supervisor.detail


def test_native_lowcmd_settling_zero_then_low_then_stable_passes(
    tmp_path: Path,
) -> None:
    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    session = tmp_path / "session"
    session.mkdir()
    responses = iter(
        [
            _native_lowcmd_probe_output(0.0),
            _native_lowcmd_probe_output(73.0),
            _native_lowcmd_probe_output(500.0),
        ]
    )
    commands: list[list[str]] = []

    def fake_probe(command: list[str], **_kwargs: object) -> str:
        commands.append(command)
        output = next(responses)
        if output.startswith("FAIL "):
            raise RuntimeError(output)
        return output

    supervisor._run_readonly_probe = fake_probe  # type: ignore[method-assign]

    result = supervisor._wait_for_stable_native_lowcmd(
        interface="enp8s0", session_dir=session, environment={}
    )

    assert result.startswith("PASS ")
    assert len(commands) == 3
    assert all(command[-1] == "--expect-native-lowcmd" for command in commands)
    log = (session / "readonly-lowstate.log").read_text(encoding="utf-8")
    assert log.count("native LowCmd window") == 3
    assert "lowcmd_rate_hz=0.000" in log
    assert "lowcmd_rate_hz=73.000" in log
    assert "lowcmd_rate_hz=500.000" in log
    assert sum("settling window" in event for event in supervisor.events) == 2
    assert any("原生恢复中：第1/5次" in event for event in supervisor.events)
    assert any("原生恢复中：第3/5次" in event for event in supervisor.events)
    assert any("stabilized on window 3/5" in event for event in supervisor.events)


def test_native_lowcmd_settling_persistent_zero_fails_after_five_windows(
    tmp_path: Path,
) -> None:
    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    session = tmp_path / "session"
    session.mkdir()
    calls = 0

    def zero_probe(_command: list[str], **_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return _native_lowcmd_probe_output(0.0)

    supervisor._run_readonly_probe = zero_probe  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="bounded 15 s settling period"):
        supervisor._wait_for_stable_native_lowcmd(
            interface="enp8s0", session_dir=session, environment={}
        )

    assert calls == 5


def test_native_lowcmd_800_hz_is_rejected_without_retry(tmp_path: Path) -> None:
    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    session = tmp_path / "session"
    session.mkdir()
    calls = 0

    def high_probe(_command: list[str], **_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return _native_lowcmd_probe_output(800.0)

    supervisor._run_readonly_probe = high_probe  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="exceeds the 650 Hz ceiling"):
        supervisor._wait_for_stable_native_lowcmd(
            interface="enp8s0", session_dir=session, environment={}
        )

    assert calls == 1


def test_native_lowcmd_crc_error_is_rejected_without_retry(tmp_path: Path) -> None:
    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    session = tmp_path / "session"
    session.mkdir()
    calls = 0

    def crc_probe(_command: list[str], **_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return _native_lowcmd_probe_output(0.0, crc_errors=1)

    supervisor._run_readonly_probe = crc_probe  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="data-integrity failure"):
        supervisor._wait_for_stable_native_lowcmd(
            interface="enp8s0", session_dir=session, environment={}
        )

    assert calls == 1


def test_hardware_connect_rejects_live_local_orphan_gateway(
    tmp_path: Path,
) -> None:
    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    supervisor._scan_orphan_hardware_gateways = (  # type: ignore[method-assign]
        lambda _gateway: [
            {"pid": 4321, "interface": "enp8s0", "command": "gateway --hardware"}
        ]
    )

    with pytest.raises(ValueError, match="遗留真机 gateway PID 4321"):
        supervisor.connect_hardware({"confirmed": True})

    assert supervisor.state == "IDLE"
    assert supervisor.orphan_hardware_gateways[0]["pid"] == 4321


def test_hardware_connect_rejects_down_link_before_session_camera_or_probe(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    supervisor = ControlCenterSupervisor(
        paths, PolicyRegistry(tmp_path / "policies.json")
    )
    calls: list[str] = []
    supervisor._scan_orphan_hardware_gateways = (  # type: ignore[method-assign]
        lambda _gateway: []
    )
    supervisor._scan_ros2_domain0_participants = (  # type: ignore[method-assign]
        lambda: []
    )
    supervisor._network_interfaces = lambda: [  # type: ignore[method-assign]
        {"index": 2, "name": "enp8s0", "state": "down", "address": ""}
    ]
    supervisor._session_paths = (  # type: ignore[method-assign]
        lambda _prefix: calls.append("session")
    )
    supervisor._start_hardware_camera = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: calls.append("camera")
    )
    supervisor._run_readonly_probe = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: calls.append("probe")
    )
    supervisor._connect_hardware_worker = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: calls.append("worker")
    )

    with pytest.raises(ValueError, match="物理链路未连接/网卡DOWN"):
        supervisor.connect_hardware(
            {
                "confirmed": True,
                "operator_present": True,
                "area_clear": True,
                "robot_supported": True,
                "physical_estop_ready": True,
                "command_source_zero": True,
                "policy_id": "current-policy",
                "interface": "enp8s0",
            }
        )

    assert calls == []
    assert supervisor.state == "IDLE"
    assert supervisor.hardware_lifecycle.value == "DISCONNECTED"
    assert supervisor.session_id == ""
    assert supervisor.session_dir == ""
    assert supervisor.processes == {}
    assert supervisor._worker is None
    assert not paths.logs.exists()


def test_startup_scan_reports_only_exact_hardware_gateway(tmp_path: Path) -> None:
    gateway = tmp_path / "build" / "gateway" / "b2_policy_controller"
    proc = tmp_path / "proc"
    (proc / "101").mkdir(parents=True)
    (proc / "102").mkdir()
    (proc / "101" / "cmdline").write_bytes(
        (str(gateway.resolve()) + "\0--hardware\0enp8s0\0").encode()
    )
    (proc / "102" / "cmdline").write_bytes(
        (str(gateway.resolve()) + "\0--sim-hardware-handover\0").encode()
    )

    matches = ControlCenterSupervisor._scan_orphan_hardware_gateways(
        gateway, proc_root=proc
    )

    assert matches == [
        {
            "pid": 101,
            "interface": "enp8s0",
            "command": f"{gateway.resolve()} --hardware enp8s0",
        }
    ]


def test_ros2_domain0_participant_scan_warns_but_ignores_localhost_only(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    (proc / "101").mkdir(parents=True)
    (proc / "102").mkdir()
    for pid, localhost_only in (("101", False), ("102", True)):
        root = proc / pid
        (root / "cmdline").write_bytes(
            b"/usr/bin/python3\0-c\0from ros2cli.daemon.daemonize import main\0"
            b"--ros-domain-id\0"
            b"0\0--rmw-implementation\0rmw_fastrtps_cpp\0"
        )
        environment = b"ROS_DISTRO=humble\0ROS_DOMAIN_ID=0\0"
        if localhost_only:
            environment += b"ROS_LOCALHOST_ONLY=1\0"
        (root / "environ").write_bytes(environment)
        (root / "maps").write_text(
            "/opt/ros/humble/lib/librmw_fastrtps_cpp.so\n", encoding="utf-8"
        )

    matches = ControlCenterSupervisor._scan_ros2_domain0_participants(
        proc_root=proc
    )

    assert [item["pid"] for item in matches] == [101]
    assert matches[0]["domain_id"] == 0
    assert matches[0]["rmw"] == "rmw_fastrtps_cpp"


def test_native_move_is_bounded_and_persistently_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    session = tmp_path / "logs" / "hardware-test"
    session.mkdir(parents=True)
    supervisor.state = "HARDWARE_CONNECTED"
    supervisor.interface = "enp8s0"
    supervisor.session_dir = str(session)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="OK", stderr="")

    monkeypatch.setattr(control_center_module.subprocess, "run", fake_run)

    supervisor.run_native_action(
        {
            "action": "move",
            "vx": 0.6,
            "vy": -0.2,
            "wz": 0.3,
            "confirmed": True,
        }
    )

    assert commands[0][1:3] == ["enp8s0", "move"]
    assert commands[0][commands[0].index("--vx") + 1] == "0.6"
    assert commands[0][commands[0].index("--vy") + 1] == "-0.2"
    assert commands[0][commands[0].index("--wz") + 1] == "0.3"
    records = [
        json.loads(line)
        for line in (session / "operator-actions.jsonl").read_text().splitlines()
    ]
    assert records[-1]["result"] == "completed"
    assert records[-1]["parameters"] == {"vx": 0.6, "vy": -0.2, "wz": 0.3}
    control_events = [
        json.loads(line)
        for line in (session / "control-center-events.jsonl").read_text().splitlines()
    ]
    assert control_events[-1]["event"].startswith("native action completed: move")
    with pytest.raises(ValueError, match="vx must be in"):
        supervisor.run_native_action(
            {"action": "move", "vx": 1.01, "vy": 0, "wz": 0, "confirmed": True}
        )


def test_blackbox_recovery_waits_for_complete_marker(tmp_path: Path) -> None:
    session = tmp_path / "hardware-session"
    session.mkdir()
    (session / "gateway.csv.fault.json").write_text(
        json.dumps({"monotonic_ns": time.monotonic_ns()}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="剩余约"):
        ControlCenterSupervisor._assert_blackbox_capture_complete(session)

    (session / "gateway.csv.blackbox.complete.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    ControlCenterSupervisor._assert_blackbox_capture_complete(session)


def test_embedded_safe_damping_fault_becomes_recoverable_outer_session(
    tmp_path: Path,
) -> None:
    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    session = tmp_path / "logs" / "hardware-20260814-175118"
    session.mkdir(parents=True)
    (session / "gateway.csv.fault.json").write_text(
        json.dumps({"monotonic_ns": time.monotonic_ns()}), encoding="utf-8"
    )
    (session / "gateway.csv.blackbox.complete.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (session / "gateway.pid").write_text("999999999\n", encoding="ascii")
    supervisor.state = "HARDWARE_CONNECTED"
    supervisor.hardware_lifecycle = control_center_module.ControlLifecycle.HOLD
    supervisor.hardware_output_mode = "HOLD"
    supervisor.session_id = "hardware-20260814-175118"
    supervisor.session_dir = str(session)
    supervisor.interface = "enp8s0"
    supervisor.operator_url = "http://127.0.0.1:9999/token/"
    supervisor._operator_api_request = lambda _action, _payload=None: {  # type: ignore[method-assign]
        "control_mode": "FAULT",
        "lifecycle_state": "FAULT",
        "output_mode": "DAMPING",
        "mode_detail": "policy command timeout; SAFE_DAMPING",
    }

    status = supervisor.snapshot()

    assert status["state"] == "FAULT"
    assert status["hardware_lifecycle"] == "FAULT"
    assert status["hardware_output_mode"] == "DAMPING"
    assert status["hardware_capabilities"]["fault_recovery"] is True
    assert "policy command timeout" in status["detail"]
    with pytest.raises(ValueError, match="无法验证 SAFE_DAMPING gateway") as error:
        supervisor.recover_latched_fault(
            {
                "confirmed": True,
                "operator_present": True,
                "robot_supported": True,
                "physical_estop_ready": True,
                "fault_inspected": True,
            }
        )
    assert "没有可恢复的当前真机故障会话" not in str(error.value)


def test_historical_policy_fault_exit_does_not_override_restored_native(
    tmp_path: Path,
) -> None:
    class FakeProcess:
        return_code = 2

        def stop(self, **_kwargs: object) -> None:
            return

    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    supervisor.state = "STOPPING"
    supervisor.hardware_lifecycle = control_center_module.ControlLifecycle.STOPPING
    supervisor.hardware_output_mode = "NATIVE"
    supervisor.session_id = "hardware-20260814-175118"
    supervisor.session_dir = str(tmp_path / supervisor.session_id)
    supervisor.interface = "enp8s0"
    supervisor.processes = {"policy": FakeProcess()}  # type: ignore[dict-item]
    supervisor._generate_session_report = lambda: None  # type: ignore[method-assign]

    supervisor._stop_all_worker(hardware=True, final_state="IDLE")

    assert supervisor.state == "IDLE"
    assert supervisor.hardware_lifecycle.value == "NATIVE"
    assert supervisor.hardware_output_mode == "NATIVE"
    assert "DAMPING_OR_UNKNOWN" not in supervisor.detail
    assert any("historical nonzero exit code" in event for event in supervisor.events)


def test_live_or_unconfirmed_damping_fault_is_not_hidden_on_stop(
    tmp_path: Path,
) -> None:
    class FakeProcess:
        return_code = 2

        def stop(self, **_kwargs: object) -> None:
            return

    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    supervisor.state = "STOPPING"
    supervisor.hardware_lifecycle = control_center_module.ControlLifecycle.FAULT
    supervisor.hardware_output_mode = "DAMPING"
    supervisor.session_id = "hardware-20260814-175118"
    supervisor.session_dir = str(tmp_path / supervisor.session_id)
    supervisor.interface = "enp8s0"
    supervisor.processes = {"policy": FakeProcess()}  # type: ignore[dict-item]

    supervisor._stop_all_worker(hardware=True, final_state="IDLE")

    assert supervisor.state == "FAULT"
    assert supervisor.hardware_lifecycle.value == "FAULT"
    assert supervisor.hardware_output_mode == "DAMPING"
    assert "SAFE_DAMPING" in supervisor.detail


def test_simulation_shutdown_freezes_video_before_lowcmd_and_policy(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class FakeProcess:
        return_code = 0

        def request_stop(self, **_kwargs: object) -> None:
            events.append("simulator-request")

        def stop(self, **_kwargs: object) -> None:
            events.append(self.name + "-stop")

        def wait_stopped(self, **_kwargs: object) -> None:
            events.append("simulator-wait")

        def __init__(self, name: str) -> None:
            self.name = name

    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    supervisor.processes = {
        "simulator": FakeProcess("simulator"),
        "gateway": FakeProcess("gateway"),
        "policy": FakeProcess("policy"),
    }
    supervisor._generate_session_report = lambda: events.append("report")  # type: ignore[method-assign]

    supervisor._stop_all_worker(hardware=False, final_state="IDLE")

    assert events == [
        "simulator-request",
        "gateway-stop",
        "simulator-wait",
        "policy-stop",
        "report",
    ]
    assert supervisor.state == "IDLE"


def test_web_ui_exposes_scenes_and_requires_confirmed_hardware_request(
    tmp_path: Path,
) -> None:
    supervisor = ControlCenterSupervisor(
        _paths(tmp_path), PolicyRegistry(tmp_path / "policies.json")
    )
    web = ControlCenterWeb(supervisor, port=0)
    url = web.open(open_browser=False)
    try:
        with urlopen(url, timeout=2) as response:
            page = response.read().decode()
        assert "B2 Sim2Real Control Center" in page
        assert 'id="nativePage"' in page
        assert 'id="customPage"' in page
        assert 'id="hardwareOutputMode"' in page
        assert 'data-native="restore-ai"' in page
        assert 'id="nativeMove"' in page
        assert "simulation/timed-test" in page
        assert "站立承重 10 秒" in page
        assert "自动运行、归零、停止并保存录像、CSV和JSON结果" in page
        assert "一键连接机器狗（只读检查）" in page
        assert "--expect-native-lowcmd" in page
        assert "最终必须获得完整350–650 Hz稳定窗口" in page
        assert "总速率包络不构成 DDS writer 来源确证" in page
        assert "高频、CRC/invalid、总速率异常或本机遗留 gateway 会立即拒绝" in page
        assert 'id="interfaceStateHint"' in page
        assert "物理链路未连接/网卡 DOWN" in page
        assert "||!linkUp" in page
        assert "lowcmd_samples=0" not in page
        assert "手持遥控器" in page
        assert "网页和 MuJoCo 原生窗口都接收" in page
        assert "postMessage({type:'b2-key'" in page
        assert "const c=e.code||''" in page
        assert "},true);addEventListener('keyup'" in page
        assert 'id="keyboardSource"' in page
        assert 'id="remoteSource"' in page
        assert 'id="hardwareProfile"' in page
        assert "Hardware Operation Readiness" in page
        assert "renderCapabilities(s.hardware_capabilities)" in page
        assert 'type="checkbox"' not in page
        assert "operator_present:true" in page
        assert "fault_inspected:true" in page
        assert "selectInputSource('keyboard')" in page
        assert 'id="keyboardCapture"' not in page
        status_url = f"http://127.0.0.1:{web.server.server_address[1]}/api/{web.token}/status"
        with urlopen(status_url, timeout=2) as response:
            status = json.load(response)
        assert {item["id"] for item in status["terrains"]} >= {
            "grass",
            "mud",
            "gravel",
            "stairs_15cm",
        }

        request = Request(
            f"http://127.0.0.1:{web.server.server_address[1]}"
            f"/api/{web.token}/hardware/connect",
            data=json.dumps({"confirmed": False}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2)
        assert error.value.code == 400
        assert supervisor.state == "IDLE"
    finally:
        web.close()


def test_web_ui_serves_only_managed_artifacts(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    artifact = paths.logs / "session" / "tracking.svg"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("<svg>ok</svg>", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    supervisor = ControlCenterSupervisor(
        paths, PolicyRegistry(tmp_path / "policies.json")
    )
    web = ControlCenterWeb(supervisor, port=0)
    web.open(open_browser=False)
    base = f"http://127.0.0.1:{web.server.server_address[1]}/artifact/{web.token}"
    try:
        with urlopen(f"{base}?path={artifact}", timeout=2) as response:
            assert response.read() == b"<svg>ok</svg>"
            assert response.headers["Content-Type"] == "image/svg+xml"
        with pytest.raises(HTTPError) as error:
            urlopen(f"{base}?path={outside}", timeout=2)
        assert error.value.code == 400
    finally:
        web.close()
