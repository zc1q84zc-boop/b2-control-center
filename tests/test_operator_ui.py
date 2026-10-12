import json
import socket
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from b2_control.interactive_command import InteractiveVelocityCommand
from b2_control.operator_ui import OperatorWebUI


@pytest.fixture
def operator_ui():
    command = InteractiveVelocityCommand(
        yaw_step=0.10,
        max_linear_speed=0.90,
        max_yaw_rate=0.40,
        max_lateral_speed=0.25,
    )
    ui = OperatorWebUI(command, port=0)
    ui.open(open_browser=False)
    try:
        yield ui, command
    finally:
        ui.close()


def test_page_contains_hold_controls_and_speed_presets(operator_ui) -> None:
    ui, _ = operator_ui
    with urlopen(ui.url, timeout=2) as response:
        page = response.read().decode("utf-8")
    assert "Unitree B2 Policy 操作台" in page
    assert 'data-hold="w"' in page
    assert 'data-hold="a"' in page
    assert "按住前进/后退，松开停止" in page
    assert 'data-speed="1"' in page
    assert 'data-speed="9"' in page
    assert "0.1–0.9 m/s" in page
    assert "停止运动 / LEARNED HOLD" in page
    assert "原生 locomotion" in page
    assert "自研 locomotion" in page
    assert "e.data.type!=='b2-key'" in page
    assert "const c=e.code||''" in page
    assert "},true);" in page
    assert "接管生命周期与当前真实输出" in page
    assert "控制与通信性能" in page
    assert "500 Hz 当前抖动" in page
    assert "Policy 端到端延迟" in page
    assert "故障原因" in page
    assert 'data-test="stand-10s"' in page
    assert 'data-test="forward-2s"' in page
    assert 'data-test="stairs"' in page


def test_http_key_command_updates_shared_policy_command(operator_ui) -> None:
    ui, command = operator_ui
    request = Request(
        f"http://127.0.0.1:{ui._server.server_address[1]}/api/{ui.token}/key",
        data=json.dumps({"key": "w", "action": "press"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=2) as response:
        payload = json.load(response)
    assert payload["command"]["vx"] == pytest.approx(0.9)
    assert payload["command_profile"]["forward_speed"] == pytest.approx(0.9)
    assert float(command.snapshot()[0]) == pytest.approx(0.9)

    release = Request(
        f"http://127.0.0.1:{ui._server.server_address[1]}/api/{ui.token}/key",
        data=json.dumps({"key": "w", "action": "release"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(release, timeout=2) as response:
        released = json.load(response)
    assert released["command"]["vx"] == 0.0
    assert float(command.snapshot()[0]) == 0.0

    ui.update(
        phase="POLICY ACTIVE",
        gateway_connected=True,
        state_sequence=123,
        inference_ms=0.7,
        tracking_error_rad=0.12,
    )
    status_url = (
        f"http://127.0.0.1:{ui._server.server_address[1]}"
        f"/api/{ui.token}/status"
    )
    with urlopen(status_url, timeout=2) as response:
        status = json.load(response)
    assert status["phase"] == "POLICY ACTIVE"
    assert status["gateway_connected"] is True
    assert status["state_sequence"] == 123


def test_http_digit_selects_speed_without_triggering_motion(operator_ui) -> None:
    ui, command = operator_ui
    speed = Request(
        f"http://127.0.0.1:{ui._server.server_address[1]}/api/{ui.token}/key",
        data=json.dumps({"key": "4"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(speed, timeout=2) as response:
        selected = json.load(response)
    assert selected["command_profile"]["forward_speed"] == pytest.approx(0.4)
    assert selected["command"]["vx"] == 0.0
    assert command.snapshot().tolist() == [0.0, 0.0, 0.0]


def test_mujoco_udp_keyboard_press_and_release() -> None:
    command = InteractiveVelocityCommand(
        yaw_step=0.10,
        max_linear_speed=0.90,
        max_yaw_rate=0.40,
        max_lateral_speed=0.25,
        initial_linear_speed=0.4,
    )
    ui = OperatorWebUI(
        command,
        port=0,
        native_keyboard_port=0,
        input_source="KEYBOARD (WEB + MUJOCO)",
    )
    ui.open(open_browser=False)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        assert ui.native_keyboard_port is not None
        target = ("127.0.0.1", ui.native_keyboard_port)
        sender.sendto(b"wp", target)
        for _ in range(50):
            if command.snapshot()[0] == pytest.approx(0.4):
                break
            time.sleep(0.01)
        assert command.snapshot()[0] == pytest.approx(0.4)
        assert ui.snapshot()["last_event"].startswith("MUJOCO forward held")
        sender.sendto(b"wr", target)
        for _ in range(50):
            if command.snapshot()[0] == pytest.approx(0.0):
                break
            time.sleep(0.01)
        assert command.snapshot()[0] == pytest.approx(0.0)
        assert ui.snapshot()["last_event"].startswith("MUJOCO W released")
    finally:
        sender.close()
        ui.close()


def test_zero_button_clears_all_axes(operator_ui) -> None:
    ui, command = operator_ui
    for key in ("w", "a", "q", "x"):
        request = Request(
            f"http://127.0.0.1:{ui._server.server_address[1]}/api/{ui.token}/key",
            data=json.dumps({"key": key}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2):
            pass
    assert command.snapshot().tolist() == [0.0, 0.0, 0.0]


def test_confirmed_mode_request_zeros_command_and_is_consumable() -> None:
    command = InteractiveVelocityCommand(
        yaw_step=0.10,
        max_linear_speed=0.90,
        max_yaw_rate=0.40,
        max_lateral_speed=0.25,
    )
    command.apply_key("w")
    ui = OperatorWebUI(command, port=0, mode_switch_enabled=True)
    ui.open(open_browser=False)
    try:
        request = Request(
            f"http://127.0.0.1:{ui._server.server_address[1]}"
            f"/api/{ui.token}/mode",
            data=json.dumps({"mode": "custom", "confirmed": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.load(response)
        assert payload["command"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        assert payload["mode_transition"] is True
        assert payload["motion_commands_enabled"] is False
        assert ui.consume_mode_request() == "CUSTOM"
        assert ui.consume_mode_request() is None
    finally:
        ui.close()


def test_motion_command_is_rejected_while_native() -> None:
    command = InteractiveVelocityCommand(
        yaw_step=0.10,
        max_linear_speed=0.90,
        max_yaw_rate=0.40,
        max_lateral_speed=0.25,
    )
    ui = OperatorWebUI(command, port=0, mode_switch_enabled=True)
    ui.update(control_mode="NATIVE", motion_commands_enabled=False)
    ui.open(open_browser=False)
    try:
        request = Request(
            f"http://127.0.0.1:{ui._server.server_address[1]}"
            f"/api/{ui.token}/key",
            data=json.dumps({"key": "w", "action": "press"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2)
        assert error.value.code == 400
        assert command.snapshot().tolist() == [0.0, 0.0, 0.0]
    finally:
        ui.close()


def test_remote_command_can_be_monitored_without_enabling_web_motion() -> None:
    command = InteractiveVelocityCommand(
        yaw_step=0.10,
        max_linear_speed=0.90,
        max_yaw_rate=0.40,
        max_lateral_speed=0.25,
    )
    ui = OperatorWebUI(command, port=0, input_source="UNITREE HAND REMOTE")
    ui.update(
        display_command=np.asarray([0.6, -0.1, 0.2], dtype=np.float32),
        motion_commands_enabled=False,
    )
    payload = ui.snapshot()
    assert payload["input_source"] == "UNITREE HAND REMOTE"
    assert payload["command"] == pytest.approx({"vx": 0.6, "vy": -0.1, "wz": 0.2})
    assert command.snapshot().tolist() == [0.0, 0.0, 0.0]


def test_lifecycle_metrics_and_structured_fault_are_exposed(operator_ui) -> None:
    ui, _ = operator_ui
    ui.update(
        lifecycle_state="WALK",
        output_mode="HOLD",
        telemetry={
            "crc_errors": 3,
            "dds_hz": 499.8,
            "control_jitter_ms": 0.04,
            "control_jitter_max_ms": 0.31,
            "policy_e2e_ms": 4.2,
            "cpu_percent": 17.5,
            "network_packet_loss_percent": 0.02,
        },
        fault={
            "joint": "RR_calf",
            "metric": "tracking_error",
            "value": 0.52,
            "threshold": 0.45,
            "duration_s": 0.12,
            "output_mode": "HOLD",
        },
    )
    snapshot = ui.snapshot()
    assert snapshot["lifecycle_state"] == "WALK"
    assert snapshot["output_mode"] == "HOLD"
    assert snapshot["telemetry"]["dds_hz"] == pytest.approx(499.8)
    assert snapshot["fault"] == pytest.approx(
        {
            "joint": "RR_calf",
            "metric": "tracking_error",
            "value": 0.52,
            "threshold": 0.45,
            "duration_s": 0.12,
            "output_mode": "HOLD",
        }
    )


def test_timed_test_requires_confirmation_and_motion_enable(operator_ui) -> None:
    ui, _ = operator_ui
    endpoint = (
        f"http://127.0.0.1:{ui._server.server_address[1]}"
        f"/api/{ui.token}/test"
    )
    unconfirmed = Request(
        endpoint,
        data=json.dumps({"template": "stand-10s"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(unconfirmed, timeout=2)
    assert error.value.code == 400

    ui.update(motion_commands_enabled=False)
    disabled = Request(
        endpoint,
        data=json.dumps({"template": "stand-10s", "confirmed": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(disabled, timeout=2)
    assert error.value.code == 400
    assert ui.consume_test_request() is None


def test_confirmed_timed_test_is_consumable_and_logged(tmp_path) -> None:
    command = InteractiveVelocityCommand(
        yaw_step=0.10,
        max_linear_speed=0.90,
        max_yaw_rate=0.40,
        max_lateral_speed=0.25,
        initial_linear_speed=0.6,
    )
    event_log = tmp_path / "operator-events.jsonl"
    ui = OperatorWebUI(command, port=0, event_log_path=event_log)
    ui.open(open_browser=False)
    try:
        key = Request(
            f"http://127.0.0.1:{ui._server.server_address[1]}"
            f"/api/{ui.token}/key",
            data=json.dumps({"key": "w", "action": "press"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(key, timeout=2):
            pass
        request = Request(
            f"http://127.0.0.1:{ui._server.server_address[1]}"
            f"/api/{ui.token}/test",
            data=json.dumps(
                {"template": "forward-2s", "confirmed": True}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.load(response)
        assert payload["command"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        assert ui.consume_test_request() == "forward-2s"
        assert ui.consume_test_request() is None
    finally:
        ui.close()

    events = [json.loads(line) for line in event_log.read_text().splitlines()]
    assert [item["event"] for item in events] == [
        "keyboard",
        "test",
        "test_consumed",
    ]
    assert events[1]["vx"] == pytest.approx(0.6)
