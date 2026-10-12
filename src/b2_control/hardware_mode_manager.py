"""Own the hardware gateway process for the persistent operator console.

The policy/UI process stays alive while native locomotion owns the robot.  A
CUSTOM request starts the reviewed C++ gateway; a NATIVE request terminates it
normally so its MotionSwitcher exit path restores ``ai``.  This module never
publishes LowCmd itself.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import time
from typing import TextIO

from .control_lifecycle import ControlLifecycle, lifecycle_for_gateway

from .parameter_profile import (
    hardware_policy_blockers,
    load_parameter_profile,
    parameter_profile_sha256,
)


@dataclass(frozen=True)
class GatewayStatus:
    control_mode: str
    detail: str
    gateway_mode: str = ""
    lifecycle_state: str = ControlLifecycle.DISCONNECTED
    output_mode: str = "NATIVE"
    metrics: dict[str, float | int] = field(default_factory=dict)
    fault: dict[str, object] = field(default_factory=dict)
    blackbox_complete: bool = False


class HardwareModeManager:
    """Start, observe, and normally stop one C++ hardware gateway."""

    def __init__(
        self,
        *,
        interface: str,
        gateway_executable: Path,
        mode_probe_executable: Path,
        sdk_library_path: Path,
        diagnostics_csv: Path,
        gateway_log: Path,
        gateway_pid: Path,
        hardware_profile: Path,
        policy_sha256: str,
    ) -> None:
        if not interface or interface == "lo":
            raise ValueError("managed hardware requires a non-loopback interface")
        for label, executable in (
            ("gateway", gateway_executable),
            ("mode probe", mode_probe_executable),
        ):
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ValueError(f"{label} executable is unavailable: {executable}")
        if not sdk_library_path.is_dir():
            raise ValueError(f"SDK library directory is unavailable: {sdk_library_path}")
        self.interface = interface
        self.gateway_executable = gateway_executable.resolve()
        self.mode_probe_executable = mode_probe_executable.resolve()
        self.sdk_library_path = sdk_library_path.resolve()
        self.diagnostics_csv = diagnostics_csv.resolve()
        self.gateway_log = gateway_log.resolve()
        self.gateway_pid = gateway_pid.resolve()
        self.hardware_profile = hardware_profile.resolve()
        self.policy_sha256 = policy_sha256
        self.process: subprocess.Popen[str] | None = None
        self._log_file: TextIO | None = None
        self._last_exit: int | None = None
        self._status = GatewayStatus(
            "UNKNOWN", "尚未检查 MotionSwitcher", lifecycle_state=ControlLifecycle.DISCONNECTED
        )

    def activation_blockers(self) -> tuple[str, ...]:
        try:
            profile = load_parameter_profile(self.hardware_profile)
        except ValueError as exc:
            return (str(exc),)
        return hardware_policy_blockers(profile, self.policy_sha256)

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        current = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            str(self.sdk_library_path)
            if not current
            else f"{self.sdk_library_path}:{current}"
        )
        return environment

    def check_native(self) -> GatewayStatus:
        try:
            result = subprocess.run(
                [str(self.mode_probe_executable), self.interface],
                env=self._environment(),
                text=True,
                capture_output=True,
                timeout=6.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            self._status = GatewayStatus(
                "FAULT", f"MotionSwitcher 检查异常：{error}",
                lifecycle_state=ControlLifecycle.FAULT,
                output_mode="UNKNOWN",
            )
            return self._status
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0 and "name=ai" in output:
            self._status = GatewayStatus(
                "NATIVE", "原生 locomotion 已确认",
                lifecycle_state=ControlLifecycle.NATIVE_CONFIRMED,
                output_mode="NATIVE",
            )
        elif result.returncode == 0 and "name=" in output:
            self._status = GatewayStatus(
                "UNKNOWN", f"原生 ai 未激活：{output}",
                lifecycle_state=ControlLifecycle.FAULT,
                output_mode="UNKNOWN",
            )
        else:
            self._status = GatewayStatus(
                "FAULT", f"MotionSwitcher 检查失败：{output}",
                lifecycle_state=ControlLifecycle.FAULT,
                output_mode="UNKNOWN",
            )
        return self._status

    def recheck_native_after_stopped_gateway(self) -> GatewayStatus:
        """Clear a pre-handover failure only when no gateway is still active.

        A live gateway may be holding or damping the robot and must never be
        hidden by a MotionSwitcher probe.  If the failed process has already
        exited, an authoritative ``name=ai`` response makes a retry safe.
        """

        if self.process is not None and self.process.poll() is None:
            return self._status
        return self.check_native()

    def start_custom(self) -> GatewayStatus:
        if self.process is not None and self.process.poll() is None:
            return self._status
        blockers = self.activation_blockers()
        if blockers:
            self._status = GatewayStatus(
                "NATIVE",
                "自研接管已阻止；安全配置/Policy 未批准：" + "; ".join(blockers),
                lifecycle_state=ControlLifecycle.NATIVE_CONFIRMED,
                output_mode="NATIVE",
            )
            return self._status
        self._close_log()
        self.diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
        self.gateway_log.parent.mkdir(parents=True, exist_ok=True)
        self.gateway_pid.parent.mkdir(parents=True, exist_ok=True)
        self.diagnostics_csv.unlink(missing_ok=True)
        self._log_file = self.gateway_log.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [
                    str(self.gateway_executable),
                    "--hardware",
                    self.interface,
                    "--acknowledge-b2-low-level-control",
                    "--diagnostics-csv",
                    str(self.diagnostics_csv),
                    "--hardware-profile",
                    str(self.hardware_profile),
                    "--hardware-profile-sha256",
                    parameter_profile_sha256(self.hardware_profile),
                    "--policy-sha256",
                    self.policy_sha256,
                ],
                env=self._environment(),
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            self._close_log()
            self._status = GatewayStatus(
                "FAULT", f"无法启动自研 gateway：{error}",
                lifecycle_state=ControlLifecycle.FAULT,
                output_mode="NATIVE",
            )
            return self._status
        self._last_exit = None
        assert self.process.pid is not None
        self.gateway_pid.write_text(f"{self.process.pid}\n", encoding="ascii")
        self._status = GatewayStatus(
            "CUSTOM_STARTING",
            "等待 LowState、零命令 policy target 和站姿门控",
            lifecycle_state=ControlLifecycle.ARMED,
            output_mode="NATIVE",
        )
        return self._status

    def poll(self) -> GatewayStatus:
        if self.process is None:
            return self._status
        return_code = self.process.poll()
        if return_code is not None:
            self._last_exit = return_code
            self.process = None
            self._close_log()
            native = self.check_native()
            if return_code == 0 and native.control_mode == "NATIVE":
                return native
            self._status = GatewayStatus(
                "FAULT",
                f"自研 gateway 已退出（code={return_code}）；{native.detail}",
                lifecycle_state=ControlLifecycle.FAULT,
                output_mode="UNKNOWN",
            )
            return self._status
        row = self._read_gateway_status()
        gateway_mode = row.get("mode", "")
        fault_reason = row.get("fault_reason", "")
        metrics = self._gateway_metrics(row)
        fault = self._fault_detail(row)
        blackbox_complete = Path(
            str(self.diagnostics_csv) + ".blackbox.complete.json"
        ).is_file()
        if gateway_mode == "SAFE_DAMPING":
            self._status = GatewayStatus(
                "FAULT",
                "安全故障已锁存，保持 SAFE_DAMPING；禁止自动恢复原生模式："
                + (fault_reason or "查看 gateway 黑匣子"),
                gateway_mode,
                lifecycle_state=ControlLifecycle.FAULT,
                output_mode="DAMPING",
                metrics=metrics,
                fault=fault,
                blackbox_complete=blackbox_complete,
            )
            return self._status
        if gateway_mode:
            self._status = GatewayStatus(
                "CUSTOM",
                f"自研 LowCmd 已接管：{gateway_mode}",
                gateway_mode,
                lifecycle_state=lifecycle_for_gateway(gateway_mode),
                output_mode=(
                    "DAMPING" if gateway_mode == "SAFE_DAMPING" else "HOLD"
                    if gateway_mode in {"SAFE_HOLD", "SAFE_HOLD_EXIT"}
                    else "LOWCMD"
                ),
                metrics=metrics,
                fault=fault,
                blackbox_complete=blackbox_complete,
            )
        else:
            self._status = GatewayStatus(
                "CUSTOM_STARTING",
                "gateway 运行中，尚未通过接管门控",
                lifecycle_state=ControlLifecycle.ARMED,
                output_mode="NATIVE",
                metrics=metrics,
            )
        return self._status

    def stop_to_native(self) -> GatewayStatus:
        return_code = 0
        if self.process is not None and self.process.poll() is None:
            status = self.poll()
            if status.control_mode == "FAULT":
                raise RuntimeError(status.detail)
            if self.process is None:
                native = self.check_native()
                if native.control_mode == "NATIVE":
                    self.gateway_pid.unlink(missing_ok=True)
                    return native
                raise RuntimeError(native.detail)
            self.process.terminate()
            try:
                return_code = self.process.wait(timeout=12.0)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    "gateway 正常退出超时；未强杀，原生模式尚未确认"
                ) from error
            finally:
                if self.process.poll() is not None:
                    self.process = None
                    self._close_log()
        native = self.check_native()
        if native.control_mode == "NATIVE":
            self.gateway_pid.unlink(missing_ok=True)
            return native
        if return_code != 0:
            raise RuntimeError(
                f"gateway 退出失败：code={return_code}；{native.detail}"
            )
        if native.control_mode != "NATIVE":
            raise RuntimeError(native.detail)
        return native

    def close(self) -> GatewayStatus:
        """Return control to native mode when the console itself exits."""
        return self.stop_to_native()

    @staticmethod
    def _number(row: dict[str, str], key: str, *, integer: bool = False) -> float | int:
        try:
            value = float(row.get(key, "0"))
        except (TypeError, ValueError):
            value = 0.0
        return int(value) if integer else value

    def _gateway_metrics(self, row: dict[str, str]) -> dict[str, float | int]:
        return {
            "dds_lowstate_hz": self._number(row, "dds_lowstate_hz"),
            "crc_errors": self._number(row, "state_crc_errors", integer=True),
            "invalid_states": self._number(row, "invalid_states", integer=True),
            "publish_jitter_ms": self._number(row, "publish_jitter_ms"),
            "publish_jitter_max_ms": self._number(row, "publish_jitter_max_ms"),
            "lowcmd_count": self._number(row, "lowcmd", integer=True),
        }

    def _fault_detail(self, row: dict[str, str]) -> dict[str, object]:
        def nullable_number(key: str) -> float | None:
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                return None
            return value if value == value else None

        return {
            "reason": row.get("fault_reason", ""),
            "metric": row.get("fault_metric", ""),
            "joint": row.get("fault_joint", ""),
            "value": nullable_number("fault_value"),
            "threshold": nullable_number("fault_threshold"),
            "duration_s": nullable_number("fault_duration_s") or 0.0,
        }

    def fault_post_window_remaining_s(self) -> float:
        path = Path(str(self.diagnostics_csv) + ".fault.json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            latched_ns = int(payload["monotonic_ns"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return 30.0
        elapsed = max(0.0, (time.monotonic_ns() - latched_ns) / 1_000_000_000.0)
        return max(0.0, 30.0 - elapsed)

    def _read_gateway_status(self) -> dict[str, str]:
        try:
            with self.diagnostics_csv.open("rb") as stream:
                header = stream.readline().decode("utf-8")
                header_end = stream.tell()
                stream.seek(0, os.SEEK_END)
                end = stream.tell()
                start = max(header_end, end - 65536)
                stream.seek(start)
                tail = stream.read().decode("utf-8", errors="replace")
            lines = [line for line in tail.splitlines() if line.strip()]
            if start > header_end and lines:
                lines = lines[1:]
            if not lines:
                return {}
            last = next(csv.DictReader([header, lines[-1]]))
            return {str(key): str(value or "") for key, value in last.items()}
        except OSError:
            return {}

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


__all__ = ["GatewayStatus", "HardwareModeManager"]
