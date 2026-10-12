"""Independent local web control center for B2 simulation and hardware tests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import Request, urlopen
import webbrowser

from .control_lifecycle import ControlLifecycle
from .policy_registry import PolicyProfile, PolicyRegistry
from .parameter_profile import (
    hardware_policy_blockers,
    load_parameter_profile,
    parameter_profile_sha256,
)
from .terrain_scene import stage_scene, terrain_catalog


@dataclass(frozen=True)
class ControlCenterPaths:
    root: Path
    simulator_script: Path
    gateway: Path
    hardware_probe: Path
    mode_probe: Path
    native_action: Path
    sdk_library: Path
    robot_model_dir: Path
    logs: Path

    @property
    def default_hardware_profile(self) -> Path:
        return self.root / "configs" / "b2-reference-profile.json"

    @property
    def hardware_profile_directory(self) -> Path:
        return self.root / "configs" / "hardware-profiles"

    def hardware_profiles(self) -> tuple[Path, ...]:
        """Return selectable repository-owned profiles in deterministic order."""

        candidates: list[Path] = []
        if self.default_hardware_profile.is_file():
            candidates.append(self.default_hardware_profile.resolve())
        if self.hardware_profile_directory.is_dir():
            candidates.extend(
                path.resolve()
                for path in sorted(self.hardware_profile_directory.glob("*.json"))
                if path.is_file()
            )
        return tuple(dict.fromkeys(candidates))

    @classmethod
    def for_repository(cls, root: Path) -> "ControlCenterPaths":
        resolved = root.expanduser().resolve()
        sdk_prefix = Path(
            os.environ.get(
                "UNITREE_SDK2_PREFIX",
                str(resolved / "third_party" / "unitree_sdk2" / "install"),
            )
        ).expanduser().resolve()
        return cls(
            resolved,
            resolved / "scripts" / "run_b2_mujoco.sh",
            resolved / "build" / "gateway" / "b2_policy_controller",
            resolved / "build" / "gateway" / "b2_hw_probe",
            resolved / "build" / "gateway" / "b2_mode_probe",
            resolved / "build" / "gateway" / "b2_native_action",
            sdk_prefix / "lib",
            resolved / "third_party" / "unitree_mujoco" / "unitree_robots" / "b2",
            resolved / "logs" / "control-center",
        )

    def validate(self) -> None:
        for label, path in (
            ("simulator launcher", self.simulator_script),
            ("B2 gateway", self.gateway),
            ("hardware probe", self.hardware_probe),
            ("mode probe", self.mode_probe),
            ("native action", self.native_action),
        ):
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError(f"{label} is unavailable: {path}")
        if not self.sdk_library.is_dir():
            raise ValueError(f"Unitree SDK library directory is unavailable: {self.sdk_library}")
        if not (self.robot_model_dir / "b2.xml").is_file():
            raise ValueError(f"official B2 MuJoCo model is unavailable: {self.robot_model_dir}")
        if not self.hardware_profiles():
            raise ValueError(
                "no hardware safety profiles are available in configs/ or "
                "configs/hardware-profiles/"
            )


class ManagedProcess:
    """A child process with a timestamped log and a bounded live tail."""

    def __init__(
        self,
        name: str,
        command: list[str],
        *,
        log_path: Path,
        environment: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.log_path = log_path
        self.environment = environment
        self.cwd = cwd
        self.process: subprocess.Popen[str] | None = None
        self.lines: deque[str] = deque(maxlen=240)
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        with self.log_path.open("w", encoding="utf-8") as output:
            for raw_line in self.process.stdout:
                line = raw_line.rstrip()
                output.write(raw_line)
                output.flush()
                with self._condition:
                    self.lines.append(line)
                    self._condition.notify_all()
        with self._condition:
            self._condition.notify_all()

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def return_code(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def wait_for_text(self, prefix: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                for line in reversed(self.lines):
                    if prefix in line:
                        return line
                if not self.running:
                    raise RuntimeError(
                        f"{self.name} exited before reporting readiness "
                        f"(code={self.return_code})"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"timed out waiting for {self.name}: {prefix}")
                self._condition.wait(min(remaining, 0.25))

    def stop(
        self,
        *,
        stop_signal: signal.Signals = signal.SIGINT,
        timeout: float = 15.0,
        allow_kill: bool = True,
    ) -> None:
        self.request_stop(stop_signal=stop_signal)
        self.wait_stopped(timeout=timeout, allow_kill=allow_kill)

    def request_stop(
        self, *, stop_signal: signal.Signals = signal.SIGINT
    ) -> None:
        """Request shutdown without waiting, for coordinated process stops."""

        if self.process is None or self.process.poll() is not None:
            return
        self.process.send_signal(stop_signal)

    def wait_stopped(self, *, timeout: float, allow_kill: bool = True) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if not allow_kill:
                raise RuntimeError(
                    f"{self.name} did not stop normally; it was not force-killed"
                ) from exc
            self.process.kill()
            self.process.wait(timeout=5.0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "running": self.running,
            "pid": None if self.process is None else self.process.pid,
            "return_code": self.return_code,
            "log": str(self.log_path),
            "tail": list(self.lines)[-40:],
        }


class ControlCenterSupervisor:
    """Own complete simulation sessions and read-only hardware connections."""

    VALID_STATES = {
        "IDLE",
        "SIM_STARTING",
        "SIMULATION",
        "HARDWARE_CONNECTING",
        "HARDWARE_CONNECTED",
        "STOPPING",
        "FAULT",
    }

    def __init__(self, paths: ControlCenterPaths, policies: PolicyRegistry) -> None:
        self.paths = paths
        self.policies = policies
        self._lock = threading.RLock()
        self.state = "IDLE"
        self.hardware_lifecycle = ControlLifecycle.DISCONNECTED
        self.hardware_output_mode = "NONE"
        self.detail = "等待选择仿真或真机"
        self.session_id = ""
        self.operator_url = ""
        self.scene_id = ""
        self.policy_id = ""
        self.interface = ""
        self.input_source = "keyboard"
        self.hardware_profile_id = ""
        self.video_path = ""
        self.session_dir = ""
        self.processes: dict[str, ManagedProcess] = {}
        self.events: deque[str] = deque(maxlen=160)
        self._worker: threading.Thread | None = None
        self._timed_test_active = False
        self.orphan_hardware_gateways = self._scan_orphan_hardware_gateways(
            self.paths.gateway
        )
        self.ros2_domain0_participants = self._scan_ros2_domain0_participants()
        self.startup_warnings: list[str] = []
        if self.orphan_hardware_gateways:
            self.startup_warnings.append(
                "检测到遗留真机 gateway；控制台保持断开，禁止自动恢复自研状态"
            )
        if self.ros2_domain0_participants:
            self.startup_warnings.append(
                "检测到本机 ROS 2 domain 0 participant；可能产生跨网卡 DDS 发现噪声，"
                "控制台不会自动终止"
            )
        self._event("Control Center ready; no robot connection attempted")
        if self.orphan_hardware_gateways:
            pids = ", ".join(str(item["pid"]) for item in self.orphan_hardware_gateways)
            self._event(f"WARNING: orphan hardware gateway process detected: PID {pids}")
        if self.ros2_domain0_participants:
            pids = ", ".join(
                str(item["pid"]) for item in self.ros2_domain0_participants
            )
            self._event(
                "WARNING: local ROS 2 domain 0 DDS participants detected: PID "
                f"{pids}; no process was terminated"
            )

    @staticmethod
    def _scan_orphan_hardware_gateways(
        gateway: Path, *, proc_root: Path = Path("/proc")
    ) -> list[dict[str, Any]]:
        """Find live hardware gateways without touching or adopting them.

        A process is reported only when its command line contains this repository's
        exact gateway executable and the explicit ``--hardware`` switch.  Startup
        recovery remains passive: the caller merely exposes the warning.
        """

        expected = str(gateway.expanduser().resolve())
        matches: list[dict[str, Any]] = []
        try:
            entries = tuple(proc_root.iterdir())
        except OSError:
            return matches
        for entry in entries:
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            args = [
                item.decode("utf-8", errors="replace")
                for item in raw.split(b"\0")
                if item
            ]
            if expected not in args or "--hardware" not in args:
                continue
            interface = ""
            try:
                interface = args[args.index("--hardware") + 1]
            except (IndexError, ValueError):
                pass
            matches.append(
                {
                    "pid": int(entry.name),
                    "interface": interface,
                    "command": " ".join(args),
                }
            )
        return sorted(matches, key=lambda item: int(item["pid"]))

    @staticmethod
    def _scan_ros2_domain0_participants(
        *, proc_root: Path = Path("/proc")
    ) -> list[dict[str, Any]]:
        """Report non-localhost ROS 2 domain-0 participants without touching them."""

        matches: list[dict[str, Any]] = []
        try:
            entries = tuple(proc_root.iterdir())
        except OSError:
            return matches
        for entry in entries:
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                raw_command = (entry / "cmdline").read_bytes()
                raw_environment = (entry / "environ").read_bytes()
            except OSError:
                continue
            arguments = [
                item.decode("utf-8", errors="replace")
                for item in raw_command.split(b"\0")
                if item
            ]
            environment = {
                key.decode("utf-8", errors="replace"): value.decode(
                    "utf-8", errors="replace"
                )
                for item in raw_environment.split(b"\0")
                if b"=" in item
                for key, value in (item.split(b"=", 1),)
            }
            command = " ".join(arguments)
            command_lower = command.lower()
            has_ros_hint = (
                "ros2cli" in command_lower
                or "/opt/ros/" in command_lower
                or "ROS_DISTRO" in environment
                or "RMW_IMPLEMENTATION" in environment
            )
            if not has_ros_hint:
                continue
            try:
                maps = (entry / "maps").read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                maps = ""
            is_ros2 = (
                "ros2cli" in command_lower
                or "/opt/ros/" in command_lower
                or "librmw_" in maps
            )
            if not is_ros2:
                continue
            domain_id = environment.get("ROS_DOMAIN_ID", "0")
            for index, argument in enumerate(arguments):
                if argument == "--ros-domain-id" and index + 1 < len(arguments):
                    domain_id = arguments[index + 1]
                elif argument.startswith("--ros-domain-id="):
                    domain_id = argument.split("=", 1)[1]
            if domain_id != "0" or environment.get("ROS_LOCALHOST_ONLY") == "1":
                continue
            rmw = environment.get("RMW_IMPLEMENTATION", "")
            for index, argument in enumerate(arguments):
                if argument == "--rmw-implementation" and index + 1 < len(arguments):
                    rmw = arguments[index + 1]
            matches.append(
                {
                    "pid": int(entry.name),
                    "domain_id": 0,
                    "rmw": rmw,
                    "command": command,
                }
            )
        return sorted(matches, key=lambda item: int(item["pid"]))

    def _hardware_profile_identifier(self, path: Path) -> str:
        return path.resolve().relative_to((self.paths.root / "configs").resolve()).as_posix()

    def _resolve_hardware_profile(
        self, identifier: str, *, policy: PolicyProfile | None = None
    ) -> tuple[str, Path]:
        available = {
            self._hardware_profile_identifier(path): path
            for path in self.paths.hardware_profiles()
        }
        selected = identifier
        if not selected and policy is not None:
            def approved_for_policy(path: Path) -> bool:
                try:
                    profile = load_parameter_profile(path)
                except ValueError:
                    return False
                return not hardware_policy_blockers(profile, policy.sha256)

            selected = next(
                (
                    profile_id
                    for profile_id, path in available.items()
                    if approved_for_policy(path)
                ),
                "",
            )
        if not selected:
            selected = next(iter(available), "")
        if not selected or selected not in available:
            raise ValueError(f"unknown hardware safety profile: {identifier}")
        path = available[selected]
        load_parameter_profile(path)
        return selected, path

    def _event(self, message: str) -> None:
        now = datetime.now()
        stamp = now.strftime("%H:%M:%S")
        with self._lock:
            self.events.append(f"[{stamp}] {message}")
            session_dir = self.session_dir
        if session_dir:
            try:
                target = Path(session_dir) / "control-center-events.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {"timestamp": now.isoformat(), "event": message},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except OSError:
                # A logging failure must never alter robot-control state changes.
                pass

    def _record_operation(self, event: dict[str, Any]) -> None:
        """Append one durable, machine-readable operator action record."""

        with self._lock:
            session_dir = self.session_dir
        if not session_dir:
            return
        payload = {"timestamp": datetime.now().isoformat(), **event}
        target = Path(session_dir) / "operator-actions.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _validate_speed(value: Any, label: str, lower: float, upper: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc
        if not lower <= parsed <= upper:
            raise ValueError(f"{label} must be in [{lower}, {upper}]")
        return parsed

    @staticmethod
    def _validate_native_axis(
        value: Any, label: str, lower: float, upper: float
    ) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{label} must be finite")
        if not lower <= parsed <= upper:
            raise ValueError(f"{label} must be in [{lower}, {upper}]")
        return parsed

    def _assert_idle(self) -> None:
        if self.state not in ("IDLE", "FAULT"):
            raise ValueError(f"another session is active: {self.state}")
        if any(process.running for process in self.processes.values()):
            raise ValueError("a managed child process is still running")

    def _session_paths(self, prefix: str) -> tuple[str, Path]:
        session_id = f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        session_dir = self.paths.logs / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        return session_id, session_dir

    def _base_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        current = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            str(self.paths.sdk_library)
            if not current
            else f"{self.paths.sdk_library}:{current}"
        )
        return environment

    @staticmethod
    def _hardware_camera_capability() -> dict[str, Any]:
        device = Path("/dev/video0")
        launcher = shutil.which("gst-launch-1.0")
        return {
            "available": bool(device.exists() and launcher),
            "device": str(device),
            "launcher": launcher or "",
        }

    def _start_hardware_camera(
        self, session_dir: Path, environment: dict[str, str]
    ) -> None:
        capability = self._hardware_camera_capability()
        if not capability["available"]:
            self._event("hardware camera unavailable; connection continues without video")
            return
        output = session_dir / "hardware-camera.webm"
        camera = ManagedProcess(
            "hardware-camera",
            [
                str(capability["launcher"]),
                "-e",
                "v4l2src",
                f"device={capability['device']}",
                "!",
                "videoconvert",
                "!",
                "vp8enc",
                "deadline=1",
                "!",
                "webmmux",
                "!",
                "filesink",
                f"location={output}",
            ],
            log_path=session_dir / "hardware-camera.log",
            environment=environment,
            cwd=self.paths.root,
        )
        with self._lock:
            self.processes["camera"] = camera
            self.video_path = str(output)
        try:
            camera.start()
        except OSError as exc:
            with self._lock:
                self.processes.pop("camera", None)
                self.video_path = ""
            self._event(f"hardware camera failed to start; connection continues: {exc}")
            return
        self._event("hardware camera recording started")

    def start_simulation(self, request: dict[str, Any]) -> None:
        policy = self.policies.get(str(request.get("policy_id", "")))
        scene_id = str(request.get("scene_id", "flat"))
        forward = self._validate_speed(
            request.get("forward_speed", 0.6), "forward speed", 0.05, 1.0
        )
        lateral = self._validate_speed(
            request.get("lateral_speed", 0.25), "lateral speed", 0.05, 0.8
        )
        yaw = self._validate_speed(request.get("yaw_step", 0.10), "yaw step", 0.01, 0.4)
        input_source = str(request.get("input_source", "keyboard"))
        if input_source != "keyboard":
            raise ValueError("仿真控制中心使用网页键盘；手持遥控器仅用于真机连接")
        # Stack10's standalone sim2sim contract uses 200 Hz physics and direct
        # position targets. Keep the existing 500 Hz/8 rad/s path for the
        # legacy 45-value policy and never apply this profile to hardware.
        stack10_sim = policy.input_dim == 423
        scene_path = stage_scene(
            scene_id,
            self.paths.robot_model_dir,
            physics_timestep=0.005 if stack10_sim else 0.002,
        )
        with self._lock:
            self._assert_idle()
            self.state = "SIM_STARTING"
            self.hardware_lifecycle = ControlLifecycle.DISCONNECTED
            self.detail = "正在启动官方 Unitree DDS MuJoCo"
            self.scene_id = scene_id
            self.policy_id = policy.id
            self.interface = "lo / DDS domain 1"
            self.input_source = "keyboard"
            self.operator_url = ""
            self.session_dir = ""
            self.processes = {}
        self._worker = threading.Thread(
            target=self._start_simulation_worker,
            args=(policy, scene_path, forward, lateral, yaw),
            daemon=True,
        )
        self._worker.start()

    def _start_simulation_worker(
        self,
        policy: PolicyProfile,
        scene_path: Path,
        forward: float,
        lateral: float,
        yaw: float,
    ) -> None:
        try:
            session_id, session_dir = self._session_paths("sim")
            video = self.paths.root / "logs" / "videos" / f"{session_id}-{self.scene_id}.mp4"
            environment = self._base_environment()
            simulator_environment = environment.copy()
            simulator_environment["B2_SIM_SCENE"] = str(scene_path)
            simulator_environment["B2_SIM_VIDEO"] = str(video)
            simulator_environment["B2_SIM_KEYBOARD_PORT"] = "15002"
            simulator = ManagedProcess(
                "official-mujoco",
                [str(self.paths.simulator_script)],
                log_path=session_dir / "simulator.log",
                environment=simulator_environment,
                cwd=self.paths.root,
            )
            with self._lock:
                self.session_id = session_id
                self.session_dir = str(session_dir)
                self.video_path = str(video)
                self.processes["simulator"] = simulator
            simulator.start()
            self._event(f"simulator started with scene {self.scene_id}")
            time.sleep(2.0)
            if not simulator.running:
                raise RuntimeError("official MuJoCo simulator exited during startup")

            policy_process = ManagedProcess(
                "policy-ui",
                [
                    sys.executable,
                    "-m",
                    "b2_control.cli_sim_policy",
                    policy.bundle,
                    "--operator-ui",
                    "--no-open-browser",
                    "--operator-ui-port",
                    "0",
                    "--sim-keyboard-port",
                    "15002",
                    "--max-linear-speed",
                    "0.9",
                    "--initial-linear-speed",
                    str(forward),
                    "--max-lateral-speed",
                    str(lateral),
                    "--yaw-step",
                    str(yaw),
                    "--warmup-hold-duration",
                    "0.5" if policy.input_dim == 423 else "3",
                    *(["--disable-target-slew"] if policy.input_dim == 423 else []),
                    "--diagnostics-csv",
                    str(session_dir / "policy.csv"),
                    "--operator-event-log",
                    str(session_dir / "operator-events.jsonl"),
                    "--test-results",
                    str(session_dir / "test-results.jsonl"),
                ],
                log_path=session_dir / "policy.log",
                environment=environment,
                cwd=self.paths.root,
            )
            with self._lock:
                self.processes["policy"] = policy_process
            policy_process.start()
            ready = policy_process.wait_for_text("B2 operator UI:", 20.0)
            operator_url = ready.split("B2 operator UI:", 1)[1].strip()
            self._event("verified policy runtime and operator panel are online")

            gateway = ManagedProcess(
                "sim-gateway",
                [
                    str(self.paths.gateway),
                    "--sim-stack10" if policy.input_dim == 423 else "--sim-hardware-handover",
                    "--diagnostics-csv",
                    str(session_dir / "gateway.csv"),
                ],
                log_path=session_dir / "gateway.log",
                environment=environment,
                cwd=self.paths.root,
            )
            with self._lock:
                self.processes["gateway"] = gateway
            gateway.start()
            gateway.wait_for_text("rt/lowstate -> policy 50 Hz", 12.0)
            with self._lock:
                self.operator_url = operator_url
                self.state = "SIMULATION"
                self.detail = "仿真链路在线；等待 handover 完成后可在下方控制"
            self._event("full LowState -> policy -> LowCmd simulation chain is online")
        except Exception as exc:
            self._event(f"simulation startup failed: {exc}")
            with self._lock:
                self.state = "FAULT"
                self.detail = str(exc)
            self._stop_all_worker(hardware=False, final_state="FAULT")

    def start_timed_simulation_test(self, request: dict[str, Any]) -> None:
        """Run a bounded hold/forward template through the loopback operator API.

        This endpoint is intentionally simulation-only.  It cannot start a hardware
        session and it refuses to run unless the already-active session uses the
        loopback DDS interface.
        """

        if request.get("confirmed") is not True:
            raise ValueError("timed simulation test requires explicit confirmation")
        kind = str(request.get("kind", ""))
        if kind not in {"hold", "forward"}:
            raise ValueError("timed simulation test kind must be hold or forward")
        duration = self._validate_speed(
            request.get("duration_s"), "test duration", 0.5, 30.0
        )
        speed = 0.0
        if kind == "forward":
            speed = self._validate_speed(
                request.get("speed_mps"), "test forward speed", 0.1, 0.9
            )
            if not math.isclose(speed * 10.0, round(speed * 10.0), abs_tol=1e-6):
                raise ValueError("test forward speed must use a 0.1 m/s preset")
        required_scene = str(request.get("required_scene", "")).strip()
        stop_after = request.get("stop_after", True)
        if not isinstance(stop_after, bool):
            raise ValueError("stop_after must be boolean")
        with self._lock:
            if (
                self.state != "SIMULATION"
                or self.interface != "lo / DDS domain 1"
                or not self.operator_url
            ):
                raise ValueError("timed tests are available only in an online simulation")
            if required_scene and required_scene != self.scene_id:
                raise ValueError(
                    f"template requires scene {required_scene}; active scene is {self.scene_id}"
                )
            if self._timed_test_active:
                raise ValueError("another timed simulation test is already active")
            self._timed_test_active = True
        spec = {
            "kind": kind,
            "duration_s": duration,
            "speed_mps": speed,
            "scene_id": self.scene_id,
            "stop_after": stop_after,
        }
        self._record_operation(
            {"category": "timed_simulation_test", "result": "requested", **spec}
        )
        self._event(
            f"timed simulation test requested: {kind}, {duration:.1f}s, "
            f"vx={speed:.1f}"
        )
        self._worker = threading.Thread(
            target=self._timed_simulation_test_worker,
            args=(spec,),
            daemon=True,
        )
        self._worker.start()

    def _operator_api_request(
        self, endpoint: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        with self._lock:
            operator_url = self.operator_url
        parsed = urlsplit(operator_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise RuntimeError("operator API is not a validated loopback URL")
        token = parsed.path.strip("/")
        if not token or not parsed.netloc:
            raise RuntimeError("operator API URL is incomplete")
        target = f"{parsed.scheme}://{parsed.netloc}/api/{token}/{endpoint}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            target,
            data=data,
            headers={} if data is None else {"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        with urlopen(request, timeout=2.0) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise RuntimeError("operator API returned an invalid response")
        return result

    def _timed_simulation_test_worker(self, spec: dict[str, Any]) -> None:
        started_at = datetime.now()
        motion_started = False
        result = "failed"
        detail = ""
        try:
            deadline = time.monotonic() + 30.0
            while True:
                with self._lock:
                    if self.state != "SIMULATION":
                        raise RuntimeError("simulation stopped before timed test was ready")
                status = self._operator_api_request("status")
                if status.get("motion_commands_enabled") is True:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("operator did not enable simulation commands in 30 s")
                time.sleep(0.1)

            self._operator_api_request("key", {"key": "x", "action": "press"})
            if spec["kind"] == "forward":
                preset = str(int(round(float(spec["speed_mps"]) * 10.0)))
                self._operator_api_request(
                    "key", {"key": preset, "action": "press"}
                )
                self._operator_api_request("key", {"key": "w", "action": "press"})
                motion_started = True
            end = time.monotonic() + float(spec["duration_s"])
            while time.monotonic() < end:
                with self._lock:
                    if self.state != "SIMULATION":
                        raise RuntimeError("simulation stopped during timed test")
                time.sleep(min(0.1, max(0.01, end - time.monotonic())))
            result = "completed"
        except Exception as exc:  # worker boundary: persist the exact failure
            detail = str(exc)
            self._event(f"timed simulation test failed: {detail}")
        finally:
            try:
                if motion_started:
                    self._operator_api_request(
                        "key", {"key": "w", "action": "release"}
                    )
                self._operator_api_request("key", {"key": "x", "action": "press"})
            except Exception as zero_exc:
                if not detail:
                    detail = f"zero command failed: {zero_exc}"
                    result = "failed"
                else:
                    detail += f"; zero command failed: {zero_exc}"
            finished_at = datetime.now()
            record = {
                "category": "timed_simulation_test",
                "result": result,
                **spec,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "detail": detail,
            }
            self._record_operation(record)
            with self._lock:
                session_dir = self.session_dir
                self._timed_test_active = False
                should_stop = bool(
                    spec["stop_after"] and self.state == "SIMULATION"
                )
            if session_dir:
                filename = (
                    "timed-test-result-"
                    + finished_at.strftime("%Y%m%d-%H%M%S-%f")
                    + ".json"
                )
                (Path(session_dir) / filename).write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            if result == "completed":
                self._event("timed simulation test completed; command returned to zero")
            if should_stop:
                self.stop(confirmed=True)

    @staticmethod
    def _network_interfaces() -> list[dict[str, Any]]:
        result = []
        for index, name in socket.if_nameindex():
            if name == "lo":
                continue
            operstate = "unknown"
            address = ""
            try:
                operstate = Path(f"/sys/class/net/{name}/operstate").read_text().strip()
            except OSError:
                pass
            try:
                command = subprocess.run(
                    ["ip", "-brief", "address", "show", "dev", name],
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                    check=False,
                )
                address = command.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
            result.append(
                {"index": index, "name": name, "state": operstate, "address": address}
            )
        return result

    def connect_hardware(self, request: dict[str, Any]) -> None:
        if request.get("confirmed") is not True:
            raise ValueError("hardware connection requires explicit confirmation")
        current_orphans = self._scan_orphan_hardware_gateways(self.paths.gateway)
        current_ros2_participants = self._scan_ros2_domain0_participants()
        orphan_warning = (
            "检测到遗留真机 gateway；控制台保持断开，禁止自动恢复自研状态"
        )
        ros2_warning = (
            "检测到本机 ROS 2 domain 0 participant；可能产生跨网卡 DDS 发现噪声，"
            "控制台不会自动终止"
        )
        with self._lock:
            self.orphan_hardware_gateways = current_orphans
            self.ros2_domain0_participants = current_ros2_participants
            self.startup_warnings = [
                warning
                for warning in self.startup_warnings
                if warning not in {orphan_warning, ros2_warning}
            ]
            if current_orphans:
                self.startup_warnings.append(orphan_warning)
            if current_ros2_participants:
                self.startup_warnings.append(ros2_warning)
        if current_orphans:
            pids = ", ".join(str(item["pid"]) for item in current_orphans)
            raise ValueError(
                f"检测到本机遗留真机 gateway PID {pids}；"
                "拒绝只读连接，且不会自动终止或恢复该进程"
            )
        required_checks = {
            "operator_present": "现场操作员已就位",
            "area_clear": "机器人周边和行进区域已清空",
            "robot_supported": "首次接管时机器人有可靠保护/吊架",
            "physical_estop_ready": "物理急停已确认可用",
            "command_source_zero": "键盘已释放或遥控摇杆已回中",
        }
        missing_checks = [
            label for key, label in required_checks.items() if request.get(key) is not True
        ]
        if missing_checks:
            raise ValueError("真机预检未完成：" + "；".join(missing_checks))
        interface = str(request.get("interface", "")).strip()
        interfaces = {
            str(item["name"]): item for item in self._network_interfaces()
        }
        if not interface or interface == "lo" or interface not in interfaces:
            raise ValueError(f"physical network interface is unavailable: {interface}")
        link_state = str(interfaces[interface].get("state", "unknown")).lower()
        if link_state != "up":
            raise ValueError(
                f"物理链路未连接/网卡DOWN：{interface} 当前状态为 "
                f"{link_state or 'unknown'}；请连接网线并等待网卡变为 UP"
            )
        policy = self.policies.get(str(request.get("policy_id", "")))
        forward = self._validate_speed(
            request.get("forward_speed", 0.6), "forward speed", 0.05, 1.0
        )
        lateral = self._validate_speed(
            request.get("lateral_speed", 0.25), "lateral speed", 0.05, 0.8
        )
        yaw = self._validate_speed(request.get("yaw_step", 0.10), "yaw step", 0.01, 0.4)
        input_source = str(request.get("input_source", "keyboard"))
        if input_source not in ("keyboard", "remote"):
            raise ValueError("input source must be keyboard or remote")
        hardware_profile_id, hardware_profile = self._resolve_hardware_profile(
            str(request.get("hardware_profile_id", "")), policy=policy
        )
        with self._lock:
            self._assert_idle()
            self.state = "HARDWARE_CONNECTING"
            self.hardware_lifecycle = ControlLifecycle.READ_ONLY_CHECK
            self.hardware_output_mode = "NONE"
            self.detail = "只读检查 LowState、CRC 和原生 locomotion"
            self.policy_id = policy.id
            self.interface = interface
            self.input_source = input_source
            self.hardware_profile_id = hardware_profile_id
            self.scene_id = ""
            self.video_path = ""
            self.operator_url = ""
            self.session_dir = ""
            self.processes = {}
        self._worker = threading.Thread(
            target=self._connect_hardware_worker,
            args=(
                policy,
                hardware_profile,
                interface,
                input_source,
                forward,
                lateral,
                yaw,
            ),
            daemon=True,
        )
        self._worker.start()

    def _run_readonly_probe(
        self, command: list[str], *, timeout: float, environment: dict[str, str]
    ) -> str:
        result = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            raise RuntimeError(output or f"probe exited with code {result.returncode}")
        return output

    @staticmethod
    def _readonly_probe_summary(output: str) -> tuple[str, dict[str, str]]:
        summary = next(
            (
                line.strip()
                for line in reversed(output.splitlines())
                if line.strip().startswith(("PASS ", "FAIL "))
            ),
            "",
        )
        fields = {
            key: value
            for token in summary.split()
            if "=" in token
            for key, value in (token.split("=", 1),)
        }
        return summary, fields

    def _wait_for_stable_native_lowcmd(
        self,
        *,
        interface: str,
        session_dir: Path,
        environment: dict[str, str],
        max_attempts: int = 5,
    ) -> str:
        """Require one complete 3 s native 350..650 Hz window.

        A freshly restored ``ai`` service may need a few seconds before its
        LowCmd publisher reappears.  Only a low/zero rate gets this bounded
        settling allowance; data-integrity failures and an excessive rate are
        rejected on the first window.
        """

        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        command = [
            str(self.paths.hardware_probe),
            interface,
            "3",
            "--expect-native-lowcmd",
        ]
        log_path = session_dir / "readonly-lowstate.log"
        log_path.write_text("", encoding="utf-8")
        required_fields = {
            "crc_errors",
            "invalid_samples",
            "lowcmd_policy",
            "lowcmd_samples",
            "lowcmd_rate_hz",
            "lowcmd_native_expected",
            "lowcmd_native_rate_ok",
            "lowcmd_publisher_conflict",
        }
        for attempt in range(1, max_attempts + 1):
            remaining_s = (max_attempts - attempt + 1) * 3
            progress = (
                f"原生恢复中：第{attempt}/{max_attempts}次，采集完整3秒窗口，"
                f"最多剩余{remaining_s}s"
            )
            self._event(progress)
            with self._lock:
                if self.state == "HARDWARE_CONNECTING":
                    self.detail = progress
            try:
                output = self._run_readonly_probe(
                    command,
                    timeout=10.0,
                    environment=environment,
                )
            except RuntimeError as error:
                output = str(error)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"=== native LowCmd window {attempt}/{max_attempts} "
                    f"at {datetime.now().isoformat()} ===\n"
                )
                stream.write(output.rstrip() + "\n")
            summary, fields = self._readonly_probe_summary(output)
            missing = sorted(required_fields.difference(fields))
            if not summary or missing:
                raise RuntimeError(
                    "native LowCmd probe returned an incomplete result; "
                    "immediate rejection: " + ", ".join(missing or ["PASS/FAIL summary"])
                )
            try:
                crc_errors = int(float(fields["crc_errors"]))
                invalid_samples = int(float(fields["invalid_samples"]))
                lowcmd_samples = int(float(fields["lowcmd_samples"]))
                lowcmd_rate_hz = float(fields["lowcmd_rate_hz"])
            except ValueError as error:
                raise RuntimeError(
                    "native LowCmd probe returned non-numeric safety fields"
                ) from error
            if (
                fields["lowcmd_policy"] != "native_500hz"
                or fields["lowcmd_native_expected"] != "1"
            ):
                raise RuntimeError(
                    "native LowCmd probe did not run in the required explicit "
                    "native-baseline mode; immediate rejection"
                )
            if crc_errors != 0 or invalid_samples != 0:
                raise RuntimeError(
                    "native LowCmd probe data-integrity failure; immediate rejection: "
                    f"crc_errors={crc_errors} invalid_samples={invalid_samples}"
                )
            if not math.isfinite(lowcmd_rate_hz) or lowcmd_rate_hz < 0:
                raise RuntimeError(
                    "native LowCmd probe returned an invalid rate; immediate rejection"
                )
            if lowcmd_rate_hz > 650.0:
                raise RuntimeError(
                    "native LowCmd rate exceeds the 650 Hz ceiling; immediate rejection: "
                    f"rate={lowcmd_rate_hz:.3f} Hz"
                )
            if lowcmd_rate_hz < 350.0:
                remaining_s = (max_attempts - attempt) * 3
                message = (
                    f"native LowCmd settling window {attempt}/{max_attempts}: "
                    f"rate={lowcmd_rate_hz:.3f} Hz samples={lowcmd_samples}; "
                    f"waiting for a complete 350-650 Hz window, up to {remaining_s} s remain"
                )
                self._event(message)
                with self._lock:
                    if self.state == "HARDWARE_CONNECTING":
                        self.detail = message
                if attempt == max_attempts:
                    raise RuntimeError(
                        "native LowCmd did not stabilize within the bounded "
                        f"{max_attempts * 3} s "
                        f"settling period; last rate={lowcmd_rate_hz:.3f} Hz"
                    )
                continue
            stable = (
                summary.startswith("PASS ")
                and fields["lowcmd_policy"] == "native_500hz"
                and fields["lowcmd_native_expected"] == "1"
                and fields["lowcmd_native_rate_ok"] == "1"
                and fields["lowcmd_publisher_conflict"] == "0"
            )
            if not stable:
                raise RuntimeError(
                    "native LowCmd rate is in range but the complete stable-window "
                    "gate failed; immediate rejection: " + summary
                )
            if attempt > 1:
                self._event(
                    f"native LowCmd stabilized on window {attempt}/{max_attempts}: "
                    f"rate={lowcmd_rate_hz:.3f} Hz"
                )
            return output
        raise RuntimeError("native LowCmd stabilization probe exhausted unexpectedly")

    def _connect_hardware_worker(
        self,
        policy: PolicyProfile,
        hardware_profile: Path,
        interface: str,
        input_source: str,
        forward: float,
        lateral: float,
        yaw: float,
    ) -> None:
        try:
            session_id, session_dir = self._session_paths("hardware")
            environment = self._base_environment()
            with self._lock:
                self.session_id = session_id
                self.session_dir = str(session_dir)
            self._start_hardware_camera(session_dir, environment)
            if self.ros2_domain0_participants:
                pids = ", ".join(
                    str(item["pid"]) for item in self.ros2_domain0_participants
                )
                self._event(
                    "local ROS 2 domain 0 participants may add cross-interface DDS "
                    f"discovery noise (PID {pids}); no process was terminated"
                )
            mode = self._run_readonly_probe(
                [str(self.paths.mode_probe), interface],
                timeout=8.0,
                environment=environment,
            )
            (session_dir / "readonly-mode.log").write_text(mode + "\n", encoding="utf-8")
            if "name=ai" not in mode.split():
                raise RuntimeError(f"native ai locomotion is not active: {mode}")
            self._event(
                "MotionSwitcher confirmed native ai; continuing with the explicit "
                "native LowCmd-rate probe"
            )
            lowstate = self._wait_for_stable_native_lowcmd(
                interface=interface,
                session_dir=session_dir,
                environment=environment,
            )
            required_probe_fields = (
                "PASS",
                "lowcmd_policy=native_500hz",
                "lowcmd_native_expected=1",
                "lowcmd_native_rate_ok=1",
                "lowcmd_publisher_conflict=0",
            )
            missing_probe_fields = [
                field for field in required_probe_fields if field not in lowstate
            ]
            if missing_probe_fields:
                raise RuntimeError(
                    "LowState/native-LowCmd read-only probe did not pass: "
                    + ", ".join(missing_probe_fields)
                )
            self._event(
                "read-only LowState/CRC/native-LowCmd probe passed "
                "(complete rate window is consistent with the native 350-650 Hz "
                "envelope; no rate anomaly detected)"
            )
            with self._lock:
                self.hardware_lifecycle = ControlLifecycle.NATIVE_CONFIRMED
                self.hardware_output_mode = "NATIVE"
            self._event("native ai and its expected LowCmd stream are both confirmed")

            policy_command = [
                sys.executable,
                "-m",
                "b2_control.cli_sim_policy",
                policy.bundle,
                "--operator-ui",
                "--no-open-browser",
                "--operator-ui-port",
                "0",
                "--hardware-console",
                interface,
                "--acknowledge-b2-low-level-control",
                "--max-lateral-speed",
                str(lateral),
                "--yaw-step",
                str(yaw),
                "--warmup-hold-duration",
                "5",
                "--diagnostics-csv",
                str(session_dir / "policy.csv"),
                "--operator-event-log",
                str(session_dir / "operator-events.jsonl"),
                "--test-results",
                str(session_dir / "test-results.jsonl"),
                "--managed-gateway-csv",
                str(session_dir / "gateway.csv"),
                "--managed-gateway-log",
                str(session_dir / "gateway.log"),
                "--managed-gateway-pid",
                str(session_dir / "gateway.pid"),
                "--hardware-profile",
                str(hardware_profile),
            ]
            if input_source == "remote":
                policy_command.extend(
                    ["--max-linear-speed", str(forward), "--remote-control"]
                )
            else:
                policy_command.extend(
                    [
                        "--max-linear-speed",
                        "0.9",
                        "--initial-linear-speed",
                        str(forward),
                    ]
                )
            policy_process = ManagedProcess(
                "hardware-policy-ui",
                policy_command,
                log_path=session_dir / "policy-console.log",
                environment=environment,
                cwd=self.paths.root,
            )
            with self._lock:
                self.processes["policy"] = policy_process
            policy_process.start()
            ready = policy_process.wait_for_text("B2 operator UI:", 20.0)
            with self._lock:
                self.operator_url = ready.split("B2 operator UI:", 1)[1].strip()
                self.state = "HARDWARE_CONNECTED"
                self.detail = (
                    "真机只读连接成功，仍由原生 locomotion 控制；"
                    "下方“自研 locomotion”需要再次确认才会接管"
                )
            self._event(
                "hardware console connected in verified NATIVE mode; input="
                + input_source
            )
        except Exception as exc:
            self._event(f"hardware connection failed: {exc}")
            with self._lock:
                self.state = "FAULT"
                self.hardware_lifecycle = ControlLifecycle.FAULT
                self.hardware_output_mode = "UNKNOWN"
                self.detail = str(exc)
            self._stop_all_worker(hardware=True, final_state="FAULT")

    def stop(self, *, confirmed: bool) -> None:
        if not confirmed:
            raise ValueError("stop/disconnect requires explicit confirmation")
        with self._lock:
            if self.state in ("IDLE", "STOPPING"):
                return
            hardware = self.state in ("HARDWARE_CONNECTED", "HARDWARE_CONNECTING") or (
                self.state == "FAULT" and self.session_id.startswith("hardware-")
            )
            self.state = "STOPPING"
            if hardware:
                self.hardware_lifecycle = ControlLifecycle.STOPPING
            self.detail = (
                "正在归还原生 locomotion 并正常退出"
                if hardware
                else "正在停止策略、gateway 和仿真器并封装录像"
            )
        self._worker = threading.Thread(
            target=self._stop_all_worker,
            args=(hardware, "IDLE"),
            daemon=True,
        )
        self._worker.start()

    def run_native_action(self, request: dict[str, Any]) -> None:
        action = str(request.get("action", ""))
        allowed = {
            "stop",
            "move",
            "stand-up",
            "stand-down",
            "recovery-stand",
            "balance-stand",
            "damping",
            "low-power-on",
            "low-power-off",
            "low-power-status",
            "restore-ai",
        }
        if action not in allowed:
            raise ValueError("unsupported native action")
        if request.get("confirmed") is not True:
            raise ValueError("native robot action requires explicit confirmation")
        if action in {"stand-down", "damping", "low-power-on"} and request.get(
            "danger_confirmed"
        ) is not True:
            raise ValueError("this native action requires danger confirmation")
        with self._lock:
            if self.state != "HARDWARE_CONNECTED":
                raise ValueError("native actions require a verified hardware connection")
            interface = self.interface
            session_dir = Path(self.session_dir)
        command = [str(self.paths.native_action), interface, action]
        parameters: dict[str, float] = {}
        if action == "move":
            parameters = {
                "vx": self._validate_native_axis(request.get("vx", 0.0), "vx", -1.0, 1.0),
                "vy": self._validate_native_axis(request.get("vy", 0.0), "vy", -0.8, 0.8),
                "wz": self._validate_native_axis(request.get("wz", 0.0), "wz", -0.8, 0.8),
            }
            command.extend(
                [
                    "--vx",
                    str(parameters["vx"]),
                    "--vy",
                    str(parameters["vy"]),
                    "--wz",
                    str(parameters["wz"]),
                ]
            )
        command.append("--acknowledge-native-robot-action")
        self._record_operation(
            {
                "category": "native_action",
                "action": action,
                "parameters": parameters,
                "result": "requested",
            }
        )
        try:
            result = subprocess.run(
                command,
                env=self._base_environment(),
                capture_output=True,
                text=True,
                timeout=12.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._record_operation(
                {
                    "category": "native_action",
                    "action": action,
                    "parameters": parameters,
                    "result": "error",
                    "detail": str(exc),
                }
            )
            raise ValueError(f"native action failed: {exc}") from exc
        output = (result.stdout + result.stderr).strip()
        with (session_dir / "native-actions.log").open("a", encoding="utf-8") as log:
            log.write(
                f"{datetime.now().isoformat()} {action} "
                f"{json.dumps(parameters, sort_keys=True)}: {output}\n"
            )
        if result.returncode != 0:
            self._record_operation(
                {
                    "category": "native_action",
                    "action": action,
                    "parameters": parameters,
                    "result": "rejected",
                    "return_code": result.returncode,
                    "detail": output[-1000:],
                }
            )
            raise ValueError(output or f"native action failed: {result.returncode}")
        self._record_operation(
            {
                "category": "native_action",
                "action": action,
                "parameters": parameters,
                "result": "completed",
                "detail": output[-1000:],
            }
        )
        detail = "" if not parameters else f" {parameters}"
        self._event(f"native action completed: {action}{detail}")

    @staticmethod
    def _assert_blackbox_capture_complete(session_dir: Path) -> None:
        blackbox_marker = session_dir / "gateway.csv.blackbox.complete.json"
        if blackbox_marker.is_file():
            try:
                marker = json.loads(blackbox_marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"黑匣子完成标记不可读，拒绝结束 gateway：{exc}") from exc
            if marker.get("status") != "complete":
                raise ValueError("黑匣子完成标记无效，拒绝结束 gateway")
            return
        fault_path = session_dir / "gateway.csv.fault.json"
        try:
            fault = json.loads(fault_path.read_text(encoding="utf-8"))
            fault_ns = int(fault["monotonic_ns"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "故障黑匣子的30秒后窗口尚未确认，且无法读取故障时间；"
                "保持 SAFE_DAMPING gateway 运行"
            ) from exc
        elapsed_s = max(0.0, (time.monotonic_ns() - fault_ns) / 1_000_000_000.0)
        remaining_s = max(0, math.ceil(30.0 - elapsed_s))
        if remaining_s:
            raise ValueError(
                f"故障黑匣子仍在采集后窗口，剩余约 {remaining_s} 秒；"
                "完成标记生成前拒绝结束 gateway"
            )
        raise ValueError(
            "故障后30秒已到，但黑匣子完成标记尚未生成；"
            "保持 SAFE_DAMPING gateway 运行并检查写盘状态"
        )

    @staticmethod
    def _blackbox_capture_is_complete(session_dir: Path) -> bool:
        marker_path = session_dir / "gateway.csv.blackbox.complete.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return marker.get("status") == "complete"

    def _managed_hardware_gateway_is_running(self) -> bool:
        """Return true only for this session's verified hardware gateway PID."""

        with self._lock:
            if not self.session_dir or not self.interface:
                return False
            session_dir = Path(self.session_dir)
            interface = self.interface
        try:
            pid = int((session_dir / "gateway.pid").read_text(encoding="ascii").strip())
            command_line = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except (OSError, ValueError):
            return False
        arguments = [
            item.decode("utf-8", errors="replace") for item in command_line if item
        ]
        return bool(
            arguments
            and str(self.paths.gateway) in arguments
            and "--hardware" in arguments
            and interface in arguments
        )

    def _native_ai_confirmed_after_gateway_exit(self) -> bool:
        """Distinguish a historical policy exit code from a live damping fault."""

        if self._managed_hardware_gateway_is_running():
            return False
        with self._lock:
            interface = self.interface
            already_confirmed = (
                self.hardware_output_mode == "NATIVE"
                and self.hardware_lifecycle
                in {
                    ControlLifecycle.NATIVE,
                    ControlLifecycle.NATIVE_CONFIRMED,
                    ControlLifecycle.STOPPING,
                }
            )
        if already_confirmed:
            return True
        if not interface or interface == "lo":
            return False
        try:
            mode = self._run_readonly_probe(
                [str(self.paths.mode_probe), interface],
                timeout=8.0,
                environment=self._base_environment(),
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            return False
        if "name=ai" not in mode.split():
            return False
        with self._lock:
            self.hardware_lifecycle = ControlLifecycle.NATIVE
            self.hardware_output_mode = "NATIVE"
        return True

    def recover_latched_fault(self, request: dict[str, Any]) -> None:
        required = {
            "confirmed": "确认执行故障恢复",
            "operator_present": "现场操作员已就位",
            "robot_supported": "机器人已可靠支撑",
            "physical_estop_ready": "物理急停可用",
            "fault_inspected": "已检查故障原因和黑匣子",
        }
        missing = [label for key, label in required.items() if request.get(key) is not True]
        if missing:
            raise ValueError("故障恢复预检未完成：" + "；".join(missing))
        with self._lock:
            if self.state != "FAULT" or not self.session_dir or not self.interface:
                raise ValueError("没有可恢复的当前真机故障会话")
            session_dir = Path(self.session_dir)
            interface = self.interface
        pid_path = session_dir / "gateway.pid"
        try:
            pid = int(pid_path.read_text(encoding="ascii").strip())
            command_line = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
                b"\0", b" "
            ).decode("utf-8", errors="replace")
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法验证 SAFE_DAMPING gateway：{exc}") from exc
        if (
            str(self.paths.gateway) not in command_line
            or "--hardware" not in command_line
            or interface not in command_line
        ):
            raise ValueError("PID 不属于当前真机 gateway，拒绝发送信号")
        self._assert_blackbox_capture_complete(session_dir)
        self._record_operation(
            {
                "category": "fault_recovery",
                "action": "terminate_gateway_and_restore_ai",
                "result": "blackbox_complete",
                "gateway_pid": pid,
            }
        )
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 30.0
        while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if Path(f"/proc/{pid}").exists():
            raise ValueError("gateway 未正常退出；未强杀，继续保持故障状态")
        result = subprocess.run(
            [
                str(self.paths.native_action),
                interface,
                "restore-ai",
                "--acknowledge-native-robot-action",
            ],
            env=self._base_environment(),
            capture_output=True,
            text=True,
            timeout=12.0,
            check=False,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            raise ValueError(output or "恢复原生 ai 失败")
        pid_path.unlink(missing_ok=True)
        with self._lock:
            self.state = "IDLE"
            self.hardware_lifecycle = ControlLifecycle.NATIVE
            self.hardware_output_mode = "NATIVE"
            self.detail = "现场故障恢复完成；原生 ai 已恢复，控制台未重新接管"
            self.operator_url = ""
        self._event("onsite fault recovery completed; native ai restored")

    def _stop_all_worker(self, hardware: bool, final_state: str) -> None:
        errors: list[str] = []
        process = self.processes.get("policy")
        if hardware and process is not None:
            try:
                process.stop(timeout=20.0, allow_kill=not hardware)
                if hardware and process.return_code not in (None, 0):
                    if self._native_ai_confirmed_after_gateway_exit():
                        self._event(
                            "hardware policy console retained a historical nonzero "
                            "exit code, but its gateway is gone and native ai is confirmed"
                        )
                    else:
                        errors.append(
                            "硬件策略控制台因安全故障退出；SAFE_DAMPING 可能仍在运行，"
                            "必须使用现场故障恢复流程"
                        )
            except RuntimeError as exc:
                errors.append(str(exc))
        if hardware:
            camera = self.processes.get("camera")
            if camera is not None:
                try:
                    camera.stop(
                        stop_signal=signal.SIGINT,
                        timeout=15.0,
                        allow_kill=True,
                    )
                    self._event("hardware camera recording finalized")
                except (OSError, RuntimeError) as exc:
                    self._event(f"hardware camera finalization failed: {exc}")
        if not hardware:
            # Freeze and close the physics/recorder first while the policy is
            # still publishing.  Then stop the LowCmd gateway and policy.  The
            # former policy-first order created a false timeout fault and let
            # the simulated robot lose actuation in the final video frames.
            simulator = self.processes.get("simulator")
            if simulator is not None:
                try:
                    simulator.request_stop(stop_signal=signal.SIGTERM)
                except OSError as exc:
                    errors.append(f"simulator stop request failed: {exc}")
            gateway = self.processes.get("gateway")
            if gateway is not None:
                try:
                    gateway.stop(timeout=12.0)
                except RuntimeError as exc:
                    errors.append(str(exc))
            if simulator is not None:
                try:
                    simulator.wait_stopped(timeout=35.0)
                except RuntimeError as exc:
                    errors.append(str(exc))
            if process is not None:
                try:
                    process.stop(timeout=20.0)
                except RuntimeError as exc:
                    errors.append(str(exc))
        if not errors and final_state == "IDLE":
            self._generate_session_report()
        with self._lock:
            if errors:
                self.state = "FAULT"
                if hardware:
                    self.hardware_lifecycle = ControlLifecycle.FAULT
                    if self.hardware_output_mode != "DAMPING":
                        self.hardware_output_mode = "DAMPING_OR_UNKNOWN"
                self.detail = "; ".join(errors)
            else:
                self.state = final_state
                if hardware:
                    self.hardware_lifecycle = (
                        ControlLifecycle.FAULT
                        if final_state == "FAULT"
                        else ControlLifecycle.NATIVE
                    )
                    self.hardware_output_mode = (
                        "UNKNOWN" if final_state == "FAULT" else "NATIVE"
                    )
                if final_state == "IDLE":
                    self.detail = "会话已正常结束"
                    self.operator_url = ""
            self._event("session stopped" if not errors else f"stop errors: {self.detail}")

    def _generate_session_report(self) -> None:
        """Generate charts when the session contains actual motion samples."""

        if not self.session_dir:
            return
        session_dir = Path(self.session_dir)
        source = session_dir / "policy.csv"
        plotter = self.paths.root / "scripts" / "plot_policy_tracking.py"
        if not source.is_file() or not plotter.is_file():
            return
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(plotter),
                    str(source),
                    "--output-dir",
                    str(session_dir),
                    "--prefix",
                    "policy-tracking",
                ],
                cwd=self.paths.root,
                env=self._base_environment(),
                capture_output=True,
                text=True,
                timeout=60.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._event(f"diagnostic chart generation failed: {exc}")
            return
        if result.returncode == 0:
            self._event("tracking statistics and visual charts were generated")
        elif "no motion samples" not in (result.stdout + result.stderr):
            message = (result.stdout + result.stderr).strip()
            self._event(f"diagnostic chart generation failed: {message[-240:]}")

    def artifacts(self) -> list[dict[str, Any]]:
        roots = (self.paths.logs, self.paths.root / "logs" / "videos")
        files: list[Path] = []
        for root in roots:
            if root.is_dir():
                files.extend(path for path in root.rglob("*") if path.is_file())
        existing: list[tuple[Path, os.stat_result]] = []
        for path in files:
            try:
                existing.append((path, path.stat()))
            except OSError:
                continue
        existing.sort(key=lambda item: item[1].st_mtime, reverse=True)
        return [
            {
                "name": path.name,
                "path": str(path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
            for path, stat in existing[:30]
        ]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            should_refresh_operator = bool(
                self.state == "HARDWARE_CONNECTED" and self.operator_url
            )
        embedded_fault_latched = False
        if should_refresh_operator:
            try:
                operator_status = self._operator_api_request("status")
                lifecycle = ControlLifecycle(
                    str(operator_status.get("lifecycle_state", ""))
                )
                output_mode = str(operator_status.get("output_mode", ""))
                with self._lock:
                    if self.state == "HARDWARE_CONNECTED":
                        self.hardware_lifecycle = lifecycle
                        self.hardware_output_mode = (
                            output_mode or self.hardware_output_mode
                        )
                        if (
                            lifecycle == ControlLifecycle.FAULT
                            and output_mode == "DAMPING"
                            and self.session_dir
                        ):
                            self.state = "FAULT"
                            self.detail = str(
                                operator_status.get("mode_detail")
                                or "自研 gateway 安全故障已锁存；"
                                "保持 SAFE_DAMPING，等待现场故障恢复"
                            )
                            embedded_fault_latched = True
            except (OSError, RuntimeError, ValueError):
                # Do not infer ownership when the embedded loopback UI is
                # temporarily unavailable.
                pass
        if embedded_fault_latched:
            self._event(
                "embedded hardware gateway reported latched SAFE_DAMPING; "
                "outer session marked recoverable FAULT"
            )
        with self._lock:
            registered_policies = self.policies.profiles()
            selected_policy_profile = next(
                (
                    policy
                    for policy in registered_policies
                    if policy.id == self.policy_id
                ),
                next(
                    (policy for policy in registered_policies if policy.available),
                    None,
                ),
            )
            selected_policy_id = (
                "" if selected_policy_profile is None else selected_policy_profile.id
            )
            hardware_profiles: list[dict[str, Any]] = []
            for profile_path in self.paths.hardware_profiles():
                profile_id = self._hardware_profile_identifier(profile_path)
                try:
                    profile = load_parameter_profile(profile_path)
                    digest = parameter_profile_sha256(profile_path)
                    profile_error = ""
                except ValueError as exc:
                    profile = {}
                    digest = ""
                    profile_error = str(exc)
                policy_approvals: dict[str, dict[str, Any]] = {}
                for policy in registered_policies:
                    blockers = (
                        (profile_error,)
                        if profile_error
                        else hardware_policy_blockers(profile, policy.sha256)
                    )
                    policy_approvals[policy.id] = {
                        "allowed": not blockers,
                        "blockers": list(blockers),
                    }
                hardware_profiles.append(
                    {
                        "id": profile_id,
                        "name": str(profile.get("name") or profile_path.stem),
                        "path": str(profile_path),
                        "sha256": digest,
                        "status": str(profile.get("status") or "invalid"),
                        "error": profile_error,
                        "policy_approvals": policy_approvals,
                    }
                )
            selected_hardware_profile = next(
                (
                    profile
                    for profile in hardware_profiles
                    if profile["id"] == self.hardware_profile_id
                ),
                None,
            )
            if selected_hardware_profile is None and not self.hardware_profile_id:
                selected_hardware_profile = next(
                    (
                        profile
                        for profile in hardware_profiles
                        if selected_policy_id
                        and profile["policy_approvals"]
                        .get(selected_policy_id, {})
                        .get("allowed")
                        is True
                    ),
                    None,
                )
            if selected_hardware_profile is None:
                selected_hardware_profile = next(iter(hardware_profiles), None)
            selected_hardware_profile_id = (
                ""
                if selected_hardware_profile is None
                else str(selected_hardware_profile["id"])
            )
            profiles = []
            for policy in registered_policies:
                payload = policy.to_dict()
                approval = (
                    None
                    if selected_hardware_profile is None
                    else selected_hardware_profile["policy_approvals"].get(policy.id)
                )
                blockers = (
                    ["no hardware safety profile is available"]
                    if approval is None
                    else list(approval["blockers"])
                )
                payload["hardware_allowed"] = not blockers
                payload["hardware_blockers"] = blockers
                profiles.append(payload)
            selected = next(
                (profile for profile in profiles if profile["id"] == selected_policy_id),
                None,
            )
            selected_blockers = (
                selected["hardware_blockers"]
                if selected is not None
                else ["no verified policy is selected"]
            )
            hardware_connected = self.state == "HARDWARE_CONNECTED"
            custom_takeover_ready = bool(
                hardware_connected and selected and selected["hardware_allowed"]
            )
            camera_capability = self._hardware_camera_capability()
            camera_process = self.processes.get("camera")
            camera_recording = bool(
                camera_process is not None and getattr(camera_process, "running", False)
            )
            fault_recovery_ready = bool(
                self.state == "FAULT"
                and self.session_dir
                and self._blackbox_capture_is_complete(Path(self.session_dir))
            )
            return {
                "state": self.state,
                "hardware_lifecycle": self.hardware_lifecycle.value,
                "hardware_output_mode": self.hardware_output_mode,
                "detail": self.detail,
                "session_id": self.session_id,
                "operator_url": self.operator_url,
                "scene_id": self.scene_id,
                "policy_id": selected_policy_id,
                "interface": self.interface,
                "input_source": self.input_source,
                "hardware_profile_id": selected_hardware_profile_id,
                "video_path": self.video_path,
                "session_dir": self.session_dir,
                "processes": {
                    name: process.snapshot() for name, process in self.processes.items()
                },
                "events": list(self.events)[-50:],
                "startup_warnings": list(self.startup_warnings),
                "orphan_hardware_gateways": list(self.orphan_hardware_gateways),
                "ros2_domain0_participants": list(self.ros2_domain0_participants),
                "timed_test_active": self._timed_test_active,
                "policies": profiles,
                "hardware_profiles": hardware_profiles,
                "hardware_capabilities": {
                    "readonly_connected": hardware_connected,
                    "operator_panel": bool(hardware_connected and self.operator_url),
                    "native_actions": hardware_connected,
                    "native_velocity": hardware_connected,
                    "hardware_video_recording": camera_capability["available"],
                    "input_source_switch": hardware_connected,
                    "custom_takeover": custom_takeover_ready,
                    "motion_after_custom_policy": custom_takeover_ready,
                    "fault_recovery": fault_recovery_ready,
                },
                "terrains": terrain_catalog(),
                "interfaces": self._network_interfaces(),
                "artifacts": self.artifacts(),
                "hardware_camera": {
                    **camera_capability,
                    "recording": camera_recording,
                    "output": self.video_path if camera_recording else "",
                },
                "hardware_safety": {
                    "profile": (
                        ""
                        if selected_hardware_profile is None
                        else str(selected_hardware_profile["path"])
                    ),
                    "sha256": (
                        ""
                        if selected_hardware_profile is None
                        else str(selected_hardware_profile["sha256"])
                    ),
                    "approved": bool(selected and selected["hardware_allowed"]),
                    "blockers": selected_blockers,
                    "profile_count": len(hardware_profiles),
                    "profile_directory": str(self.paths.hardware_profile_directory),
                    "legacy_profile_ignored": str(
                        self.paths.root / "config" / "b2-reference-profile.json"
                    ),
                },
            }

    def close(self) -> None:
        with self._lock:
            hardware = self.state in ("HARDWARE_CONNECTED", "HARDWARE_CONNECTING") or (
                self.state == "FAULT" and self.session_id.startswith("hardware-")
            )
        self._stop_all_worker(hardware=hardware, final_state="IDLE")


_PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B2 Control Center</title><style>
*{box-sizing:border-box}body{margin:0;background:#091017;color:#e8f1f6;font-family:Inter,"Noto Sans SC",system-ui,sans-serif}.shell{width:min(1460px,calc(100% - 32px));margin:20px auto;display:grid;gap:14px}header,.panel{background:#111c25;border:1px solid #293b49;border-radius:14px;padding:17px}header{display:flex;align-items:center;justify-content:space-between}h1{margin:0;font-size:22px}.sub{color:#879cac;font-size:12px;margin-top:5px}.badge{padding:8px 13px;border-radius:999px;background:#273540;color:#c5d2dc;font-weight:800;font-size:12px}.badge.live{background:#12392e;color:#7af0bd}.badge.warn{background:#4b2526;color:#ffabb1}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.title{font-size:12px;color:#8ea6b7;letter-spacing:.11em;text-transform:uppercase;margin-bottom:12px}.row{display:flex;gap:9px;align-items:end;flex-wrap:wrap}.field{display:grid;gap:5px;min-width:145px;flex:1}.field label{font-size:11px;color:#8499a9}.field input,.field select{width:100%;background:#0b141b;border:1px solid #304655;color:#edf5fa;border-radius:8px;padding:10px}.button{border:1px solid #3f6074;background:#173245;color:#eaf6fc;border-radius:9px;padding:11px 15px;font-weight:800;cursor:pointer}.button:hover{background:#21506a}.button.danger{background:#8e2a38;border-color:#d65164}.button.hardware{background:#493118;border-color:#95622b}.button:disabled{opacity:.35;cursor:not-allowed}.input-source{display:grid;grid-template-columns:1fr 1fr;gap:7px}.source-button{border:1px solid #3b5363;background:#101d26;color:#b9cad5;border-radius:8px;padding:10px;font-weight:800;cursor:pointer}.source-button.active{border-color:#57d0ed;background:#174459;color:#effbff;outline:2px solid #57d0ed}.warning{color:#ffbd74;font-size:12px;line-height:1.55;margin:10px 0}.fatal-warning{border-left:4px solid #e25d67;color:#ffd1d5}.terrain{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.terrain button{background:#0e1820;border:1px solid #2b414f;color:#dce8ef;text-align:left;padding:10px;border-radius:9px;cursor:pointer}.terrain button.active{border-color:#58cbe7;background:#123341}.terrain b,.terrain small{display:block}.terrain small{color:#8498a7;margin-top:4px}.status{font-size:20px;font-weight:900}.detail{color:#9eb2c1;margin-top:4px}.events{height:190px;overflow:auto;background:#071016;border:1px solid #263c4a;border-radius:9px;padding:10px;font:12px/1.55 ui-monospace,monospace;white-space:pre-wrap;color:#b9ccd8}.operator{width:100%;height:720px;border:1px solid #2b4454;border-radius:11px;background:#070c10}.hidden{display:none!important}.artifact{display:grid;grid-template-columns:2fr 4fr 1fr;gap:8px;border-top:1px solid #263743;padding:8px 0;font-size:12px}.artifact a{color:#67d4ee;text-decoration:none}.artifact span:nth-child(2){color:#8195a4;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.hint{color:#7e93a3;font-size:12px}.source{font-size:11px;color:#778d9c;line-height:1.5}code{color:#73d9f3}.tabs{display:grid;grid-template-columns:1fr 1fr;gap:10px}.tab{padding:15px;border:1px solid #355064;border-radius:11px;background:#111c25;color:#aabecb;font-size:15px;font-weight:900;cursor:pointer}.tab.active{border-color:#62d2ed;background:#153c4d;color:#f4fbff;outline:2px solid #62d2ed}.tab-page{display:none;gap:14px}.tab-page.active{display:grid}.safety{border-left:4px solid #d89a47}.safety.ok{border-left-color:#45d39d}.blockers{color:#ffb5a8;font-size:12px;line-height:1.6;margin-top:8px}.cap-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.cap{background:#0c161d;border:1px solid #2a3e4b;border-radius:9px;padding:11px;color:#8197a6;font-size:12px}.cap b{display:block;margin-top:5px;color:#ffad78}.cap.ready{border-color:#2e765e;color:#a6b9c5}.cap.ready b{color:#70e6b3}.template-running{color:#70e6b3;font-weight:800}@media(max-width:900px){.grid{grid-template-columns:1fr}.terrain{grid-template-columns:1fr 1fr}.operator{height:640px}.cap-grid{grid-template-columns:1fr 1fr}}
</style></head><body><div class="shell">
<header><div><h1>B2 Sim2Real Control Center</h1><div class="sub">独立运行 · 官方原生与自研低层完全分区 · 真机只读连接与显式接管</div></div><div id="badge" class="badge">IDLE</div></header>
<section id="startupWarning" class="panel fatal-warning hidden"></section>
<section class="grid"><div class="panel"><div class="title">Runtime</div><div id="state" class="status">IDLE</div><div id="detail" class="detail"></div><div class="hint" style="margin-top:10px">真机生命周期：<strong id="hardwareLifecycle">DISCONNECTED</strong>　实际输出：<strong id="hardwareOutputMode">NONE</strong></div><div id="session" class="hint" style="margin-top:6px"></div></div><div class="panel"><div class="title">Event Log</div><div id="events" class="events"></div></div></section>
<section class="panel"><div class="title">Real B2 Connection · Shared Read-only Gate</div><div class="row"><div class="field"><label>物理网卡</label><select id="interface"></select></div><div class="field"><label>自研运动命令源（下次真机连接生效）</label><div class="input-source"><button id="keyboardSource" class="source-button active" type="button">键盘控制</button><button id="remoteSource" class="source-button" type="button">手持遥控器</button></div><select id="inputSource" class="hidden"><option value="keyboard">键盘控制</option><option value="remote">宇树手持遥控器</option></select></div><button id="connectHw" class="button hardware">一键连接机器狗（只读检查）</button><button id="disconnectHw" class="button danger">断开 / 归还原生 locomotion</button></div><div id="interfaceStateHint" class="warning">未选择可用物理网卡；网卡为 DOWN 时不会创建会话或启动任何 probe。</div><div class="warning">连接先只读确认 MotionSwitcher <code>name=ai</code>，再以 <code>--expect-native-lowcmd</code> 检查 LowState/CRC 和原生 LowCmd。零/低频允许最多15秒原生恢复等待，但最终必须获得完整350–650 Hz稳定窗口；高频、CRC/invalid、总速率异常或本机遗留 gateway 会立即拒绝。总速率包络不构成 DDS writer 来源确证，不会自动恢复上次自研状态。</div></section>
<nav class="tabs"><button class="tab active" data-tab="nativePage">官方原生 locomotion</button><button class="tab" data-tab="customPage">自研低层 / 仿真</button></nav>

<div id="nativePage" class="tab-page active">
 <section class="panel"><div class="title">Native B2 Operations · 与自研页面隔离</div><div class="row"><button class="button" data-native="stop">停止运动</button><button class="button" data-native="balance-stand">平衡站立</button><button class="button" data-native="stand-up">站起</button><button class="button" data-native="recovery-stand">恢复站立</button><button class="button hardware" data-native="stand-down" data-danger>趴下</button><button class="button" data-native="restore-ai">恢复 ai</button><button class="button danger" data-native="damping" data-danger>原生阻尼</button><button class="button" data-native="low-power-status">低功耗状态</button><button class="button hardware" data-native="low-power-on" data-danger>进入低功耗</button><button class="button" data-native="low-power-off">退出低功耗</button></div><div class="warning">这里只调用官方高层接口；自研 LowCmd 活跃时 C++ 端拒绝原生动作。趴下、阻尼和低功耗要求危险确认。</div></section>
 <section class="panel"><div class="title">Native Velocity Control</div><div class="row"><div class="field"><label>vx m/s（-1.0…1.0）</label><input id="nativeVx" type="number" value="0" min="-1" max="1" step="0.1"></div><div class="field"><label>vy m/s（-0.8…0.8）</label><input id="nativeVy" type="number" value="0" min="-0.8" max="0.8" step="0.1"></div><div class="field"><label>wz rad/s（-0.8…0.8）</label><input id="nativeWz" type="number" value="0" min="-0.8" max="0.8" step="0.1"></div><button id="nativeMove" class="button hardware">发送一次原生速度</button><button id="nativeZero" class="button danger">原生速度归零</button></div><div class="warning">每次发送都要求现场确认。后端和 C++ 同时校验 finite 与轴向边界，越界直接拒绝，不会静默改值。</div></section>
 <section class="panel"><div class="title">Latched Fault Recovery</div><div class="row"><button id="recoverFault" class="button danger">结束 SAFE_DAMPING 并显式恢复原生 ai</button></div><div class="warning">只有故障后30秒黑匣子完成标记已经落盘，才允许结束 gateway；任何一步失败都保持 FAULT。</div></section>
</div>

<div id="customPage" class="tab-page">
 <section class="grid"><div class="panel"><div class="title">Policy Profile</div><div class="row"><div class="field"><label>已验证 Policy</label><select id="policy"></select></div></div><div class="row" style="margin-top:10px"><div class="field"><label>新 Policy bundle 绝对路径</label><input id="policyPath" placeholder="包含 manifest.json 的目录"></div><div class="field"><label>显示名称</label><input id="policyName" placeholder="例如 stairs-policy-v2"></div><button class="button" id="addPolicy">校验并添加</button></div><div id="policyInfo" class="hint" style="margin-top:9px"></div></div><div class="panel"><div class="title">Command Profile</div><div class="row"><div class="field"><label>W/S 默认速度 m/s</label><input id="forward" type="number" value="0.60" min="0.05" max="0.9" step="0.05"></div><div class="field"><label>A/D 固定速度 m/s</label><input id="lateral" type="number" value="0.25" min="0.05" max="0.8" step="0.05"></div><div class="field"><label>Q/E 角速度步长 rad/s</label><input id="yaw" type="number" value="0.10" min="0.01" max="0.4" step="0.01"></div></div><div class="hint" style="margin-top:10px">1–9 选择 0.1–0.9 m/s；W/S/A/D 按住运动、松开停止，Q/E 每次改变角速度。</div></div></section>
 <section class="panel"><div class="title">Automatic Simulation Test Templates</div><div class="row"><button class="button" data-test-template data-kind="hold" data-duration="10" data-scene="flat" data-speed="0">站立承重 10 秒</button><button class="button" data-test-template data-kind="forward" data-duration="2" data-scene="flat" data-speed="0.2">前进 0.2 · 2 秒</button><button class="button" data-test-template data-kind="forward" data-duration="2" data-scene="flat" data-speed="0.4">前进 0.4 · 2 秒</button><button class="button" data-test-template data-kind="forward" data-duration="2" data-scene="flat" data-speed="0.6">前进 0.6 · 2 秒</button><button class="button" data-test-template data-kind="forward" data-duration="4" data-scene="stairs_10cm" data-speed="0.2">10 cm 楼梯 · 4 秒</button><button class="button" data-test-template data-kind="forward" data-duration="4" data-scene="stairs_reference_12cm" data-speed="0.2">参考 12 cm 楼梯 · 4 秒</button><button class="button" data-test-template data-kind="forward" data-duration="4" data-scene="stairs_15cm" data-speed="0.2">15 cm 楼梯 · 4 秒</button></div><div id="templateStatus" class="hint" style="margin-top:9px">点击后会启动所选仿真（若尚未启动），等待操作台可用，自动运行、归零、停止并保存录像、CSV和JSON结果。该入口在任何真机状态都会拒绝执行。</div></section>
 <section class="panel"><div class="title">Simulation Terrain Library</div><div id="terrains" class="terrain"></div><div class="row" style="margin-top:12px"><button id="startSim" class="button">一键启动仿真、DDS gateway、Policy 与录像</button><button id="stopSim" class="button danger">停止并保存录像</button></div><div class="warning">仿真固定使用 DDS domain 1 和 loopback。网页和 MuJoCo 原生窗口都接收 1–9/W/S/A/D/Q/E；仅本页转发键盘命令。</div></section>
 <section id="safetyPanel" class="panel safety"><div class="title">Hardware Activation Gate</div><div class="row"><div class="field"><label>硬件安全配置</label><select id="hardwareProfile"></select></div><div><div id="safetyState" class="status">LOCKED</div></div></div><div id="safetyDigest" class="hint" style="margin-top:9px"></div><div id="safetyBlockers" class="blockers"></div></section>
 <section class="panel"><div class="title">Hardware Operation Readiness</div><div id="capabilities" class="cap-grid"></div><div class="hint" style="margin-top:9px">只读连接、原生动作、自研接管和Policy运动命令分别显示，页面切换不会改变控制权。</div></section>
 <section id="operatorPanel" class="panel hidden"><div class="title">Live Policy Operator</div><div class="warning">键盘控制默认开启；页面失焦会释放网页侧 W/S/A/D。原生页面不会把键盘事件转发到此操作台。</div><iframe id="operator" class="operator"></iframe></section>
</div>

<section class="panel"><div class="title">Recent Diagnostics & Recordings</div><div id="artifacts"></div></section>
<section class="panel source">设计依据：Unitree 官方 MuJoCo LowCmd/LowState DDS 路径；仿真 domain 1 + lo、真机 domain 0 + 物理网卡；官方 B2 MotionSwitcher/CRC 示例。Web 服务仅监听 <code>127.0.0.1</code>。</section>
</div><script>
const token='__TOKEN__',api=`/api/${token}`,$=id=>document.getElementById(id),esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let status={},scene='flat',activeTab='nativePage';
async function call(path,body={}){const r=await fetch(`${api}/${path}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||JSON.stringify(data));return data}
async function getStatus(){const r=await fetch(`${api}/status`,{cache:'no-store'});if(!r.ok)throw Error(await r.text());return r.json()}
function payload(){return{policy_id:$('policy').value,hardware_profile_id:$('hardwareProfile').value,scene_id:scene,forward_speed:$('forward').value,lateral_speed:$('lateral').value,yaw_step:$('yaw').value,input_source:$('inputSource').value}}
function selectTab(id){releaseParentKeys();activeTab=id;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===id));document.querySelectorAll('.tab-page').forEach(x=>x.classList.toggle('active',x.id===id))}document.querySelectorAll('.tab').forEach(x=>x.onclick=()=>selectTab(x.dataset.tab));
$('startSim').onclick=async()=>{try{await call('simulation/start',payload());refresh()}catch(e){alert(e.message)}};$('stopSim').onclick=async()=>{if(!confirm('停止仿真并封装录像？'))return;try{await call('session/stop',{confirmed:true});refresh()}catch(e){alert(e.message)}};
$('connectHw').onclick=async()=>{if(!confirm('确认现场操作员、区域清空、机械保护、物理急停和零命令均已就绪？\n\n本次连接先确认原生 ai，再检查其350–650 Hz LowCmd基线和LowState/CRC；不会自动接管或行走。'))return;try{await call('hardware/connect',{...payload(),interface:$('interface').value,confirmed:true,operator_present:true,area_clear:true,robot_supported:true,physical_estop_ready:true,command_source_zero:true});refresh()}catch(e){alert(e.message)}};
$('disconnectHw').onclick=async()=>{if(!confirm('确认断开控制台并正常归还原生 locomotion？'))return;try{await call('session/stop',{confirmed:true});refresh()}catch(e){alert(e.message)}};
$('addPolicy').onclick=async()=>{try{await call('policies/add',{bundle:$('policyPath').value,name:$('policyName').value});$('policyPath').value='';$('policyName').value='';refresh()}catch(e){alert(e.message)}};
async function runTimedTemplate(button){const requiredScene=button.dataset.scene,kind=button.dataset.kind,duration=Number(button.dataset.duration),speed=Number(button.dataset.speed);if(!['IDLE','SIMULATION'].includes(status.state)){alert('自动模板仅允许在断开状态或仿真状态执行，绝不会操作真机。');return}if(!confirm(`确认运行仿真模板：${button.textContent}？\n完成后自动归零、停止并保存录像、CSV和结果。`))return;try{scene=requiredScene;if(speed>0)$('forward').value=speed.toFixed(2);$('templateStatus').textContent='正在准备仿真并等待操作台可用…';$('templateStatus').classList.add('template-running');if(status.state==='IDLE'){await call('simulation/start',payload());const deadline=Date.now()+60000;while(true){const next=await getStatus();render(next);if(next.state==='SIMULATION')break;if(next.state==='FAULT')throw Error(next.detail);if(Date.now()>deadline)throw Error('仿真在60秒内未完成启动');await new Promise(resolve=>setTimeout(resolve,500))}}else if(status.scene_id!==requiredScene){throw Error(`当前仿真场景是 ${status.scene_id}，模板需要 ${requiredScene}；请先停止当前仿真`)}await call('simulation/timed-test',{confirmed:true,kind,duration_s:duration,speed_mps:speed,required_scene:requiredScene,stop_after:true});$('templateStatus').textContent='模板正在运行；结束后会自动归零、停止并保存结果。'}catch(e){$('templateStatus').textContent=`模板失败：${e.message}`;alert(e.message)}finally{$('templateStatus').classList.remove('template-running');refresh()}}document.querySelectorAll('[data-test-template]').forEach(button=>button.onclick=()=>runTimedTemplate(button));
async function sendNative(action,parameters={}){if(!confirm(`确认执行原生 B2 操作：${action}？\n机器人必须由现场人员看护。`))return;try{await call('hardware/native-action',{action,...parameters,confirmed:true});refresh()}catch(e){alert(e.message)}}
document.querySelectorAll('[data-native]').forEach(button=>button.onclick=async()=>{const action=button.dataset.native;if(!confirm(`确认执行原生 B2 操作：${button.textContent}？\n机器人必须由现场人员看护。`))return;let danger=false;if(button.hasAttribute('data-danger')){danger=confirm('这是可能改变承重/供电状态的危险操作。再次确认现场已做好支撑和急停准备？');if(!danger)return}try{await call('hardware/native-action',{action,confirmed:true,danger_confirmed:danger});refresh()}catch(e){alert(e.message)}});
$('nativeMove').onclick=()=>sendNative('move',{vx:$('nativeVx').value,vy:$('nativeVy').value,wz:$('nativeWz').value});$('nativeZero').onclick=()=>sendNative('move',{vx:0,vy:0,wz:0});
$('recoverFault').onclick=async()=>{if(!confirm('确认操作员在场、机器人已支撑、物理急停可用、故障证据和30秒黑匣子均已检查？'))return;try{await call('hardware/recover-fault',{confirmed:true,operator_present:true,robot_supported:true,physical_estop_ready:true,fault_inspected:true});refresh()}catch(e){alert(e.message)}};
const parentHeld=new Set();function normalizedKey(e){const c=e.code||'';if(c==='Escape')return '\u001b';if(c==='Space')return ' ';if(c.startsWith('Key')&&c.length===4)return c.slice(3).toLowerCase();if(c.startsWith('Digit')&&c.length===6)return c.slice(5);if(e.key==='Escape')return '\u001b';if(e.key==='Spacebar'||e.key===' ')return ' ';return (e.key||'').toLowerCase()}function sendOperatorKey(key,action){const frame=$('operator');if(activeTab!=='customPage'||!status.operator_url||status.input_source!=='keyboard'||!frame.contentWindow)return;frame.contentWindow.postMessage({type:'b2-key',key,action},new URL(status.operator_url).origin)}function releaseParentKeys(){for(const key of [...parentHeld])sendOperatorKey(key,'release');parentHeld.clear()}addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target?.tagName))return;const k=normalizedKey(e);if('wasd'.includes(k)){e.preventDefault();if(!e.repeat&&!parentHeld.has(k)){parentHeld.add(k);sendOperatorKey(k,'press')}return}if('qe123456789'.includes(k)){e.preventDefault();if(!e.repeat)sendOperatorKey(k,'press');return}if(k===' '||k==='x'||k==='z'||k==='r'||k==='\u001b'){e.preventDefault();if(!e.repeat){releaseParentKeys();sendOperatorKey(k,'press')}}},true);addEventListener('keyup',e=>{const k=normalizedKey(e);if(parentHeld.delete(k)){e.preventDefault();sendOperatorKey(k,'release')}},true);addEventListener('blur',releaseParentKeys);document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseParentKeys()});
function selectInputSource(value){$('inputSource').value=value;$('keyboardSource').classList.toggle('active',value==='keyboard');$('remoteSource').classList.toggle('active',value==='remote')}$('keyboardSource').onclick=()=>selectInputSource('keyboard');$('remoteSource').onclick=()=>selectInputSource('remote');selectInputSource('keyboard');
$('policy').onchange=()=>renderPolicy();$('hardwareProfile').onchange=()=>renderPolicy();$('interface').onchange=()=>render(status);function renderPolicy(){const p=(status.policies||[]).find(x=>x.id===$('policy').value),h=(status.hardware_profiles||[]).find(x=>x.id===$('hardwareProfile').value),approval=h?.policy_approvals?.[p?.id],allowed=Boolean(approval?.allowed),blockers=approval?.blockers||['未选择有效的硬件安全配置'];$('policyInfo').textContent=p?`${p.adapter} · ${p.input_dim}→${p.output_dim} · ${p.policy_rate_hz} Hz · SHA-256 ${p.sha256.slice(0,12)}… · 真机${allowed?'已批准':'未批准'}`:'';$('safetyState').textContent=allowed?'READY':'LOCKED';$('safetyPanel').classList.toggle('ok',allowed);$('safetyDigest').textContent=`已选配置：${h?.path||'--'} · SHA-256 ${(h?.sha256||'').slice(0,16)}… · 共 ${status.hardware_profiles?.length||0} 套`;$('safetyBlockers').innerHTML=blockers.length?blockers.map(x=>`• ${esc(x)}`).join('<br>'):'该配置与所选 Policy 哈希已通过审核。'}
function renderCapabilities(c={}){const items=[['LowState / CRC 连接',c.readonly_connected],['实时操作台',c.operator_panel],['原生 B2 动作',c.native_actions],['原生速度控制',c.native_velocity],['真机相机录像',c.hardware_video_recording],['自研 locomotion 接管',c.custom_takeover],['Policy 运动命令',c.motion_after_custom_policy]];$('capabilities').innerHTML=items.map(([label,ready])=>`<div class="cap ${ready?'ready':''}">${label}<b>${ready?'AVAILABLE':'LOCKED'}</b></div>`).join('')}
function selectOptions(el,items,value,label){const old=el.value;el.innerHTML=items.map(x=>`<option value="${esc(x.id??x.name)}">${esc(label(x))}</option>`).join('');if([...el.options].some(x=>x.value===old))el.value=old;else if(value)el.value=value}
function render(s){status=s;$('state').textContent=s.state;$('hardwareLifecycle').textContent=s.hardware_lifecycle||'DISCONNECTED';$('hardwareOutputMode').textContent=s.hardware_output_mode||'NONE';$('detail').textContent=s.detail;$('session').textContent=[s.session_id,s.interface,s.video_path].filter(Boolean).join(' · ');const b=$('badge');b.textContent=s.state;b.className='badge '+(s.state==='SIMULATION'||s.state==='HARDWARE_CONNECTED'?'live':s.state==='FAULT'?'warn':'');const warning=$('startupWarning');if(s.startup_warnings?.length){warning.classList.remove('hidden');warning.innerHTML=`<b>启动恢复告警</b><br>${s.startup_warnings.map(esc).join('<br>')}<br>控制台不会自动接管、终止或恢复这些进程。`}else warning.classList.add('hidden');selectOptions($('policy'),s.policies.filter(x=>x.available),s.policy_id,x=>x.name);selectOptions($('hardwareProfile'),s.hardware_profiles||[],s.hardware_profile_id,x=>`${x.name} · ${x.status}`);selectOptions($('interface'),s.interfaces,null,x=>`${x.name} · ${x.state} ${x.address}`);const selectedInterface=(s.interfaces||[]).find(x=>x.name===$('interface').value);const linkUp=String(selectedInterface?.state||'').toLowerCase()==='up';$('interfaceStateHint').textContent=linkUp?`物理链路已连接：${selectedInterface.name} · UP`:`物理链路未连接/网卡 DOWN：${selectedInterface?.name||'未选择'} · ${selectedInterface?.state||'unknown'}；不会创建会话或启动 camera/probe`;renderPolicy();renderCapabilities(s.hardware_capabilities);$('terrains').innerHTML=s.terrains.map(t=>`<button data-scene="${esc(t.id)}" class="${t.id===scene?'active':''}"><b>${esc(t.name)} · ${esc(t.difficulty)}</b><small>${esc(t.description)}</small></button>`).join('');document.querySelectorAll('[data-scene]').forEach(x=>x.onclick=()=>{scene=x.dataset.scene;render(s)});$('events').textContent=s.events.join('\n');$('events').scrollTop=$('events').scrollHeight;const active=!['IDLE','FAULT'].includes(s.state);$('startSim').disabled=active;$('connectHw').disabled=active||Boolean(s.orphan_hardware_gateways?.length)||!linkUp;$('hardwareProfile').disabled=active;$('stopSim').disabled=!['SIM_STARTING','SIMULATION'].includes(s.state);$('disconnectHw').disabled=!['HARDWARE_CONNECTING','HARDWARE_CONNECTED'].includes(s.state);document.querySelectorAll('[data-native]').forEach(x=>x.disabled=!s.hardware_capabilities?.native_actions);$('nativeMove').disabled=!s.hardware_capabilities?.native_velocity;$('nativeZero').disabled=!s.hardware_capabilities?.native_velocity;$('recoverFault').disabled=!s.hardware_capabilities?.fault_recovery;document.querySelectorAll('[data-test-template]').forEach(x=>x.disabled=s.timed_test_active||!['IDLE','SIMULATION'].includes(s.state));const panel=$('operatorPanel'),frame=$('operator');if(s.operator_url){panel.classList.remove('hidden');if(frame.src!==s.operator_url)frame.src=s.operator_url}else{panel.classList.add('hidden');frame.removeAttribute('src')}const fmt=n=>n>1048576?`${(n/1048576).toFixed(1)} MB`:`${(n/1024).toFixed(1)} KB`;$('artifacts').innerHTML=s.artifacts.slice(0,15).map(a=>`<div class="artifact"><a target="_blank" rel="noopener" href="/artifact/${token}?path=${encodeURIComponent(a.path)}">${esc(a.name)}</a><span title="${esc(a.path)}">${esc(a.path)}</span><span>${fmt(a.size)}</span></div>`).join('')||'<div class="hint">暂无记录</div>'}
async function refresh(){try{render(await getStatus())}catch(e){$('detail').textContent=`Control Center 服务断开：${e.message}`}}setInterval(refresh,500);refresh();
</script></body></html>'''


class ControlCenterWeb:
    def __init__(
        self,
        supervisor: ControlCenterSupervisor,
        *,
        port: int = 8770,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("control center port must be 0..65535")
        self.supervisor = supervisor
        self.port = port
        self.token = secrets.token_urlsafe(18)
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.url = ""

    def open(self, *, open_browser: bool = True) -> str:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlsplit(self.path)
                if parsed.path == f"/{owner.token}/":
                    body = _PAGE.replace("__TOKEN__", owner.token).encode()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == f"/api/{owner.token}/status":
                    self._json(owner.supervisor.snapshot())
                elif parsed.path == f"/artifact/{owner.token}":
                    try:
                        requested = parse_qs(parsed.query).get("path", [""])[0]
                        target = Path(requested).expanduser().resolve(strict=True)
                        roots = (
                            owner.supervisor.paths.logs.resolve(),
                            (owner.supervisor.paths.root / "logs" / "videos").resolve(),
                        )
                        if not target.is_file() or not any(
                            target.is_relative_to(root) for root in roots
                        ):
                            raise ValueError("artifact path is outside managed logs")
                        content_type = mimetypes.guess_type(target.name)[0]
                        self.send_response(HTTPStatus.OK)
                        self.send_header(
                            "Content-Type", content_type or "application/octet-stream"
                        )
                        self.send_header("Content-Length", str(target.stat().st_size))
                        self.send_header("Content-Disposition", "inline")
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        with target.open("rb") as stream:
                            shutil.copyfileobj(stream, self.wfile)
                    except (OSError, ValueError) as exc:
                        self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                prefix = f"/api/{owner.token}/"
                if not self.path.startswith(prefix):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 1 <= length <= 32768:
                        raise ValueError("invalid request length")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("JSON body must be an object")
                    action = unquote(self.path.removeprefix(prefix))
                    if action == "simulation/start":
                        owner.supervisor.start_simulation(payload)
                    elif action == "simulation/timed-test":
                        owner.supervisor.start_timed_simulation_test(payload)
                    elif action == "hardware/connect":
                        owner.supervisor.connect_hardware(payload)
                    elif action == "session/stop":
                        owner.supervisor.stop(confirmed=payload.get("confirmed") is True)
                    elif action == "hardware/native-action":
                        owner.supervisor.run_native_action(payload)
                    elif action == "hardware/recover-fault":
                        owner.supervisor.recover_latched_fault(payload)
                    elif action == "policies/add":
                        owner.supervisor.policies.add(
                            payload.get("bundle", ""), name=payload.get("name") or None
                        )
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    self._json(owner.supervisor.snapshot())
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        actual_port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{actual_port}/{self.token}/"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        if open_browser:
            webbrowser.open(self.url, new=1)
        return self.url

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.server = None
        self.thread = None


__all__ = [
    "ControlCenterPaths",
    "ControlCenterSupervisor",
    "ControlCenterWeb",
    "ManagedProcess",
]
