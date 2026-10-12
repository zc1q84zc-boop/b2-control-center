"""Deterministic timed-test sequencing for the browser operator console."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimedTestDefinition:
    id: str
    name: str
    duration_s: float
    kind: str
    default_speed_m_s: float


TEST_DEFINITIONS = {
    item.id: item
    for item in (
        TimedTestDefinition("stand-10s", "站立 10 秒", 10.0, "hold", 0.0),
        TimedTestDefinition("forward-2s", "前进 2 秒", 2.0, "forward", 0.25),
        TimedTestDefinition("speed-0.1", "前进 0.1 m/s · 2 秒", 2.0, "forward", 0.1),
        TimedTestDefinition("speed-0.25", "前进 0.25 m/s · 2 秒", 2.0, "forward", 0.25),
        TimedTestDefinition("speed-0.4", "前进 0.4 m/s · 2 秒", 2.0, "forward", 0.4),
        TimedTestDefinition("speed-0.6", "前进 0.6 m/s · 2 秒", 2.0, "forward", 0.6),
        TimedTestDefinition("stairs", "楼梯测试 · 4 秒", 4.0, "forward", 0.25),
    )
}


class TimedTestRunner:
    """Own at most one bounded test and force zero on every terminal path."""

    def __init__(self, result_path: Path | None = None) -> None:
        self.result_path = result_path
        self.active: TimedTestDefinition | None = None
        self.started_at = 0.0
        self.speed_m_s = 0.0
        self.samples = 0
        self.max_tracking_error_rad = 0.0
        self.max_temperature_c = 0
        self.start_lost = 0
        self.last_lost = 0

    def start(
        self,
        test_id: str,
        *,
        now: float,
        selected_speed_m_s: float,
        initial_lost: int,
    ) -> np.ndarray:
        if self.active is not None:
            raise ValueError("已有测试正在运行")
        if test_id not in TEST_DEFINITIONS:
            raise ValueError(f"未知测试模板：{test_id}")
        definition = TEST_DEFINITIONS[test_id]
        speed = (
            selected_speed_m_s
            if definition.id == "forward-2s"
            else definition.default_speed_m_s
        )
        if definition.kind == "forward" and not 0.05 <= speed <= 0.9:
            raise ValueError("测试速度必须在 0.05..0.9 m/s")
        self.active = definition
        self.started_at = now
        self.speed_m_s = speed
        self.samples = 0
        self.max_tracking_error_rad = 0.0
        self.max_temperature_c = 0
        self.start_lost = initial_lost
        self.last_lost = initial_lost
        return self.command()

    def command(self) -> np.ndarray:
        if self.active is None or self.active.kind == "hold":
            return np.zeros(3, dtype=np.float32)
        return np.asarray([self.speed_m_s, 0.0, 0.0], dtype=np.float32)

    def observe(
        self,
        *,
        tracking_error_rad: float,
        temperature_c: int,
        lost: int,
    ) -> None:
        if self.active is None:
            return
        self.samples += 1
        self.max_tracking_error_rad = max(
            self.max_tracking_error_rad, float(tracking_error_rad)
        )
        self.max_temperature_c = max(self.max_temperature_c, int(temperature_c))
        self.last_lost = int(lost)

    def finish_if_due(
        self,
        *,
        now: float,
        fault: str = "",
    ) -> dict[str, Any] | None:
        if self.active is None or (not fault and now - self.started_at < self.active.duration_s):
            return None
        definition = self.active
        criteria = {
            "received_samples": self.samples > 0,
            "tracking_error_le_1_rad": self.max_tracking_error_rad <= 1.0,
            "motor_temperature_lt_90_c": self.max_temperature_c < 90,
            "motor_lost_unchanged": self.last_lost == self.start_lost,
            "no_fault": not fault,
        }
        result = {
            **asdict(definition),
            "started_at_monotonic_s": self.started_at,
            "finished_at_monotonic_s": now,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "speed_m_s": self.speed_m_s,
            "samples": self.samples,
            "max_tracking_error_rad": self.max_tracking_error_rad,
            "max_temperature_c": self.max_temperature_c,
            "lost_increment": max(0, self.last_lost - self.start_lost),
            "fault": fault,
            "criteria": criteria,
            "verdict": "PASS" if all(criteria.values()) else "FAIL",
            "final_command": [0.0, 0.0, 0.0],
        }
        self.active = None
        if self.result_path is not None:
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            with self.result_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    def abort(self, *, now: float, reason: str) -> dict[str, Any] | None:
        return self.finish_if_due(now=now, fault=reason)


__all__ = ["TEST_DEFINITIONS", "TimedTestDefinition", "TimedTestRunner"]
