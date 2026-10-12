import json

import pytest

from b2_control.timed_test import TimedTestRunner


def test_forward_test_is_bounded_and_finishes_at_zero(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    runner = TimedTestRunner(results)
    command = runner.start(
        "forward-2s", now=10.0, selected_speed_m_s=0.4, initial_lost=3
    )
    assert command.tolist() == pytest.approx([0.4, 0.0, 0.0])
    runner.observe(tracking_error_rad=0.12, temperature_c=48, lost=3)
    assert runner.finish_if_due(now=11.9) is None
    result = runner.finish_if_due(now=12.0)
    assert result is not None
    assert result["verdict"] == "PASS"
    assert result["final_command"] == [0.0, 0.0, 0.0]
    assert json.loads(results.read_text(encoding="utf-8"))["speed_m_s"] == 0.4


def test_fault_aborts_test_and_records_failure() -> None:
    runner = TimedTestRunner()
    runner.start("stand-10s", now=1.0, selected_speed_m_s=0.6, initial_lost=0)
    result = runner.abort(now=2.0, reason="SAFE_DAMPING")
    assert result is not None
    assert result["verdict"] == "FAIL"
    assert runner.active is None
