from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import b2_control.cli_verify as cli


PINNED_HOLD_ACTION = np.array(
    (
        0.3856276273727417,
        -0.4630189836025238,
        1.292534351348877,
        -0.4785652160644531,
        -0.5396023988723755,
        1.1809988021850586,
        0.3231983184814453,
        -0.3763878345489502,
        0.8902627229690552,
        -0.055662669241428375,
        -0.16711482405662537,
        0.8141254186630249,
    ),
    dtype=np.float32,
)


class FakePolicy:
    action = PINNED_HOLD_ACTION

    def __init__(self, path: Path):
        self.path = path

    def infer(self, observation):
        expected = np.zeros((1, 45), dtype=np.float32)
        expected[0, 3:6] = (0.0, 0.0, -1.0)
        np.testing.assert_array_equal(observation, expected)
        return self.action.copy()


def _patch_bundle_and_policy(monkeypatch, tmp_path):
    bundle = SimpleNamespace(policy_path=tmp_path / "policy.onnx")
    monkeypatch.setattr(cli, "open_verified_bundle", lambda path: bundle)
    monkeypatch.setattr(cli, "OnnxPolicy", FakePolicy)


def test_verify_reports_success(monkeypatch, capsys, tmp_path):
    _patch_bundle_and_policy(monkeypatch, tmp_path)
    assert cli.main([str(tmp_path)]) == 0
    assert "Bundle verified" in capsys.readouterr().out


def test_verify_rejects_golden_mismatch(monkeypatch, tmp_path):
    _patch_bundle_and_policy(monkeypatch, tmp_path)
    monkeypatch.setattr(FakePolicy, "action", np.zeros(12, dtype=np.float32))
    with pytest.raises(AssertionError):
        cli.main([str(tmp_path)])
