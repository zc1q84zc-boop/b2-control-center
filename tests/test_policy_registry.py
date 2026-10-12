import hashlib
import json
from pathlib import Path

import pytest

from b2_control.policy_registry import PolicyRegistry


def _bundle(root: Path) -> Path:
    model = root / "models" / "policy.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fake-policy")
    manifest = {
        "policy_contract": {
            "actor_input_dim": 45,
            "actor_output_dim": 12,
            "input_dtype": "float32",
            "output_dtype": "float32",
            "policy_rate_hz": 50,
        },
        "artifacts": [
            {"path": "models/policy.onnx", "sha256": hashlib.sha256(b"fake-policy").hexdigest()}
        ],
        "recommended_runtime_artifact": "models/policy.onnx",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_registry_persists_and_reverifies_policy_profiles(tmp_path: Path) -> None:
    registry_path = tmp_path / "config" / "policies.json"
    bundle = _bundle(tmp_path / "bundle")
    registry = PolicyRegistry(registry_path)
    profile = registry.add(bundle, name="stairs-v2")
    assert profile.input_dim == 45
    assert profile.output_dim == 12
    assert profile.adapter == "b2_45x12_v1"

    reloaded = PolicyRegistry(registry_path)
    assert reloaded.get(profile.id).sha256 == hashlib.sha256(b"fake-policy").hexdigest()


def test_registry_marks_policy_unavailable_after_artifact_changes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    registry = PolicyRegistry(tmp_path / "policies.json")
    profile = registry.add(bundle, name="verified")
    (bundle / "models" / "policy.onnx").write_bytes(b"changed")
    unavailable = registry.profiles()[0]
    assert unavailable.id == profile.id
    assert unavailable.available is False
    assert "SHA-256 mismatch" in unavailable.error
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        registry.get(profile.id)


def test_registry_rejects_unknown_runtime_adapter(tmp_path: Path) -> None:
    registry = PolicyRegistry(tmp_path / "policies.json")
    with pytest.raises(ValueError, match="unsupported runtime adapter"):
        registry.add(_bundle(tmp_path / "bundle"), adapter="future_adapter")
