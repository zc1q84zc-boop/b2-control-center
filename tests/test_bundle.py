import hashlib
import json
from pathlib import Path

import pytest

import b2_control.bundle as bundle_module
from b2_control.bundle import BundleError, open_verified_bundle


def _write_bundle(root: Path, payload: bytes = b"policy") -> Path:
    model = root / "models" / "policy.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "recommended_runtime_artifact": "models/policy.onnx",
        "policy_contract": {
            "actor_input_dim": 45,
            "actor_output_dim": 12,
            "input_dtype": "float32",
            "output_dtype": "float32",
            "policy_rate_hz": 50,
        },
        "artifacts": [
            {"path": "models/policy.onnx", "sha256": digest, "role": "actor"}
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return model


def test_open_verified_bundle_accepts_matching_hash(tmp_path: Path) -> None:
    model = _write_bundle(tmp_path)
    bundle = open_verified_bundle(tmp_path)
    assert bundle.policy_path == model.resolve()
    assert bundle.policy_rate_hz == 50


def test_open_verified_bundle_rejects_hash_mismatch(tmp_path: Path) -> None:
    model = _write_bundle(tmp_path)
    model.write_bytes(b"changed")
    with pytest.raises(BundleError, match="SHA-256 mismatch"):
        open_verified_bundle(tmp_path)


def test_open_verified_bundle_rejects_wrong_contract(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policy_contract"]["actor_input_dim"] = 48
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="45 -> 12"):
        open_verified_bundle(tmp_path)


def test_open_verified_bundle_rejects_non_object_contract(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policy_contract"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="policy_contract must be an object"):
        open_verified_bundle(tmp_path)


def test_open_verified_bundle_rejects_non_object_artifact(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [None]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="Artifact entries must be objects"):
        open_verified_bundle(tmp_path)


def test_open_verified_bundle_rejects_unverified_recommended_policy(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    unverified = tmp_path / "models" / "unverified.onnx"
    unverified.write_bytes(b"unverified")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recommended_runtime_artifact"] = "models/unverified.onnx"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        BundleError, match="recommended runtime artifact is not a verified artifact"
    ):
        open_verified_bundle(tmp_path)


def test_open_verified_bundle_rejects_parent_artifact_escape(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        {
            "path": "../outside.onnx",
            "sha256": hashlib.sha256(b"outside").hexdigest(),
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="Artifact escapes bundle root"):
        open_verified_bundle(root)


def test_open_verified_bundle_rejects_artifact_symlink_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    link = root / "models" / "external.onnx"
    link.symlink_to(outside)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(
        {
            "path": "models/external.onnx",
            "sha256": hashlib.sha256(b"outside").hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="Artifact escapes bundle root"):
        open_verified_bundle(root)


def test_open_verified_bundle_rejects_recommended_symlink_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    link = root / "models" / "external.onnx"
    link.symlink_to(outside)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recommended_runtime_artifact"] = "models/external.onnx"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        BundleError, match="Recommended runtime artifact escapes bundle root"
    ):
        open_verified_bundle(root)


def test_open_verified_bundle_rejects_manifest_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    manifest_path = root / "manifest.json"
    external_manifest = tmp_path / "external.json"
    external_manifest.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(external_manifest)
    with pytest.raises(BundleError, match="Manifest escapes bundle root"):
        open_verified_bundle(root)


def test_open_verified_bundle_accepts_internal_manifest_symlink(
    tmp_path: Path,
) -> None:
    model = _write_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    internal_manifest = tmp_path / "metadata" / "manifest.json"
    internal_manifest.parent.mkdir()
    manifest_path.replace(internal_manifest)
    manifest_path.symlink_to(Path("metadata") / "manifest.json")
    bundle = open_verified_bundle(tmp_path)
    assert bundle.policy_path == model.resolve()


def test_open_verified_bundle_rejects_invalid_manifest_utf8(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_bytes(b"\xff")
    with pytest.raises(BundleError, match="Cannot read bundle manifest"):
        open_verified_bundle(tmp_path)


def test_open_verified_bundle_translates_artifact_hash_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bundle(tmp_path)

    def fail_to_hash(path: Path) -> str:
        raise OSError("denied")

    monkeypatch.setattr(bundle_module, "_sha256_file", fail_to_hash)
    with pytest.raises(BundleError, match="Cannot hash artifact") as exc_info:
        open_verified_bundle(tmp_path)
    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "denied"


def test_open_verified_bundle_translates_artifact_resolution_error(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    loop = tmp_path / "models" / "loop.onnx"
    loop.symlink_to("loop.onnx")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(
        {"path": "models/loop.onnx", "sha256": "0" * 64}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="Cannot resolve artifact"):
        open_verified_bundle(tmp_path)


def test_open_verified_bundle_translates_malformed_artifact_path(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "models/policy.onnx\u0000suffix"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="Cannot resolve artifact") as exc_info:
        open_verified_bundle(tmp_path)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_open_verified_bundle_translates_malformed_root_path() -> None:
    with pytest.raises(
        BundleError, match="Cannot resolve bundle directory"
    ) as exc_info:
        open_verified_bundle("/tmp/b2\u0000bundle")
    assert isinstance(exc_info.value.__cause__, ValueError)
