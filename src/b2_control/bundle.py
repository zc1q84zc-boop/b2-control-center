from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


class BundleError(RuntimeError):
    """The policy bundle is incomplete or violates the deployment contract."""


@dataclass(frozen=True)
class VerifiedBundle:
    root: Path
    manifest: dict[str, Any]
    policy_path: Path
    policy_rate_hz: int
    input_dim: int = 45
    output_dim: int = 12
    adapter: str = "b2_45x12_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_bundle_child(root: Path, relative: str, label: str) -> Path:
    try:
        resolved = (root / relative).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise BundleError(f"Cannot resolve {label.lower()} {relative}: {exc}") from exc
    if root not in resolved.parents:
        raise BundleError(f"{label} escapes bundle root: {relative}")
    return resolved


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = _resolve_bundle_child(root, "manifest.json", "Manifest")
    if not manifest_path.is_file():
        raise BundleError(f"Missing bundle manifest: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"Cannot read bundle manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError("Bundle manifest root must be an object")
    return value


def open_verified_bundle(path: str | Path) -> VerifiedBundle:
    candidate = Path(path)
    try:
        root = candidate.expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise BundleError(f"Cannot resolve bundle directory {candidate}: {exc}") from exc
    if not root.is_dir():
        raise BundleError(f"Bundle directory does not exist: {root}")
    manifest = _load_manifest(root)

    contract = manifest.get("policy_contract")
    if not isinstance(contract, dict):
        raise BundleError("policy_contract must be an object")
    signature = (
        contract.get("actor_input_dim"),
        contract.get("actor_output_dim"),
        contract.get("input_dtype"),
        contract.get("output_dtype"),
    )
    if signature not in ((45, 12, "float32", "float32"), (423, 12, "float32", "float32")):
        raise BundleError(
            f"Expected float32 45 -> 12 or 423 -> 12 policy contract, got {signature}"
        )
    if contract.get("policy_rate_hz") != 50:
        raise BundleError("Expected 50 Hz policy rate")
    input_dim = int(contract["actor_input_dim"])
    output_dim = int(contract["actor_output_dim"])
    adapter = str(contract.get("adapter", "b2_45x12_v1"))
    expected_adapter = "b2_stack10_v1" if input_dim == 423 else "b2_45x12_v1"
    if adapter != expected_adapter:
        raise BundleError(
            f"input dimension {input_dim} requires adapter {expected_adapter}, got {adapter}"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BundleError("Bundle manifest has no artifacts")
    verified_artifacts: set[Path] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            raise BundleError("Artifact entries must be objects")
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise BundleError("Artifact entries require path and sha256 strings")
        artifact = _resolve_bundle_child(root, relative, "Artifact")
        if not artifact.is_file():
            raise BundleError(f"Missing artifact: {relative}")
        try:
            actual = _sha256_file(artifact)
        except OSError as exc:
            raise BundleError(f"Cannot hash artifact {relative}: {exc}") from exc
        if actual != expected:
            raise BundleError(
                f"SHA-256 mismatch for {relative}: {actual} != {expected}"
            )
        verified_artifacts.add(artifact)

    recommended = manifest.get("recommended_runtime_artifact")
    if not isinstance(recommended, str):
        raise BundleError("Missing recommended_runtime_artifact")
    policy_path = _resolve_bundle_child(root, recommended, "Recommended runtime artifact")
    if not policy_path.is_file():
        raise BundleError(f"Invalid recommended policy path: {recommended}")
    if policy_path not in verified_artifacts:
        raise BundleError(
            f"recommended runtime artifact is not a verified artifact: {recommended}"
        )
    return VerifiedBundle(root, manifest, policy_path, 50, input_dim, output_dim, adapter)
