"""Persistent, verified policy profiles for the B2 control center."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .bundle import BundleError, open_verified_bundle


@dataclass(frozen=True)
class PolicyProfile:
    id: str
    name: str
    bundle: str
    adapter: str
    input_dim: int
    output_dim: int
    policy_rate_hz: int
    sha256: str
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicyRegistry:
    """Store policy bundle paths; verify every profile before it can run."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._entries: list[dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read policy registry {self.path}: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("policies"), list):
            raise ValueError("policy registry must contain a policies list")
        entries: list[dict[str, str]] = []
        for item in value["policies"]:
            if not isinstance(item, dict):
                continue
            bundle = item.get("bundle")
            name = item.get("name")
            adapter = item.get("adapter", "b2_45x12_v1")
            if isinstance(bundle, str) and isinstance(name, str) and isinstance(adapter, str):
                entries.append({"bundle": bundle, "name": name, "adapter": adapter})
        self._entries = entries

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "policies": self._entries}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _profile_id(bundle: Path, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "policy"
        fingerprint = hashlib.sha256(str(bundle).encode()).hexdigest()[:10]
        return f"{slug[:32]}-{fingerprint}"

    @staticmethod
    def _verify(entry: dict[str, str]) -> PolicyProfile:
        bundle_path = Path(entry["bundle"]).expanduser()
        name = entry["name"]
        adapter = entry["adapter"]
        policy_id = PolicyRegistry._profile_id(bundle_path.resolve(strict=False), name)
        try:
            verified = open_verified_bundle(bundle_path)
            digest = hashlib.sha256(verified.policy_path.read_bytes()).hexdigest()
            contract = verified.manifest["policy_contract"]
            if adapter not in {"b2_45x12_v1", "b2_stack10_v1"}:
                raise BundleError(f"unsupported runtime adapter: {adapter}")
            if adapter != verified.adapter:
                raise BundleError(
                    f"bundle adapter mismatch: registry={adapter}, manifest={verified.adapter}"
                )
            return PolicyProfile(
                policy_id,
                name,
                str(verified.root),
                adapter,
                int(contract["actor_input_dim"]),
                int(contract["actor_output_dim"]),
                verified.policy_rate_hz,
                digest,
            )
        except (BundleError, OSError, ValueError, TypeError, KeyError) as exc:
            return PolicyProfile(
                policy_id,
                name,
                str(bundle_path.resolve(strict=False)),
                adapter,
                0,
                0,
                0,
                "",
                False,
                str(exc),
            )

    def profiles(self) -> list[PolicyProfile]:
        return [self._verify(entry) for entry in self._entries]

    def add(
        self,
        bundle: str | Path,
        *,
        name: str | None = None,
        adapter: str = "b2_45x12_v1",
    ) -> PolicyProfile:
        resolved = Path(bundle).expanduser().resolve()
        display_name = (name or resolved.name).strip()
        if not display_name or len(display_name) > 80:
            raise ValueError("policy name must contain 1..80 characters")
        candidate = {"bundle": str(resolved), "name": display_name, "adapter": adapter}
        profile = self._verify(candidate)
        if not profile.available:
            raise ValueError(profile.error)
        self._entries = [
            entry
            for entry in self._entries
            if Path(entry["bundle"]).expanduser().resolve(strict=False) != resolved
        ]
        self._entries.append(candidate)
        self._save()
        return profile

    def get(self, policy_id: str) -> PolicyProfile:
        for profile in self.profiles():
            if profile.id == policy_id:
                if not profile.available:
                    raise ValueError(profile.error)
                return profile
        raise ValueError(f"unknown policy profile: {policy_id}")


__all__ = ["PolicyProfile", "PolicyRegistry"]
