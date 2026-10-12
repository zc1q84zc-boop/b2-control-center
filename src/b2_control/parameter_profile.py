from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping


REQUIRED_HARDWARE_APPROVAL_FIELDS = (
    "position_kp",
    "velocity_kd",
    "gain_ramp_s",
    "damping_kd",
    "target_slew_rad_s",
    "soft_limit_margin_rad",
    "torque_limit",
    "current_limit",
    "joint_velocity_trip_rad_s",
    "tracking_error_trip_rad",
    "tracking_error_duration_s",
    "motor_temperature_trip_c",
    "motor_temperature_reset_c",
    "battery_soc_min_percent",
    "battery_current_abs_limit_raw",
    "battery_temperature_trip_c",
    "battery_temperature_reset_c",
    "tilt_trip_rad",
    "angular_rate_trip_rad_s",
    "state_timeout_s",
    "policy_timeout_s",
    "command_timeout_s",
    "publisher_timeout_s",
    "recovery_timeout_s",
    "lowcmd_conflict_check_s",
    "fault_debounce_s",
    "policy_sha256",
    "robot_serial",
    "firmware_version",
    "reviewer",
    "reviewed_at",
)

_VECTOR_FIELDS = (
    "position_kp",
    "velocity_kd",
    "target_slew_rad_s",
    "soft_limit_margin_rad",
    "torque_limit",
    "current_limit",
    "joint_velocity_trip_rad_s",
    "tracking_error_trip_rad",
)
_POSITIVE_SCALAR_FIELDS = (
    "gain_ramp_s",
    "damping_kd",
    "tracking_error_duration_s",
    "motor_temperature_trip_c",
    "motor_temperature_reset_c",
    "tilt_trip_rad",
    "angular_rate_trip_rad_s",
    "state_timeout_s",
    "policy_timeout_s",
    "command_timeout_s",
    "publisher_timeout_s",
    "recovery_timeout_s",
    "lowcmd_conflict_check_s",
    "fault_debounce_s",
    "battery_current_abs_limit_raw",
    "battery_temperature_trip_c",
    "battery_temperature_reset_c",
)

_NONEMPTY_STRING_FIELDS = (
    "policy_sha256",
    "robot_serial",
    "firmware_version",
    "reviewer",
    "reviewed_at",
)


def load_parameter_profile(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load parameter profile: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("parameter profile must be a JSON object")
    if value.get("schema_version") != 1:
        raise ValueError("unsupported parameter profile schema_version")
    return value


def parameter_profile_sha256(path: str | Path) -> str:
    """Return the digest of the exact reviewed profile bytes."""

    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read parameter profile: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def hardware_activation_blockers(profile: Mapping[str, Any]) -> tuple[str, ...]:
    if profile.get("parameter_source") == "compiled_simulation_parameters":
        blockers: list[str] = []
        if profile.get("status") != "operator_authorized_experimental":
            blockers.append(
                "experimental profile status is not operator_authorized_experimental"
            )
        if profile.get("activation_permitted") is not True:
            blockers.append("activation_permitted is not true")
        policy_sha = profile.get("policy_sha256")
        if (
            not isinstance(policy_sha, str)
            or len(policy_sha) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in policy_sha
            )
        ):
            blockers.append("policy_sha256 must be a SHA-256 hex digest")
        return tuple(blockers)

    blockers: list[str] = []
    if profile.get("status") != "approved":
        blockers.append("status is not approved")
    if profile.get("activation_permitted") is not True:
        blockers.append("activation_permitted is not true")

    approval = profile.get("hardware_approval")
    if not isinstance(approval, Mapping):
        blockers.append("hardware_approval is missing")
        return tuple(blockers)
    for field in REQUIRED_HARDWARE_APPROVAL_FIELDS:
        if approval.get(field) is None:
            blockers.append(f"hardware_approval.{field} is missing")
    for field in _VECTOR_FIELDS:
        value = approval.get(field)
        if value is not None and not _is_positive_12_vector(value):
            blockers.append(f"hardware_approval.{field} must be a positive 12-vector")
    for field in _POSITIVE_SCALAR_FIELDS:
        value = approval.get(field)
        if value is not None and not _is_positive_finite_number(value):
            blockers.append(f"hardware_approval.{field} must be positive and finite")

    soc = approval.get("battery_soc_min_percent")
    if soc is not None and (
        not _is_finite_number(soc) or not 0.0 < float(soc) <= 100.0
    ):
        blockers.append(
            "hardware_approval.battery_soc_min_percent must be in (0,100]"
        )

    trip = approval.get("motor_temperature_trip_c")
    reset = approval.get("motor_temperature_reset_c")
    if trip is not None and not _is_finite_number(trip):
        blockers.append("hardware_approval.motor_temperature_trip_c must be finite")
    if reset is not None and not _is_finite_number(reset):
        blockers.append("hardware_approval.motor_temperature_reset_c must be finite")
    if _is_finite_number(trip) and _is_finite_number(reset) and float(trip) <= float(reset):
        blockers.append("motor temperature trip must exceed reset")

    battery_trip = approval.get("battery_temperature_trip_c")
    battery_reset = approval.get("battery_temperature_reset_c")
    if battery_reset is not None and not _is_finite_number(battery_reset):
        blockers.append("hardware_approval.battery_temperature_reset_c must be finite")
    if (
        _is_finite_number(battery_trip)
        and _is_finite_number(battery_reset)
        and float(battery_trip) <= float(battery_reset)
    ):
        blockers.append("battery temperature trip must exceed reset")

    for field in _NONEMPTY_STRING_FIELDS:
        value = approval.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            blockers.append(f"hardware_approval.{field} must be a non-empty string")
    policy_sha = approval.get("policy_sha256")
    if policy_sha is not None and (
        not isinstance(policy_sha, str)
        or len(policy_sha) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in policy_sha)
    ):
        blockers.append("hardware_approval.policy_sha256 must be a SHA-256 hex digest")
    command_timeout = approval.get("command_timeout_s")
    policy_timeout = approval.get("policy_timeout_s")
    if (
        _is_finite_number(command_timeout)
        and _is_finite_number(policy_timeout)
        and float(policy_timeout) < float(command_timeout)
    ):
        blockers.append("policy timeout must be greater than or equal to command timeout")
    publisher_timeout = approval.get("publisher_timeout_s")
    if _is_finite_number(publisher_timeout) and float(publisher_timeout) <= 0.002:
        blockers.append("publisher timeout must exceed the 2 ms control period")
    return tuple(blockers)


def hardware_policy_blockers(
    profile: Mapping[str, Any], policy_sha256: str
) -> tuple[str, ...]:
    blockers = list(hardware_activation_blockers(profile))
    if profile.get("parameter_source") == "compiled_simulation_parameters":
        approved_sha = profile.get("policy_sha256")
        if (
            isinstance(approved_sha, str)
            and approved_sha.lower() != policy_sha256.lower()
        ):
            blockers.append(
                "selected policy SHA-256 does not match the experimental profile"
            )
        return tuple(blockers)
    approval = profile.get("hardware_approval")
    approved_sha = approval.get("policy_sha256") if isinstance(approval, Mapping) else None
    if approved_sha is not None and approved_sha.lower() != policy_sha256.lower():
        blockers.append("selected policy SHA-256 is not the reviewed hardware policy")
    return tuple(blockers)


def require_hardware_activation_approval(profile: Mapping[str, Any]) -> None:
    blockers = hardware_activation_blockers(profile)
    if blockers:
        raise ValueError("hardware activation blocked: " + "; ".join(blockers))


def require_hardware_policy_approval(
    profile: Mapping[str, Any], policy_sha256: str
) -> None:
    blockers = hardware_policy_blockers(profile, policy_sha256)
    if blockers:
        raise ValueError("hardware activation blocked: " + "; ".join(blockers))


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _is_positive_finite_number(value: Any) -> bool:
    return _is_finite_number(value) and float(value) > 0.0


def _is_positive_12_vector(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 12
        and all(_is_positive_finite_number(item) for item in value)
    )
