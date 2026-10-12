from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

import numpy as np


STATE_MAGIC = 0x54533242
COMMAND_MAGIC = 0x4D433242
PROTOCOL_VERSION = 4
STATE_PORT = 15000
COMMAND_PORT = 15001
STATE_STRUCT = struct.Struct("<IIQQ35fII12f12B12IBBHi4B15HII5Q5f")
COMMAND_STRUCT = struct.Struct("<IIQQI12f")

COMMAND_ENABLED = 1 << 0
COMMAND_EMERGENCY_STOP = 1 << 1
COMMAND_REMOTE_REQUIRED = 1 << 2
REMOTE_VALID = 1 << 0
REMOTE_EMERGENCY_STOP_LATCHED = 1 << 1
SAFETY_FAULT_LATCHED = 1 << 0
GATEWAY_MODE_NAMES = {
    0: "UNKNOWN",
    1: "WAIT_FIRST_POLICY",
    2: "INTERPOLATE",
    3: "SAFE_HOLD",
    4: "POLICY",
    5: "SAFE_DAMPING",
    6: "SAFE_HOLD_EXIT",
}


@dataclass(frozen=True)
class SimState:
    sequence: int
    monotonic_ns: int
    q: np.ndarray
    dq: np.ndarray
    gyro: np.ndarray
    quaternion_wxyz: np.ndarray
    remote_axes: np.ndarray
    remote_keys: int
    remote_valid: bool
    emergency_stop_latched: bool
    tau_est: np.ndarray
    motor_temperature_c: np.ndarray
    motor_lost: np.ndarray
    battery_soc_percent: int
    battery_status: int
    battery_current_raw: int
    battery_temperature_c: np.ndarray
    battery_cell_mv: np.ndarray
    safety_fault_latched: bool
    gateway_mode_code: int
    state_samples: int
    state_crc_errors: int
    invalid_states: int
    published_commands: int
    policy_command_sequence: int
    dds_lowstate_hz: float
    publish_jitter_ms: float
    publish_jitter_max_ms: float
    lowstate_age_ms: float
    policy_command_age_ms: float


def decode_state(packet: bytes) -> SimState:
    if len(packet) != STATE_STRUCT.size:
        raise ValueError(f"state packet must be {STATE_STRUCT.size} bytes")
    values = STATE_STRUCT.unpack(packet)
    if values[:2] != (STATE_MAGIC, PROTOCOL_VERSION):
        raise ValueError("state packet magic/version mismatch")
    samples = np.asarray(values[4:39], dtype=np.float32)
    if not np.all(np.isfinite(samples)):
        raise ValueError("state packet contains non-finite samples")
    quaternion = samples[27:31].copy()
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("state quaternion is invalid")
    quaternion /= norm
    return SimState(
        sequence=values[2],
        monotonic_ns=values[3],
        q=samples[:12].copy(),
        dq=samples[12:24].copy(),
        gyro=samples[24:27].copy(),
        quaternion_wxyz=quaternion,
        remote_axes=samples[31:35].copy(),
        remote_keys=int(values[39]),
        remote_valid=bool(values[40] & REMOTE_VALID),
        emergency_stop_latched=bool(
            values[40] & REMOTE_EMERGENCY_STOP_LATCHED
        ),
        tau_est=np.asarray(values[41:53], dtype=np.float32),
        motor_temperature_c=np.asarray(values[53:65], dtype=np.uint8),
        motor_lost=np.asarray(values[65:77], dtype=np.uint32),
        battery_soc_percent=int(values[77]),
        battery_status=int(values[78]),
        battery_current_raw=int(values[80]),
        battery_temperature_c=np.asarray(values[81:85], dtype=np.uint8),
        battery_cell_mv=np.asarray(values[85:100], dtype=np.uint16),
        safety_fault_latched=bool(values[100] & SAFETY_FAULT_LATCHED),
        gateway_mode_code=int(values[101]),
        state_samples=int(values[102]),
        state_crc_errors=int(values[103]),
        invalid_states=int(values[104]),
        published_commands=int(values[105]),
        policy_command_sequence=int(values[106]),
        dds_lowstate_hz=float(values[107]),
        publish_jitter_ms=float(values[108]),
        publish_jitter_max_ms=float(values[109]),
        lowstate_age_ms=float(values[110]),
        policy_command_age_ms=float(values[111]),
    )


def encode_command(
    *,
    sequence: int,
    monotonic_ns: int,
    q_desired: Any,
    enabled: bool = True,
    emergency_stop: bool = False,
    remote_required: bool = False,
) -> bytes:
    q = np.asarray(q_desired, dtype=np.float32)
    if q.shape != (12,) or not np.all(np.isfinite(q)):
        raise ValueError("q_desired must be a finite float32 vector with shape (12,)")
    flags = 0
    if enabled:
        flags |= COMMAND_ENABLED
    if emergency_stop:
        flags |= COMMAND_EMERGENCY_STOP
    if remote_required:
        flags |= COMMAND_REMOTE_REQUIRED
    return COMMAND_STRUCT.pack(
        COMMAND_MAGIC,
        PROTOCOL_VERSION,
        sequence,
        monotonic_ns,
        flags,
        *q,
    )


def quaternion_wxyz_to_rotation(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must be a finite vector with shape (4,)")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion norm must be nonzero")
    w, x, y, z = q / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def yaw_from_rotation(rotation: Any) -> float:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite matrix with shape (3, 3)")
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))
