from __future__ import annotations

import struct

import numpy as np
import pytest

from b2_control.sim_protocol import (
    COMMAND_EMERGENCY_STOP,
    COMMAND_ENABLED,
    COMMAND_REMOTE_REQUIRED,
    COMMAND_MAGIC,
    COMMAND_STRUCT,
    PROTOCOL_VERSION,
    STATE_MAGIC,
    STATE_STRUCT,
    decode_state,
    encode_command,
    quaternion_wxyz_to_rotation,
    yaw_from_rotation,
)


def test_packet_sizes_match_cpp_contract() -> None:
    assert STATE_STRUCT.size == 390
    assert COMMAND_STRUCT.size == 76


def test_decode_state_and_normalize_quaternion() -> None:
    samples = [
        *range(27),
        2.0,
        0.0,
        0.0,
        0.0,
        0.1,
        -0.2,
        0.3,
        0.4,
    ]
    packet = STATE_STRUCT.pack(
        STATE_MAGIC,
        PROTOCOL_VERSION,
        7,
        9,
        *samples,
        0x1234,
        3,
        *[float(index) for index in range(12)],
        *range(12),
        *range(12),
        77,
        5,
        0,
        -1234,
        31,
        32,
        33,
        34,
        *range(3000, 3015),
        1,
        4,
        501,
        2,
        3,
        2501,
        99,
        499.5,
        0.07,
        0.42,
        1.2,
        3.4,
    )
    state = decode_state(packet)
    assert state.sequence == 7
    np.testing.assert_array_equal(state.q, np.arange(12, dtype=np.float32))
    np.testing.assert_array_equal(state.quaternion_wxyz, [1, 0, 0, 0])
    np.testing.assert_allclose(state.remote_axes, [0.1, -0.2, 0.3, 0.4])
    assert state.remote_keys == 0x1234
    assert state.remote_valid is True
    assert state.emergency_stop_latched is True
    np.testing.assert_array_equal(state.tau_est, np.arange(12))
    np.testing.assert_array_equal(state.motor_temperature_c, np.arange(12))
    np.testing.assert_array_equal(state.motor_lost, np.arange(12))
    assert state.battery_soc_percent == 77
    assert state.battery_current_raw == -1234
    assert state.safety_fault_latched is True
    assert state.gateway_mode_code == 4
    assert state.state_samples == 501
    assert state.state_crc_errors == 2
    assert state.invalid_states == 3
    assert state.published_commands == 2501
    assert state.policy_command_sequence == 99
    assert state.dds_lowstate_hz == pytest.approx(499.5)
    assert state.publish_jitter_max_ms == pytest.approx(0.42)


def test_reject_bad_state_packet() -> None:
    with pytest.raises(ValueError, match="390 bytes"):
        decode_state(b"")


def test_encode_command_exact_layout() -> None:
    q = np.arange(12, dtype=np.float32)
    packet = encode_command(sequence=3, monotonic_ns=4, q_desired=q)
    unpacked = COMMAND_STRUCT.unpack(packet)
    assert unpacked[:5] == (COMMAND_MAGIC, PROTOCOL_VERSION, 3, 4, 1)
    np.testing.assert_array_equal(unpacked[5:], q)


def test_encode_remote_required_emergency_stop_flags() -> None:
    packet = encode_command(
        sequence=3,
        monotonic_ns=4,
        q_desired=np.zeros(12),
        emergency_stop=True,
        remote_required=True,
    )
    flags = COMMAND_STRUCT.unpack(packet)[4]
    assert flags == (
        COMMAND_ENABLED | COMMAND_EMERGENCY_STOP | COMMAND_REMOTE_REQUIRED
    )


def test_quaternion_rotation_and_yaw() -> None:
    half = np.sqrt(0.5)
    rotation = quaternion_wxyz_to_rotation([half, 0.0, 0.0, half])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert yaw_from_rotation(rotation) == pytest.approx(np.pi / 2)


def test_struct_is_explicit_little_endian() -> None:
    packet = encode_command(sequence=1, monotonic_ns=2, q_desired=np.zeros(12))
    assert packet[:4] == struct.pack("<I", COMMAND_MAGIC)
