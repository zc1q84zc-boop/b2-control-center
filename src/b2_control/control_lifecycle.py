"""Shared, explicit lifecycle names for simulation and real-B2 control.

The lifecycle is deliberately separate from process supervision states such as
``SIM_STARTING``.  It is the operator-facing source of truth for who owns the
robot and what the low-level output is doing.
"""

from __future__ import annotations

from enum import StrEnum


class ControlLifecycle(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    READ_ONLY_CHECK = "READ_ONLY_CHECK"
    NATIVE_CONFIRMED = "NATIVE_CONFIRMED"
    ARMED = "ARMED"
    CUSTOM_INTERPOLATION = "CUSTOM_INTERPOLATION"
    HOLD = "HOLD"
    WALK = "WALK"
    STOPPING = "STOPPING"
    NATIVE = "NATIVE"
    FAULT = "FAULT"


GATEWAY_TO_LIFECYCLE = {
    "WAIT_FIRST_POLICY": ControlLifecycle.ARMED,
    "INTERPOLATE": ControlLifecycle.CUSTOM_INTERPOLATION,
    "SAFE_HOLD": ControlLifecycle.HOLD,
    "SAFE_HOLD_EXIT": ControlLifecycle.STOPPING,
    "SAFE_DAMPING": ControlLifecycle.FAULT,
}


def lifecycle_for_gateway(
    gateway_mode: str,
    *,
    command_is_zero: bool = True,
) -> ControlLifecycle:
    """Map the C++ output mode to the shared operator lifecycle."""

    if gateway_mode == "POLICY":
        return ControlLifecycle.HOLD if command_is_zero else ControlLifecycle.WALK
    return GATEWAY_TO_LIFECYCLE.get(gateway_mode, ControlLifecycle.ARMED)


__all__ = ["ControlLifecycle", "lifecycle_for_gateway"]
