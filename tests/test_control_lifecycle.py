from b2_control.control_lifecycle import ControlLifecycle, lifecycle_for_gateway


def test_gateway_lifecycle_distinguishes_hold_and_walk() -> None:
    assert lifecycle_for_gateway("INTERPOLATE") is ControlLifecycle.CUSTOM_INTERPOLATION
    assert lifecycle_for_gateway("POLICY", command_is_zero=True) is ControlLifecycle.HOLD
    assert lifecycle_for_gateway("POLICY", command_is_zero=False) is ControlLifecycle.WALK
    assert lifecycle_for_gateway("SAFE_DAMPING") is ControlLifecycle.FAULT

