import pytest

from b2_control.cli_sim_policy import build_parser, validate_args
from b2_control.cli_hw_dryrun import build_parser as build_hw_parser


def test_parser_accepts_bounded_capture_duration() -> None:
    args = build_parser().parse_args(
        ["policy-bundle", "--speed", "0.25", "--duration", "10"]
    )
    assert args.speed == 0.25
    assert args.duration == 10
    assert args.target_slew_rad_s == 8.0
    assert args.disable_target_slew is False
    assert args.action_clip_abs is None
    assert args.diagnostics_csv is None


def test_parser_accepts_stable_walk_phases() -> None:
    args = build_parser().parse_args(
        [
            "policy-bundle",
            "--speed",
            "0.25",
            "--duration",
            "5",
            "--warmup-hold-duration",
            "3",
            "--keep-hold-after-duration",
        ]
    )
    assert args.warmup_hold_duration == 3.0
    assert args.keep_hold_after_duration is True


def test_parser_accepts_clip_unlimited_slew_and_diagnostics() -> None:
    args = build_parser().parse_args(
        [
            "policy-bundle",
            "--action-clip-abs",
            "3",
            "--disable-target-slew",
            "--diagnostics-csv",
            "tracking.csv",
        ]
    )
    assert args.action_clip_abs == 3.0
    assert args.disable_target_slew is True
    assert args.diagnostics_csv == "tracking.csv"


def test_parser_accepts_interactive_keyboard_controls() -> None:
    args = build_parser().parse_args(
        [
            "policy-bundle",
            "--interactive-keyboard",
            "--yaw-step",
            "0.1",
            "--max-linear-speed",
            "0.25",
            "--max-yaw-rate",
            "0.4",
        ]
    )
    validate_args(args)
    assert args.interactive_keyboard is True
    assert args.max_linear_speed == 0.25


def test_parser_accepts_browser_operator_ui_and_fixed_translation_speeds() -> None:
    args = build_parser().parse_args(
        [
            "policy-bundle",
            "--operator-ui",
            "--operator-ui-port",
            "9876",
            "--no-open-browser",
            "--initial-linear-speed",
            "0.6",
        ]
    )
    validate_args(args)
    assert args.operator_ui is True
    assert args.operator_ui_port == 9876
    assert args.max_linear_speed == 0.90
    assert args.max_lateral_speed == 0.25
    assert args.initial_linear_speed == 0.6
    assert args.no_open_browser is True


def test_parser_accepts_acknowledged_managed_hardware_console() -> None:
    args = build_parser().parse_args(
        [
            "policy-bundle",
            "--operator-ui",
            "--hardware-console",
            "enp8s0",
            "--acknowledge-b2-low-level-control",
        ]
    )
    validate_args(args)
    assert args.hardware_console == "enp8s0"


def test_managed_hardware_console_requires_ui_and_acknowledgement() -> None:
    without_ui = build_parser().parse_args(
        [
            "policy-bundle",
            "--hardware-console",
            "enp8s0",
            "--acknowledge-b2-low-level-control",
        ]
    )
    with pytest.raises(SystemExit, match="requires --operator-ui"):
        validate_args(without_ui)

    without_ack = build_parser().parse_args(
        ["policy-bundle", "--operator-ui", "--hardware-console", "enp8s0"]
    )
    with pytest.raises(SystemExit, match="requires --acknowledge"):
        validate_args(without_ack)


def test_terminal_and_browser_operator_modes_are_rejected_together() -> None:
    args = build_parser().parse_args(
        ["policy-bundle", "--interactive-keyboard", "--operator-ui"]
    )
    with pytest.raises(SystemExit, match="mutually exclusive"):
        validate_args(args)


def test_parser_accepts_hand_remote_control() -> None:
    args = build_parser().parse_args(
        ["policy-bundle", "--remote-control", "--remote-deadzone", "0.12"]
    )
    validate_args(args)
    assert args.remote_control is True
    assert args.remote_deadzone == 0.12


def test_hand_remote_can_use_browser_for_mode_switch_and_monitoring() -> None:
    args = build_parser().parse_args(
        ["policy-bundle", "--remote-control", "--operator-ui"]
    )
    validate_args(args)
    assert args.remote_control is True
    assert args.operator_ui is True


def test_hold_and_interactive_keyboard_are_rejected() -> None:
    args = build_parser().parse_args(
        ["policy-bundle", "--hold", "--interactive-keyboard"]
    )
    with pytest.raises(SystemExit, match="mutually exclusive"):
        validate_args(args)


def test_hardware_shadow_parser_accepts_observation_diagnostics() -> None:
    args = build_hw_parser().parse_args(
        ["policy-bundle", "--hold", "--diagnostics-csv", "shadow.csv"]
    )
    assert args.hold is True
    assert args.diagnostics_csv == "shadow.csv"
