from __future__ import annotations

import argparse
from collections.abc import Sequence

from .parameter_profile import (
    hardware_activation_blockers,
    load_parameter_profile,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether a B2 hardware parameter profile is complete"
    )
    parser.add_argument("profile", help="Path to the hardware parameter JSON")
    args = parser.parse_args(argv)

    try:
        profile = load_parameter_profile(args.profile)
    except ValueError as exc:
        print(f"BLOCKED: {exc}")
        return 2

    blockers = hardware_activation_blockers(profile)
    if blockers:
        print(f"BLOCKED: {len(blockers)} hardware approval requirement(s) remain")
        for blocker in blockers:
            print(f"- {blocker}")
        return 2

    print("PASS: hardware parameter profile is complete and explicitly approved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
