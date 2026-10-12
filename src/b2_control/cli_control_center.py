"""Entry point for the standalone B2 Control Center."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading

from .control_center import (
    ControlCenterPaths,
    ControlCenterSupervisor,
    ControlCenterWeb,
)
from .policy_registry import PolicyRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone B2 Control Center")
    parser.add_argument("--port", type=int, default=8770, help="loopback web port")
    parser.add_argument(
        "--no-open-browser", action="store_true", help="print URL without opening a browser"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path.home() / ".config" / "b2-control-center" / "policies.json",
        help="persistent policy registry path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "control-center.json",
        help="Control Center defaults",
    )
    return parser


def _load_defaults(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read Control Center config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("Control Center config must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ControlCenterPaths.for_repository(PROJECT_ROOT)
    paths.validate()
    registry = PolicyRegistry(args.registry)
    defaults = _load_defaults(args.config)
    bundles = defaults.get("default_policy_bundles", [])
    if isinstance(bundles, list):
        known = {profile.bundle for profile in registry.profiles()}
        for item in bundles:
            if not isinstance(item, dict):
                continue
            bundle = item.get("bundle")
            name = item.get("name")
            adapter = item.get("adapter", "b2_45x12_v1")
            if not all(isinstance(value, str) for value in (bundle, name, adapter)):
                continue
            resolved = str(Path(bundle).expanduser().resolve(strict=False))
            if resolved in known:
                continue
            try:
                profile = registry.add(bundle, name=name, adapter=adapter)
                known.add(profile.bundle)
            except ValueError as exc:
                print(f"Skipping unavailable default policy {name}: {exc}")
    supervisor = ControlCenterSupervisor(paths, registry)
    web = ControlCenterWeb(supervisor, port=args.port)
    url = web.open(open_browser=not args.no_open_browser)
    print(f"B2 Control Center: {url}", flush=True)
    print("Startup is passive: no simulator or robot connection is automatic.", flush=True)
    stopped = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        stopped.wait()
    finally:
        print("Closing active session safely...", flush=True)
        supervisor.close()
        web.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
