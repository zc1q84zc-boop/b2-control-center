#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from statistics import fmean

from b2_control.policy_math import JOINT_NAMES


JOINTS = tuple(name.removesuffix("_joint") for name in JOINT_NAMES)


def values(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def rms(items: list[float]) -> float:
    return math.sqrt(fmean(value * value for value in items))


def calculate(rows: list[dict[str, str]]) -> list[dict[str, float | str | int]]:
    output: list[dict[str, float | str | int]] = []
    for joint in JOINTS:
        raw = values(rows, f"raw_{joint}")
        applied = values(rows, f"applied_{joint}")
        target = values(rows, f"target_{joint}")
        actual = values(rows, f"actual_{joint}")
        error = [sent - measured for sent, measured in zip(target, actual)]
        target_range = max(target) - min(target)
        actual_range = max(actual) - min(actual)
        raw_mean = fmean(raw)
        clipped = sum(
            abs(before - after) > 1e-7
            for before, after in zip(raw, applied)
        )
        output.append(
            {
                "joint": joint,
                "samples": len(rows),
                "raw_min": min(raw),
                "raw_max": max(raw),
                "raw_mean": raw_mean,
                "raw_std": math.sqrt(
                    fmean((item - raw_mean) ** 2 for item in raw)
                ),
                "clip_count": clipped,
                "clip_fraction": clipped / len(rows),
                "target_min_rad": min(target),
                "target_max_rad": max(target),
                "target_range_rad": target_range,
                "actual_min_rad": min(actual),
                "actual_max_rad": max(actual),
                "actual_range_rad": actual_range,
                "movement_ratio": (
                    actual_range / target_range if target_range > 1e-9 else 0.0
                ),
                "error_mean_rad": fmean(error),
                "error_mae_rad": fmean(abs(item) for item in error),
                "error_rmse_rad": rms(error),
                "error_max_abs_rad": max(abs(item) for item in error),
            }
        )
    return output


def write_outputs(
    prefix: Path,
    source: Path,
    summary: list[dict[str, float | str | int]],
) -> None:
    with prefix.with_name(prefix.name + "-stats.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Policy target / measured tracking statistics",
        "",
        f"Source: {source}",
        "",
        "| Joint | raw min..max | clipped | target p2p | actual p2p | actual/target | MAE | RMSE | max error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {joint} | {raw_min:.3f}..{raw_max:.3f} | {clip_count}/{samples} "
            "({clip_fraction:.1%}) | {target_range_rad:.3f} | "
            "{actual_range_rad:.3f} | {movement_ratio:.2f} | "
            "{error_mae_rad:.3f} | {error_rmse_rad:.3f} | "
            "{error_max_abs_rad:.3f} |".format(**row)
        )
    prefix.with_name(prefix.name + "-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def chart(
    path: Path,
    rows: list[dict[str, str]],
    *,
    title: str,
    first_prefix: str,
    second_prefix: str,
    first_label: str,
    second_label: str,
    first_color: str,
    second_color: str,
    clip: float | None = None,
) -> None:
    width, height = 1500, 1640
    panel_w, panel_h = 450, 330
    left, top, gap_x, gap_y = 70, 120, 30, 50
    times = values(rows, "elapsed_s")
    start = times[0]
    times = [item - start for item in times]
    duration = max(times[-1], 1e-9)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="70" y="50" font-family="sans-serif" font-size="29" font-weight="bold">{html.escape(title)}</text>',
        f'<line x1="70" y1="82" x2="115" y2="82" stroke="{first_color}" stroke-width="4"/>',
        f'<text x="125" y="89" font-family="sans-serif" font-size="18">{html.escape(first_label)}</text>',
        f'<line x1="330" y1="82" x2="375" y2="82" stroke="{second_color}" stroke-width="4"/>',
        f'<text x="385" y="89" font-family="sans-serif" font-size="18">{html.escape(second_label)}</text>',
    ]
    for index, joint in enumerate(JOINTS):
        row, column = divmod(index, 3)
        x = left + column * (panel_w + gap_x)
        y = top + row * (panel_h + gap_y)
        px, py, pw, ph = x + 58, y + 34, panel_w - 78, panel_h - 76
        first = values(rows, f"{first_prefix}_{joint}")
        second = values(rows, f"{second_prefix}_{joint}")
        low, high = min(first + second), max(first + second)
        if clip is not None:
            low, high = min(low, -clip), max(high, clip)
        padding = max((high - low) * 0.08, 0.02)
        low, high = low - padding, high + padding

        def points(items: list[float]) -> str:
            return " ".join(
                f"{px + pw * time_value / duration:.2f},"
                f"{py + ph * (high - value) / (high - low):.2f}"
                for time_value, value in zip(times, items)
            )

        svg.extend(
            [
                f'<text x="{x}" y="{y + 20}" font-family="sans-serif" font-size="19" font-weight="bold">{joint}</text>',
                f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#9ca3af"/>',
                f'<polyline points="{points(first)}" fill="none" stroke="{first_color}" stroke-width="2"/>',
                f'<polyline points="{points(second)}" fill="none" stroke="{second_color}" stroke-width="2"/>',
                f'<text x="{px - 6}" y="{py + 5}" text-anchor="end" font-family="monospace" font-size="13">{high:.2f}</text>',
                f'<text x="{px - 6}" y="{py + ph}" text-anchor="end" font-family="monospace" font-size="13">{low:.2f}</text>',
                f'<text x="{px}" y="{py + ph + 22}" font-family="monospace" font-size="13">0 s</text>',
                f'<text x="{px + pw}" y="{py + ph + 22}" text-anchor="end" font-family="monospace" font-size="13">{duration:.1f} s</text>',
            ]
        )
        if clip is not None:
            for bound in (-clip, clip):
                bound_y = py + ph * (high - bound) / (high - low)
                svg.append(
                    f'<line x1="{px}" y1="{bound_y:.2f}" x2="{px + pw}" y2="{bound_y:.2f}" stroke="#dc2626" stroke-dasharray="5 5"/>'
                )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics_csv")
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--prefix", default="policy-tracking")
    args = parser.parse_args()
    source = Path(args.diagnostics_csv)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream) if row["phase"] == "motion"
        ]
    if not rows:
        raise SystemExit(f"no motion samples in {source}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / args.prefix
    write_outputs(prefix, source, calculate(rows))
    chart(
        prefix.with_name(prefix.name + "-tracking.svg"),
        rows,
        title="Policy q_target vs measured q_actual (50 Hz motion samples)",
        first_prefix="target",
        second_prefix="actual",
        first_label="policy q_target",
        second_label="measured q_actual",
        first_color="#2563eb",
        second_color="#f97316",
    )
    chart(
        prefix.with_name(prefix.name + "-raw-action.svg"),
        rows,
        title="ONNX raw output vs applied output after [-3,+3] clip",
        first_prefix="raw",
        second_prefix="applied",
        first_label="ONNX raw action",
        second_label="applied action",
        first_color="#475569",
        second_color="#a21caf",
        clip=3.0,
    )
    for suffix in ("tracking.svg", "raw-action.svg", "stats.csv", "summary.md"):
        print(prefix.with_name(prefix.name + "-" + suffix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
