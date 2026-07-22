"""Regenerate an adaptation curve from an existing task-level curve CSV."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.plots import plot_adaptation_curves  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix", default="adaptation_step_axis_fixed")
    parser.add_argument("--target-f1", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = {}
    with Path(args.csv).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["experiment"] != args.experiment:
                continue
            key = (row["method"], int(row["step"]))
            grouped.setdefault(key, []).append(float(row["macro_f1"]))
    if not grouped:
        raise ValueError(f"no rows found for experiment={args.experiment!r}")
    steps = sorted({step for _, step in grouped})
    methods = sorted({method for method, _ in grouped})
    trajectories = {
        method: [statistics.fmean(grouped[(method, step)]) for step in steps]
        for method in methods
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    values_path = output / f"{args.prefix}_values.csv"
    with values_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "step", "mean_task_macro_f1"]
        )
        writer.writeheader()
        for method in methods:
            for step, value in zip(steps, trajectories[method]):
                writer.writerow({
                    "method": method,
                    "step": step,
                    "mean_task_macro_f1": value,
                })
    plot_path = plot_adaptation_curves(
        trajectories,
        str(output),
        target_f1=args.target_f1,
        prefix=args.prefix,
        steps=steps,
    )
    print(plot_path)
    print(values_path)


if __name__ == "__main__":
    main()
