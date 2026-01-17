#!/usr/bin/env python3
"""Aggregate per-config metrics into per-axis means/SEM."""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


AXIS_ALIASES = {
    "background_and_agent": "background",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-config metrics (from get_metrics_from_runs.py output) "
            "into per-axis means and SEM, weighting each config equally."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tmp/generalization_metrics_summary.csv"),
        help="CSV from get_metrics_from_runs.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/generalization_axis_metrics.csv"),
        help="Path to write per-axis aggregated CSV",
    )
    return parser.parse_args()


def mean_and_sem(values: Iterable[float]) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        raise ValueError("mean_and_sem received an empty sequence.")
    mean_val = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean_val, 0.0
    variance = sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)
    sem_val = math.sqrt(variance) / math.sqrt(len(vals))
    return mean_val, sem_val


def normalize_axis(axis: str) -> str:
    return AXIS_ALIASES.get(axis, axis)


def load_config_means(csv_path: Path) -> Dict[Tuple[str, str, str], List[float]]:
    """Load per-config mean values grouped by (axis, metric, stat)."""
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            axis = normalize_axis(row["axis"])
            metric = row["metric"]
            stat = row["stat"]
            mean_val = float(row["mean"])
            grouped[(axis, metric, stat)].append(mean_val)
    return grouped


def write_axis_metrics(grouped: Dict[Tuple[str, str, str], List[float]], output_path: Path) -> None:
    os.makedirs(output_path.parent, exist_ok=True)
    fieldnames = ["axis", "metric", "stat", "n_configs", "mean", "sem", "mean_pm_sem"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (axis, metric, stat), values in sorted(grouped.items()):
            mean_val, sem_val = mean_and_sem(values)
            writer.writerow(
                {
                    "axis": axis,
                    "metric": metric,
                    "stat": stat,
                    "n_configs": len(values),
                    "mean": mean_val,
                    "sem": sem_val,
                    "mean_pm_sem": f"{mean_val:.6g}+/-{sem_val:.6g}",
                }
            )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")
    grouped = load_config_means(args.input)
    if not grouped:
        raise SystemExit("No metrics found in input CSV.")
    write_axis_metrics(grouped, args.output)
    print(f"Wrote per-axis metrics to {args.output}")


if __name__ == "__main__":
    main()
