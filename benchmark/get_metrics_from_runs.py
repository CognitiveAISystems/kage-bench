#!/usr/bin/env python3
"""Aggregate eval_metrics_summary.yaml files into a CSV with mean and SEM."""

"""
# Aggregating generalization metrics

This script reads every `eval_metrics_summary.yaml` under `runs/generalization_benchmark`, groups runs by axis and config id, and writes a CSV with mean and standard error (SEM) for each metric/stat pair.

## Usage

- Default paths (writes `benchmark/generalization_metrics_summary.csv`):
  ```bash
  python benchmark/get_metrics_from_runs.py
  ```
- Custom runs directory and output file:
  ```bash
  python benchmark/get_metrics_from_runs.py \
    --runs-dir runs/generalization_benchmark \
    --output tmp/generalization_metrics_summary.csv
  ```
  Progress bars (powered by `rich`) show per-axis and per-run aggregation status.

## Output columns

- `axis`: generalization axis directory name (e.g., `background`)
- `config_id`: the `config_X` id parsed from folder names like `config_1__3__...`
- `metric`: top-level metric key from the YAML
- `stat`: stat inside the metric (e.g., `mean`, `max`, `median`, `min`)
- `n`: number of runs aggregated for that config/stat
- `mean`: arithmetic mean across runs
- `sem`: standard error of the mean
- `mean_pm_sem`: formatted string `mean+/-sem`

## Example: quick sanity check

After running the script:
```bash
python benchmark/get_metrics_from_runs.py
head -n 5 benchmark/generalization_metrics_summary.csv
```
Expected first lines look like:
```
axis,config_id,metric,stat,n,mean,sem,mean_pm_sem
agent_appearance,1,eval/avg_eval_config_jump_count,max,10,250.0867,0.201524,250.087+/-0.201524
...
```
"""

import argparse
import csv
import math
import re
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


RunMetrics = Dict[Tuple[str, str], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate eval_metrics_summary.yaml files across runs for each "
            "generalization axis and config."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/generalization_benchmark"),
        help="Root directory that contains axis folders (default: runs/generalization_benchmark).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/generalization_metrics_summary.csv"),
        help="Path to write the aggregated CSV (default: tmp/generalization_metrics_summary.csv).",
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


def flatten_run_metrics(run_metrics: Dict) -> RunMetrics:
    """Flatten nested metric stats to (metric, stat) -> value."""
    flattened: RunMetrics = {}
    for metric_name, stats in run_metrics.items():
        if isinstance(stats, dict):
            for stat_name, value in stats.items():
                flattened[(metric_name, str(stat_name))] = float(value)
        else:
            # Handle scalar metrics, keep stat key as "value".
            flattened[(metric_name, "value")] = float(stats)
    return flattened


def aggregate_axis(axis_dir: Path, axis_name: str, progress: Optional[Progress] = None) -> List[Dict[str, object]]:
    """Aggregate metrics for a single axis, optionally updating a progress bar."""
    rows: List[Dict[str, object]] = []
    runs_by_config: Dict[str, List[RunMetrics]] = defaultdict(list)
    pattern = re.compile(r"config_(\d+)__\d+__\d+")

    run_dirs = [
        run_dir
        for run_dir in axis_dir.iterdir()
        if run_dir.is_dir() and pattern.fullmatch(run_dir.name)
    ]

    task_id = None
    if progress and run_dirs:
        task_id = progress.add_task(f"[cyan]{axis_name}", total=len(run_dirs))

    for run_dir in run_dirs:
        match = pattern.fullmatch(run_dir.name)
        if not match:
            if task_id:
                progress.advance(task_id)
            continue
        config_id = match.group(1)
        metrics_file = run_dir / "eval_metrics_summary.yaml"
        if metrics_file.exists():
            with metrics_file.open() as f:
                run_metrics = yaml.safe_load(f)
            runs_by_config[config_id].append(flatten_run_metrics(run_metrics))
        if task_id:
            progress.advance(task_id)

    for config_id in sorted(runs_by_config.keys(), key=lambda x: int(x)):
        metric_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for run_metrics in runs_by_config[config_id]:
            for key, value in run_metrics.items():
                metric_values[key].append(value)

        for (metric, stat), values in sorted(metric_values.items()):
            mean_val, sem_val = mean_and_sem(values)
            rows.append(
                {
                    "axis": axis_name,
                    "config_id": config_id,
                    "metric": metric,
                    "stat": stat,
                    "n": len(values),
                    "mean": mean_val,
                    "sem": sem_val,
                    "mean_pm_sem": f"{mean_val:.6g}+/-{sem_val:.6g}",
                }
            )
    return rows


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["axis", "config_id", "metric", "stat", "n", "mean", "sem", "mean_pm_sem"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {args.runs_dir}")
    os.makedirs(args.output.parent, exist_ok=True)
    axis_dirs = sorted(p for p in args.runs_dir.iterdir() if p.is_dir())
    if not axis_dirs:
        raise SystemExit("No axis directories found to aggregate.")
    all_rows: List[Dict[str, object]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        axis_task = progress.add_task("[green]Axes", total=len(axis_dirs))
        for axis_dir in axis_dirs:
            axis_rows = aggregate_axis(axis_dir, axis_dir.name, progress)
            all_rows.extend(axis_rows)
            progress.advance(axis_task)
    if not all_rows:
        raise SystemExit("No metrics found to aggregate.")
    write_csv(all_rows, args.output)
    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
