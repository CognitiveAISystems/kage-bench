#!/usr/bin/env python3
"""Aggregate multi-eval TensorBoard scalars into a CSV with mean/SEM per config."""

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

from tensorboard.backend.event_processing import event_accumulator
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    train_label: str
    train_config_id: Optional[str]
    seed: int
    timestamp: int
    event_file: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate TensorBoard scalars from multi-eval runs into a CSV "
            "with mean and SEM across seeds."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/test_multi"),
        help="Root directory containing multi-eval run folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/multi_eval_metrics_summary.csv"),
        help="Path to write the aggregated CSV.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Include all runs even if multiple timestamps share the same seed.",
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


def find_event_file(run_dir: Path) -> Optional[Path]:
    files = sorted(
        run_dir.glob("events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def parse_run_dir(run_dir: Path) -> Optional[RunInfo]:
    match = re.match(r"(.+)__([0-9]+)__([0-9]+)$", run_dir.name)
    if not match:
        return None
    train_label = match.group(1)
    seed = int(match.group(2))
    timestamp = int(match.group(3))
    config_match = re.search(r"config_([0-9]+)", train_label)
    train_config_id = config_match.group(1) if config_match else None
    event_file = find_event_file(run_dir)
    return RunInfo(
        run_dir=run_dir,
        train_label=train_label,
        train_config_id=train_config_id,
        seed=seed,
        timestamp=timestamp,
        event_file=event_file,
    )


def select_latest_per_seed(runs: Iterable[RunInfo]) -> List[RunInfo]:
    latest: Dict[Tuple[str, int], RunInfo] = {}
    for run in runs:
        key = (run.train_label, run.seed)
        if key not in latest or run.timestamp > latest[key].timestamp:
            latest[key] = run
    return list(latest.values())


def is_config_metric_tag(tag: str) -> bool:
    return ".yaml/" in tag and not tag.startswith("main_metrics/")


def gather_runs(runs_dir: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run_info = parse_run_dir(run_dir)
        if run_info:
            runs.append(run_info)
    return runs


def aggregate_runs(
    runs: Iterable[RunInfo],
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> List[Dict[str, object]]:
    values: DefaultDict[
        Tuple[str, Optional[str], str, str, int], List[float]
    ] = defaultdict(list)

    for run in runs:
        if not run.event_file or not run.event_file.exists():
            if progress and task_id is not None:
                progress.advance(task_id)
            continue
        acc = event_accumulator.EventAccumulator(
            str(run.event_file),
            size_guidance={"scalars": 0},
        )
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            if not is_config_metric_tag(tag):
                continue
            label, metric = tag.split("/", 1)
            for event in acc.Scalars(tag):
                values[(run.train_label, run.train_config_id, label, metric, int(event.step))].append(
                    float(event.value)
                )
        if progress and task_id is not None:
            progress.advance(task_id)

    rows: List[Dict[str, object]] = []
    for (train_label, train_config_id, eval_label, metric, step), vals in values.items():
        mean_val, sem_val = mean_and_sem(vals)
        rows.append(
            {
                "train_label": train_label,
                "train_config_id": train_config_id or "",
                "eval_label": eval_label,
                "metric": metric,
                "step": step,
                "n": len(vals),
                "mean": mean_val,
                "sem": sem_val,
                "mean_pm_sem": f"{mean_val:.6g}+/-{sem_val:.6g}",
            }
        )
    return rows


def sort_key(row: Dict[str, object]) -> Tuple[object, ...]:
    config_id = str(row["train_config_id"])
    if config_id.isdigit():
        config_key: Tuple[object, ...] = (0, int(config_id), row["train_label"])
    else:
        config_key = (1, row["train_label"])
    return (
        *config_key,
        row["eval_label"],
        row["metric"],
        int(row["step"]),
    )


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "train_label",
        "train_config_id",
        "eval_label",
        "metric",
        "step",
        "n",
        "mean",
        "sem",
        "mean_pm_sem",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {args.runs_dir}")

    runs = gather_runs(args.runs_dir)
    if not runs:
        raise SystemExit("No run directories found to aggregate.")

    selected_runs = runs if args.no_dedupe else select_latest_per_seed(runs)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Aggregating runs", total=len(selected_runs))
        rows = aggregate_runs(selected_runs, progress=progress, task_id=task)

    if not rows:
        raise SystemExit("No metrics found to aggregate.")

    rows_sorted = sorted(rows, key=sort_key)
    os.makedirs(args.output.parent, exist_ok=True)
    write_csv(rows_sorted, args.output)
    print(f"Wrote {len(rows_sorted)} rows to {args.output}")


if __name__ == "__main__":
    main()
