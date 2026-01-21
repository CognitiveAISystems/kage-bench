#!/usr/bin/env python3
"""Plot rain/eval metrics for every config across all axes.

Run:
    uv run python benchmark/plot_all_configs.py

Outputs:
    PDFs per axis/config to tmp/plots/<axis>/, one file per metric.
"""

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from rich.progress import Progress, TaskID

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tensorboard.backend.event_processing import event_accumulator


def _set_icml_rcparams() -> None:
    """Typography and styling."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Typography
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            # Styling
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "lines.linewidth": 1.9,
            "lines.markersize": 6.0,
            "legend.frameon": False,
        }
    )


_STYLE_SET = False


def _ensure_icml_style() -> None:
    global _STYLE_SET
    if not _STYLE_SET:
        _set_icml_rcparams()
        _STYLE_SET = True


IMPORTANT_BASE_METRICS: Sequence[str] = (
    "passed_distance",
    "progress",
    "success_once",
    "episodic_return",
)

HUMAN_YLABELS: Dict[str, str] = {
    "passed_distance": "Passed Distance",
    "progress": "Progress",
    "success_once": "Success Rate",
    "episodic_return": "Return",
}

Y_AXIS_CONFIG: Dict[str, Tuple[Tuple[float, float], Sequence[float]]] = {
    "passed_distance": ((0.0, 530.0), (0, 100, 200, 300, 400, 500)),
    "progress": ((0.0, 1.1), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
    "success_once": ((0.0, 1.1), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
    "episodic_return": ((-4000.0, 200.0), (-4000, -3000, -2000, -1000, 0)),
}

STEP_AXIS_LIMITS = (0.0, 25_000_000.0)
STEP_TICKS = (0, 5_000_000, 10_000_000, 15_000_000, 20_000_000, 25_000_000)


def _apply_axis_bounds(ax: mpl.axes.Axes, base_metric: str) -> None:
    ax.set_xlim(STEP_AXIS_LIMITS)
    ax.set_xticks(STEP_TICKS)
    xfmt = ScalarFormatter(useMathText=False)
    xfmt.set_scientific(True)
    xfmt.set_powerlimits((7, 7))
    ax.xaxis.set_major_formatter(xfmt)

    config = Y_AXIS_CONFIG.get(base_metric)
    if config:
        ylim, yticks = config
        ax.set_ylim(ylim)
        ax.set_yticks(list(yticks))


def mean_and_sem(values: Iterable[float]) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        raise ValueError("mean_and_sem received an empty sequence.")
    mean_val = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean_val, 0.0
    variance = sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)
    return mean_val, math.sqrt(variance) / math.sqrt(len(vals))


def find_event_file(run_dir: Path) -> Optional[Path]:
    files = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_metric_series(event_path: Path, tag: str) -> List[Tuple[int, float]]:
    acc = event_accumulator.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    acc.Reload()
    scalar_tags = acc.Tags().get("scalars", [])
    if tag not in scalar_tags:
        return []
    return [(ev.step, float(ev.value)) for ev in acc.Scalars(tag)]


def metric_tags(split: str, base_metric: str) -> Tuple[str, str]:
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    # prefer config-level tag, fall back to plain tag
    return (
        f"eval/avg_{split}_config_{base_metric}",
        f"eval/avg_{split}_{base_metric}",
    )


def pick_step_values(
    aggregated: Dict[str, DefaultDict[int, List[float]]],
    candidate_tags: Sequence[str],
) -> DefaultDict[int, List[float]]:
    for tag in candidate_tags:
        step_values = aggregated.get(tag)
        if step_values:
            return step_values
    return defaultdict(list)


def aggregate_across_runs(
    run_event_files: Sequence[Path],
    tags: Sequence[str],
) -> Dict[str, DefaultDict[int, List[float]]]:
    aggregated: Dict[str, DefaultDict[int, List[float]]] = {tag: defaultdict(list) for tag in tags}
    for event_file in run_event_files:
        for tag in tags:
            for step, value in load_metric_series(event_file, tag):
                aggregated[tag][step].append(value)
    return aggregated


def _plot_line_with_sem(
    ax: mpl.axes.Axes,
    *,
    steps: Sequence[int],
    means: Sequence[float],
    sems: Sequence[float],
    label: str,
    color: str,
) -> None:
    lower = [m - s for m, s in zip(means, sems)]
    upper = [m + s for m, s in zip(means, sems)]
    ax.plot(steps, means, label=label, color=color, zorder=4)
    ax.fill_between(steps, lower, upper, color=color, alpha=0.18, linewidth=0.0, zorder=2)


def plot_metric_train_vs_eval(
    axis_name: str,
    config_id: str,
    base_metric: str,
    *,
    train_step_values: DefaultDict[int, List[float]],
    eval_step_values: DefaultDict[int, List[float]],
    output_dir: Path,
    figsize: Tuple[float, float] = (3.35, 2.4),
) -> None:
    _ensure_icml_style()
    if not train_step_values and not eval_step_values:
        return

    fig, ax = plt.subplots(figsize=figsize)
    palette = {"train": "#1f77b4", "eval": "#d62728"}

    if train_step_values:
        train_steps = sorted(train_step_values.keys())
        train_means, train_sems = zip(*(mean_and_sem(train_step_values[s]) for s in train_steps))
        _plot_line_with_sem(ax, steps=train_steps, means=train_means, sems=train_sems, label="Train", color=palette["train"])

    if eval_step_values:
        eval_steps = sorted(eval_step_values.keys())
        eval_means, eval_sems = zip(*(mean_and_sem(eval_step_values[s]) for s in eval_steps))
        _plot_line_with_sem(ax, steps=eval_steps, means=eval_means, sems=eval_sems, label="Eval", color=palette["eval"])

    ax.set_xlabel("Step")
    ax.set_ylabel(HUMAN_YLABELS.get(base_metric, base_metric))
    # ax.set_title(f"{axis_name} / config {config_id}", pad=6, loc="left", fontweight="semibold")
    _apply_axis_bounds(ax, base_metric)
    ax.grid(True, which="major", alpha=0.2, linewidth=0.6)
    ax.grid(False, which="minor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.legend(
        ncol=2,
        loc="best",
        handlelength=2.8,
        columnspacing=1.2,
        handletextpad=0.5,
        borderpad=0.3,
        fontsize=11,
    )
    fig.tight_layout(pad=0.4)

    safe_metric = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_metric)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{axis_name}_config_{config_id}_{safe_metric}.pdf"
    fig.savefig(out_path)
    plt.close(fig)


def plot_metrics_for_config(
    axis_dir: Path,
    config_id: str,
    base_metrics: Sequence[str],
    output_dir: Path,
) -> None:
    axis_dir = Path(axis_dir)
    config_id = str(config_id)

    run_dirs = sorted(axis_dir.glob(f"config_{config_id}__*"))
    event_files: List[Path] = []
    for run_dir in run_dirs:
        event_file = find_event_file(run_dir)
        if event_file:
            event_files.append(event_file)

    if not event_files:
        return

    tags: List[str] = []
    for base_metric in base_metrics:
        for split in ("train", "eval"):
            tags.extend(metric_tags(split, base_metric))

    aggregated = aggregate_across_runs(event_files, tags)
    axis_name = axis_dir.name

    for base_metric in base_metrics:
        plot_metric_train_vs_eval(
            axis_name,
            config_id,
            base_metric,
            train_step_values=pick_step_values(aggregated, metric_tags("train", base_metric)),
            eval_step_values=pick_step_values(aggregated, metric_tags("eval", base_metric)),
            output_dir=output_dir,
        )


def gather_config_ids(axis_dir: Path) -> List[str]:
    ids = set()
    pattern = re.compile(r"config_(\d+)__\d+__\d+")
    for run_dir in axis_dir.iterdir():
        if not run_dir.is_dir():
            continue
        match = pattern.fullmatch(run_dir.name)
        if match:
            ids.add(match.group(1))
    return sorted(ids, key=int)


def main() -> None:
    runs_root = Path("runs/generalization_benchmark")
    output_root = Path("tmp/plots")
    output_root.mkdir(parents=True, exist_ok=True)

    axis_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    
    with Progress() as progress:
        axes_task = progress.add_task("Processing axes", total=len(axis_dirs))
        
        for axis_dir in axis_dirs:
            config_ids = gather_config_ids(axis_dir)
            configs_task = progress.add_task(f"Processing {axis_dir.name} configs", total=len(config_ids))
            
            for config_id in config_ids:
                plot_metrics_for_config(
                    axis_dir=axis_dir,
                    config_id=config_id,
                    base_metrics=IMPORTANT_BASE_METRICS,
                    output_dir=output_root / axis_dir.name,
                )
                progress.advance(configs_task)
            
            progress.remove_task(configs_task)
            progress.advance(axes_task)

    print(f"Plots saved under {output_root}")


if __name__ == "__main__":
    main()
