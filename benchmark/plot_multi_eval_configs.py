#!/usr/bin/env python3
"""Plot multi-eval TensorBoard metrics per train config from an aggregated CSV."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def _set_icml_rcparams() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.linewidth": 1.44,
            "xtick.major.width": 1.26,
            "ytick.major.width": 1.26,
            "xtick.minor.width": 0.9,
            "ytick.minor.width": 0.9,
            "lines.linewidth": 3.42,
            "lines.markersize": 10.8,
            "legend.frameon": False,
        }
    )


_STYLE_SET = False


def _ensure_icml_style() -> None:
    global _STYLE_SET
    if not _STYLE_SET:
        _set_icml_rcparams()
        _STYLE_SET = True


BASE_METRICS: Dict[str, Sequence[str]] = {
    "passed_distance": ("avg_config_passed_distance",),
    "progress": ("avg_config_progress",),
    "success_rate": ("avg_config_success_once", "avg_config_success"),
    "return": ("avg_episodic_return",),
}

HUMAN_YLABELS: Dict[str, str] = {
    "passed_distance": "Passed Distance",
    "progress": "Progress",
    "success_rate": "Success Rate",
    "return": "Return",
}

Y_AXIS_CONFIG: Dict[str, Tuple[Tuple[float, float], Sequence[float]]] = {
    "passed_distance": ((0.0, 530.0), (0, 100, 200, 300, 400, 500)),
    "progress": ((0.0, 1.1), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
    "success_rate": ((0.0, 1.1), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
    "return": ((-4000.0, 200.0), (-4000, -3000, -2000, -1000, 0)),
}

MAX_STEP = 10_000_000
STEP_AXIS_LIMITS = (0.0, float(MAX_STEP))
STEP_TICKS = tuple(int(MAX_STEP * frac) for frac in (0.0, 0.25, 0.5, 0.75, 1.0))
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 20
OFFSET_TEXT_SIZE = 20
LEGEND_X_ANCHOR = 0.82
LEGEND_RIGHT = 0.76


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot multi-eval metrics from get_multi_eval_metrics.py output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tmp/multi_eval_metrics_summary.csv"),
        help="Aggregated CSV from get_multi_eval_metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/plots/multi_eval"),
        help="Directory to write per-config plots.",
    )
    return parser.parse_args()


def _strip_yaml(label: str) -> str:
    return label[:-5] if label.endswith(".yaml") else label


def eval_sort_key(label: str, train_label: str) -> Tuple[int, float, str]:
    name = _strip_yaml(label)
    base = train_label[:-len("_train")] if train_label.endswith("_train") else train_label
    if name == train_label or name == f"{base}_train":
        return (0, 0.0, name)
    match = re.search(r"val_(-?[0-9]+(?:\.[0-9]+)?)$", name)
    if match:
        return (1, float(match.group(1)), name)
    return (2, 0.0, name)


def display_label(train_label: str, eval_label: str) -> str:
    name = _strip_yaml(eval_label)
    base = train_label[:-len("_train")] if train_label.endswith("_train") else train_label
    if name == train_label or name == f"{base}_train":
        return "train"
    prefix = f"{base}_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.startswith("val_"):
        color_abbrev = {
            "black": "b",
            "white": "w",
            "red": "r",
            "green": "g",
            "blue": "b",
        }
        tokens = name.split("_")[1:]
        if tokens and all(token in color_abbrev for token in tokens):
            return "val_" + "".join(color_abbrev[token] for token in tokens)
    return name


def pick_metric_key(
    metrics: Dict[str, Dict[str, Dict[int, Tuple[float, float]]]],
    candidates: Iterable[str],
) -> str:
    for candidate in candidates:
        if candidate in metrics:
            return candidate
    return ""


def _apply_axis_bounds(ax: mpl.axes.Axes, base_metric: str) -> None:
    ax.set_xlim(STEP_AXIS_LIMITS)
    ax.set_xticks(STEP_TICKS)
    xfmt = ScalarFormatter(useMathText=False)
    xfmt.set_scientific(True)
    xfmt.set_powerlimits((7, 7))
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis.get_offset_text().set_size(OFFSET_TEXT_SIZE)

    config = Y_AXIS_CONFIG.get(base_metric)
    if config:
        ylim, yticks = config
        ax.set_ylim(ylim)
        ax.set_yticks(list(yticks))


def load_csv(
    csv_path: Path,
) -> DefaultDict[str, Dict[str, Dict[str, Dict[int, Tuple[float, float]]]]]:
    data: DefaultDict[str, Dict[str, Dict[str, Dict[int, Tuple[float, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_label = row["train_label"]
            metric = row["metric"]
            eval_label = row["eval_label"]
            step = int(row["step"])
            mean = float(row["mean"])
            sem = float(row["sem"])
            data[train_label][metric][eval_label][step] = (mean, sem)
    return data


def _plot_line_with_sem(
    ax: mpl.axes.Axes,
    *,
    steps: Sequence[int],
    means: Sequence[float],
    sems: Sequence[float],
    label: str,
    color: str,
) -> mpl.lines.Line2D:
    lower = [m - s for m, s in zip(means, sems)]
    upper = [m + s for m, s in zip(means, sems)]
    (line,) = ax.plot(steps, means, label=label, color=color, zorder=4)
    ax.fill_between(steps, lower, upper, color=color, alpha=0.18, linewidth=0.0, zorder=2)
    return line


def plot_config_metric(
    train_label: str,
    metrics: Dict[str, Dict[str, Dict[int, Tuple[float, float]]]],
    output_dir: Path,
    *,
    base_metric: str,
    figsize: Tuple[float, float] = (8.5, 5.2),
) -> None:
    _ensure_icml_style()

    eval_labels = sorted(
        {label for metric_vals in metrics.values() for label in metric_vals.keys()},
        key=lambda label: eval_sort_key(label, train_label),
    )
    if not eval_labels:
        return

    metric_key = pick_metric_key(metrics, BASE_METRICS[base_metric])
    if not metric_key:
        return
    series = metrics[metric_key]

    fig, ax = plt.subplots(figsize=figsize)
    palette = [
        "#1f77b4",
        "#d62728",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    seen_labels = set()
    legend_handles: List[mpl.lines.Line2D] = []
    legend_labels: List[str] = []
    has_data = False

    for idx, eval_label in enumerate(eval_labels):
        step_map = series.get(eval_label)
        if not step_map:
            continue
        steps = [s for s in sorted(step_map.keys()) if s <= MAX_STEP]
        if not steps:
            continue
        means = [step_map[s][0] for s in steps]
        sems = [step_map[s][1] for s in steps]
        label = display_label(train_label, eval_label)
        color = palette[idx % len(palette)]
        line = _plot_line_with_sem(
            ax,
            steps=steps,
            means=means,
            sems=sems,
            label=label,
            color=color,
        )
        if label not in seen_labels:
            seen_labels.add(label)
            legend_handles.append(line)
            legend_labels.append(label)
        has_data = True

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Step")
    ax.set_ylabel(HUMAN_YLABELS.get(base_metric, base_metric))
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    _apply_axis_bounds(ax, base_metric)
    ax.grid(True, which="major", alpha=0.2, linewidth=0.6)
    ax.grid(False, which="minor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)

    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", train_label)
    safe_metric = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_metric)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{safe_label}_{safe_metric}_multi_eval.pdf"
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            ncol=1,
            loc="center left",
            bbox_to_anchor=(LEGEND_X_ANCHOR-0.1, 0.5),
            handlelength=2.8,
            handletextpad=0.5,
            borderpad=0.3,
            fontsize=20,
            # frameon=True,
            framealpha=0.7,
            facecolor="white",
            edgecolor="0.6",
        )
    fig.tight_layout(rect=(0, 0, LEGEND_RIGHT, 1))
    fig.savefig(out_path, bbox_inches=fig.bbox_inches)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")
    data = load_csv(args.input)
    if not data:
        raise SystemExit("No metrics found to plot.")
    for train_label, metrics in data.items():
        for base_metric in BASE_METRICS.keys():
            plot_config_metric(
                train_label,
                metrics,
                args.output_dir,
                base_metric=base_metric,
            )
    print(f"Plots saved under {args.output_dir}")


if __name__ == "__main__":
    main()
