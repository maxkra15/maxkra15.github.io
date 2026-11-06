# Copyright (c) 2026 Maximilian Krause
# SPDX-License-Identifier: MIT

"""Export scientific figures from the report's existing measured records."""

import hashlib
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TASKS = {
    "anymal": ("ANYmal-C", "tracking_error_m_s", "Velocity error [m/s]", 0.25),
    "cartpole": ("Cartpole", "pole_angle_abs_rad", "Pole angle error [rad]", 0.1),
    "franka": ("Franka", "reaching_error_m", "Tool position error [m]", 0.05),
}
BACKENDS = {"genesis": ("Genesis", "#245cab"), "isaaclab": ("Isaac Lab MJWarp", "#b04b17")}
SEED_STYLES = ("-", "--", ":")
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "svg.fonttype": "none",
    "svg.hashsalt": "genesis-isaaclab-plots-v1",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "text.color": "#192333",
    "xtick.color": "#536174",
    "ytick.color": "#536174",
    "axes.axisbelow": True,
    "grid.color": "#e8edf2",
    "grid.linewidth": 0.7,
}


def _save(figure, directory: Path, stem: str) -> dict[str, str]:
    """Save deterministic SVG, PNG and PDF versions of one figure."""
    paths = {}
    for extension in ("svg", "png", "pdf"):
        relative = f"visuals/{stem}.{extension}"
        metadata = {"Date": None} if extension == "svg" else None
        if extension == "pdf":
            metadata = {"CreationDate": None, "ModDate": None}
        figure.savefig(directory / relative, dpi=180, metadata=metadata, facecolor="white")
        paths[extension] = relative
    plt.close(figure)
    return paths


def _plot_scaling(directory: Path, groups: list[dict], task: str) -> tuple[dict, list[dict]]:
    """Plot measured state-rollout throughput with median and observed seed range."""
    figure, axis = plt.subplots(figsize=(5.7, 4.0), layout="constrained")
    series = []
    for backend, (label, color) in BACKENDS.items():
        points = []
        for count in (256, 4096, 8192):
            group = next(
                group
                for group in groups
                if (group["task"], group["backend"], group["mode"], group["vision"], group["num_envs"])
                == (task, backend, "rollout", "none", count)
            )
            points.append({"num_envs": count, **group["metrics"]["env_steps_per_s"]})
        x = [point["num_envs"] for point in points]
        y = [point["median"] / 1e6 for point in points]
        error = [
            [(point["median"] - point["min"]) / 1e6 for point in points],
            [(point["max"] - point["median"]) / 1e6 for point in points],
        ]
        axis.errorbar(x, y, yerr=error, marker="o", markersize=5, linewidth=1.8, capsize=4, color=color, label=label)
        axis.annotate(
            f"{y[-1]:.2f}M",
            (x[-1], y[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=color,
        )
        series.append({"task": task, "backend": backend, "points": points})
    axis.set(
        xlim=(0, 9900),
        ylim=(0, None),
        xticks=(256, 4096, 8192),
        xticklabels=("256", "4,096", "8,192"),
        xlabel="Parallel environments · state only",
        ylabel="Rollout throughput [M transitions/s]",
    )
    axis.grid()
    axis.legend(loc="upper left", frameon=False, fontsize=9)
    figure.suptitle(f"{TASKS[task][0]} · rollout scaling", fontsize=14, weight="bold")
    figure.supxlabel(
        "Three-seed medians; whiskers show observed min–max.\nBatches collected separately on the same per-task GPU.",
        fontsize=8.5,
        color="#536174",
    )
    return {"task": task, **_save(figure, directory, f"scaling-{task}")}, series


def _learning_series(runs: list[dict], task: str, vision: str) -> list[dict]:
    """Retain every seed and its actual evaluation sample locations."""
    series = []
    for run in runs:
        config, metrics = run["config"], run["metrics"]
        if (config["task"], config["mode"], config["vision"]) != (task, "learning", vision):
            continue
        samples = [
            {
                "iteration": point["iteration"],
                "env_steps": point["env_steps"],
                "elapsed_s": point["elapsed_s"],
                "error": point["evaluation"][TASKS[task][1]],
                "survival": point["evaluation"]["survival_fraction"],
                "target_reached": point["evaluation"]["threshold_reached"],
            }
            for point in metrics["learning"]
        ]
        series.append(
            {
                "task": task,
                "vision": vision,
                "backend": config["backend"],
                "seed": config["seed"],
                "reached": metrics["time_to_threshold"] is not None,
                "censored": metrics["threshold_censored"],
                "first_target": metrics["time_to_threshold"],
                "raw": run["download"],
                "samples": samples,
            }
        )
    return sorted(series, key=lambda item: (item["backend"], item["seed"]))


def _plot_learning(directory: Path, series: list[dict], task: str, vision: str, clock: str) -> dict:
    """Plot individual held-out traces without averaging unequal time locations."""
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(5.7, 5.6), layout="constrained")
    handles = []
    for item in series:
        label, color = BACKENDS[item["backend"]]
        points = item["samples"]
        x = [point["env_steps"] / 1e6 if clock == "samples" else point["elapsed_s"] for point in points]
        values = ([point["error"] for point in points], [point["survival"] for point in points])
        hit = next((index for index, point in enumerate(points) if point["target_reached"]), None)
        for axis, y in zip(axes, values):
            (line,) = axis.plot(
                x,
                y,
                linestyle=SEED_STYLES[item["seed"]],
                color=color,
                alpha=0.8,
                linewidth=1.6,
                marker="o",
                markersize=2.5,
                label=f"{label} · seed {item['seed']}",
            )
            if hit is not None:
                axis.scatter(
                    [x[hit]], [y[hit]], marker="D", s=35, color=color, edgecolor="white", linewidth=0.7, zorder=5
                )
        handles.append(line)
    target = TASKS[task][3]
    maximum = max(point["error"] for item in series for point in item["samples"])
    axes[0].set(ylim=(0, max(maximum, target) * 1.14), ylabel=TASKS[task][2])
    axes[1].set(ylim=(-0.03, 1.06), ylabel="Survival fraction")
    for axis, threshold, text, offset in (
        (axes[0], target, f"Error target ≤ {target:g}", 5),
        (axes[1], 0.95, "Survival target ≥ 0.95", -14),
    ):
        axis.axhline(threshold, color="#667788", linestyle=(0, (4, 3)), linewidth=1)
        axis.annotate(
            text,
            (0.99, threshold),
            xycoords=("axes fraction", "data"),
            xytext=(0, offset),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color="#536174",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
        )
        axis.grid()
        axis.tick_params(labelsize=9)
    if clock == "samples":
        axes[1].set(xlim=(0, 6.3), xticks=(0, 2, 4, 6), xlabel="Training transitions [million]")
    else:
        axes[1].set_xlim(left=0)
        axes[1].set_xlabel("Training time [s] · includes evaluation")
    observation = "State + RGB" if vision == "rgb" else "State"
    counts = {backend: sum(item["reached"] for item in series if item["backend"] == backend) for backend in BACKENDS}
    figure.suptitle(
        f"{TASKS[task][0]} · {observation}\n"
        f"Target reached: Genesis {counts['genesis']}/3 · MJWarp {counts['isaaclab']}/3",
        fontsize=12.5,
        weight="bold",
    )
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncols=2,
        fontsize=9,
        frameon=False,
        title="Diamonds: first observed joint target; both panels required",
        title_fontsize=8.2,
    )
    return {
        "task": task,
        "vision": vision,
        "axis": clock,
        **_save(figure, directory, f"learning-{task}-{vision}-{clock}"),
    }


def _rgb_ratios(runs: list[dict]) -> list[dict]:
    """Pair RGB/state budget costs by task, backend and seed at 512 environments."""
    result = []
    for task in TASKS:
        for backend in BACKENDS:
            pairs = []
            for seed in (0, 1, 2):
                selected = [
                    next(
                        run
                        for run in runs
                        if (
                            run["config"]["task"],
                            run["config"]["backend"],
                            run["config"]["mode"],
                            run["config"]["vision"],
                            run["config"]["num_envs"],
                            run["config"]["seed"],
                        )
                        == (task, backend, "learning", vision, 512, seed)
                    )
                    for vision in ("none", "rgb")
                ]
                pairs.append(
                    {
                        "seed": seed,
                        "state_s": selected[0]["metrics"]["training_s"],
                        "rgb_s": selected[1]["metrics"]["training_s"],
                        "ratio": selected[1]["metrics"]["training_s"] / selected[0]["metrics"]["training_s"],
                        "state_reached": selected[0]["metrics"]["time_to_threshold"] is not None,
                        "rgb_reached": selected[1]["metrics"]["time_to_threshold"] is not None,
                        "state_raw": selected[0]["download"],
                        "rgb_raw": selected[1]["download"],
                    }
                )
            values = [pair["ratio"] for pair in pairs]
            result.append(
                {
                    "task": task,
                    "backend": backend,
                    "pairs": pairs,
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "state_reached": sum(pair["state_reached"] for pair in pairs),
                    "rgb_reached": sum(pair["rgb_reached"] for pair in pairs),
                }
            )
    return result


def _plot_rgb_cost(directory: Path, series: list[dict]) -> dict:
    """Show seed-paired cost ratios and target counts, including censored runs."""
    figure, (axis, counts) = plt.subplots(
        1, 2, figsize=(5.8, 4.6), sharey=True, gridspec_kw={"width_ratios": (4, 1.25)}, layout="constrained"
    )
    for row, task in enumerate(TASKS):
        center = 5 - row * 2
        for offset, backend in zip((0.19, -0.19), BACKENDS):
            item = next(item for item in series if (item["task"], item["backend"]) == (task, backend))
            color, y = BACKENDS[backend][1], center + offset
            axis.hlines(y, item["min"], item["max"], color=color, linewidth=1.5)
            axis.vlines((item["min"], item["max"]), y - 0.08, y + 0.08, color=color, linewidth=1)
            axis.scatter(
                [pair["ratio"] for pair in item["pairs"]],
                [y] * 3,
                s=18,
                facecolors="white",
                edgecolors=color,
                linewidth=1,
                zorder=3,
            )
            axis.scatter([item["median"]], [y], s=30, marker="D", color=color, zorder=4)
            axis.annotate(
                f"{item['median']:.2f}×",
                (item["max"], y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=color,
                fontsize=9,
            )
            counts.text(
                0.08, y, f"{item['state_reached']}/3 → {item['rgb_reached']}/3", va="center", fontsize=10, color=color
            )
    axis.set(
        xlim=(0.8, 4.25),
        ylim=(0.4, 5.7),
        xticks=(1, 2, 3, 4),
        yticks=(5, 3, 1),
        yticklabels=[item[0] for item in TASKS.values()],
        xlabel="State + RGB / State PPO budget time [×]",
    )
    axis.axvline(1, color="#667788", linestyle=(0, (4, 3)), linewidth=1)
    axis.grid(axis="x")
    counts.set_axis_off()
    counts.set_title("Reached\nState → RGB", fontsize=9.5)
    figure.suptitle("PPO cost of RGB", fontsize=14, weight="bold")
    figure.legend(
        handles=[Line2D([0], [0], color=color, marker="D", label=label) for label, color in BACKENDS.values()],
        loc="outside lower center",
        ncols=2,
        frameon=False,
        fontsize=9,
        title="Median and min–max of three seed-paired ratios.\n"
        "512 environments · 6.144M transitions · includes missed targets.",
        title_fontsize=8.2,
    )
    return _save(figure, directory, "ppo-rgb-cost")


def export_plots(directory: Path, runs: list[dict], groups: list[dict]) -> dict:
    """Export new plot families and auditable numeric inputs, with no new measurements."""
    plt.switch_backend("Agg")
    plots = {"scaling": [], "learning": []}
    snapshot = {"schema_version": 1, "scaling": [], "learning": [], "rgb_cost": _rgb_ratios(runs)}
    with plt.rc_context(STYLE):
        for task in TASKS:
            descriptor, series = _plot_scaling(directory, groups, task)
            plots["scaling"].append(descriptor)
            snapshot["scaling"].extend(series)
            for vision in ("none", "rgb"):
                series = _learning_series(runs, task, vision)
                snapshot["learning"].extend(series)
                for clock in ("samples", "time"):
                    plots["learning"].append(_plot_learning(directory, series, task, vision, clock))
        plots["rgb_cost"] = _plot_rgb_cost(directory, snapshot["rgb_cost"])
    snapshot["inputs"] = {
        "path": "data/report-measurements.json",
        "sha256": hashlib.sha256((directory / "data/report-measurements.json").read_bytes()).hexdigest(),
    }
    snapshot["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (directory / "data/plots.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    return plots
