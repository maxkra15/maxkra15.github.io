# Copyright (c) 2026 Maximilian Krause
# SPDX-License-Identifier: MIT

"""Render the visual presentation of the unchanged, published simulator study.

Run with ``uv run --no-project --with matplotlib python render_visual_report.py``.
The original technical report is retained as appendix.html. Measured records,
source archives, original figures and checkpoints are never rewritten.
"""

import argparse
import hashlib
import json
import shutil
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

TASKS = ("anymal", "cartpole", "franka")
BACKENDS = ("genesis", "isaaclab")
LABELS = {"anymal": "ANYmal", "cartpole": "Cartpole", "franka": "Franka"}
COLORS = {"genesis": "#245cab", "isaaclab": "#b04b17"}
ERROR_KEYS = {
    "anymal": "tracking_error_m_s",
    "cartpole": "pole_angle_abs_rad",
    "franka": "reaching_error_m",
}


def digest(path: Path) -> str:
    """Return the SHA-256 of a published file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[tuple[int, float]]) -> dict:
    """Summarize three seed measurements without combining workloads."""
    numbers = [value for _, value in values]
    return {
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "n": len(numbers),
        "values": [{"seed": seed, "value": value} for seed, value in values],
    }


def presentation_data(directory: Path) -> dict:
    """Derive display values directly from the immutable measurement records."""
    source = directory / "data/measurements.json"
    publication = json.loads(source.read_text())
    original_provenance = json.loads((directory / "data/provenance.json").read_text())
    if digest(source) != original_provenance["derivatives_sha256"]["data/measurements.json"]:
        raise ValueError("Measurements differ from the original published evidence")
    original_index_sha256 = original_provenance["derivatives_sha256"]["index.html"]
    original_page = directory / "appendix.html" if (directory / "appendix.html").exists() else directory / "index.html"
    if digest(original_page) != original_index_sha256:
        raise ValueError("The archived technical report differs from the original provenance")
    runs = publication["runs"]
    if len(runs) != 126 or any(run["status"] != "ok" for run in runs):
        raise ValueError("This presentation requires the complete 126-run published study")
    grouped = defaultdict(list)
    for run in runs:
        config = run["config"]
        key = tuple(config[name] for name in ("task", "backend", "mode", "vision", "num_envs"))
        grouped[key].append(run)
    groups = []
    for key, records in sorted(grouped.items()):
        records.sort(key=lambda run: run["config"]["seed"])
        if [run["config"]["seed"] for run in records] != [0, 1, 2]:
            raise ValueError(f"Incomplete or duplicated seed group: {key}")
        group = dict(zip(("task", "backend", "mode", "vision", "num_envs"), key))
        group["metrics"] = {
            metric: summarize([(run["config"]["seed"], run["metrics"][metric]) for run in records])
            for metric in ("startup_s", "physics_steps_per_s", "env_steps_per_s", "training_s")
            if all(metric in run["metrics"] for run in records)
        }
        if group["mode"] == "learning":
            seeds = []
            for run in records:
                metrics = run["metrics"]
                first_target = metrics["time_to_threshold"]
                reached = first_target is not None
                if metrics["threshold_censored"] == reached:
                    raise ValueError("Censoring and first-target evidence disagree")
                seeds.append(
                    {
                        "seed": run["config"]["seed"],
                        "reached": reached,
                        "censored": metrics["threshold_censored"],
                        "first_target_s": first_target["elapsed_s"] if reached else None,
                        "final_error": metrics["evaluation"][ERROR_KEYS[group["task"]]],
                        "survival": metrics["evaluation"]["survival_fraction"],
                        "final_reached": metrics["evaluation"]["threshold_reached"],
                        "raw": run["download"],
                    }
                )
            hits = [(seed["seed"], seed["first_target_s"]) for seed in seeds if seed["reached"]]
            group["learning"] = {
                "reached": len(hits),
                "final_reached": sum(seed["final_reached"] for seed in seeds),
                "censored": len(seeds) - len(hits),
                "first_target_s": summarize(hits) if hits else None,
                "seeds": seeds,
            }
        groups.append(group)

    ratios = []
    for group in groups:
        if group["mode"] == "rollout" and group["backend"] == "genesis":
            peer = next(
                candidate
                for candidate in groups
                if candidate["backend"] == "isaaclab"
                and all(candidate[name] == group[name] for name in ("task", "mode", "vision", "num_envs"))
            )
            ratios.append(peer["metrics"]["env_steps_per_s"]["median"] / group["metrics"]["env_steps_per_s"]["median"])

    frames = []
    for task in TASKS:
        for backend in BACKENDS:
            run = next(
                run
                for run in runs
                if (
                    run["config"]["task"],
                    run["config"]["backend"],
                    run["config"]["mode"],
                    run["config"]["vision"],
                    run["config"]["seed"],
                )
                == (task, backend, "rollout", "rgb", 0)
            )
            frame = run["download"].replace("-run.json", "-camera.png")
            if not (directory / frame).is_file():
                raise ValueError(f"Missing original camera sample: {frame}")
            frames.append({"task": task, "backend": backend, "path": frame})
    learning_figures = []
    for task in TASKS:
        for vision in ("none", "rgb"):
            matches = list((directory / "figures").glob(f"learning-{task}-{vision}-512-*.svg"))
            if len(matches) != 1:
                raise ValueError(f"Expected one original learning figure: {task}/{vision}")
            learning_figures.append({"task": task, "vision": vision, "path": str(matches[0].relative_to(directory))})
    return {
        "scope": {
            "runs": len(runs),
            "seeds": [0, 1, 2],
            "hardware": "NVIDIA L40",
            "measured_date": "2026-11-05",
            "presentation_date": "2026-11-06",
            "iterations": 500,
            "training_transitions": 6144000,
            "learning_num_envs": 512,
        },
        "groups": groups,
        "camera_samples": frames,
        "learning_figures": learning_figures,
        "source_revisions": runs[0]["provenance"]["source_revisions"],
        "takeaway": {
            "rollout_ratio_min": min(ratios),
            "rollout_ratio_max": max(ratios),
            "rollout_settings": len(ratios),
            "rollout_mjwarp_higher": sum(ratio > 1 for ratio in ratios),
        },
        "presentation": {
            "schema_version": 1,
            "measurement_sha256": digest(source),
            "original_presentation": {
                "original_path": "index.html",
                "archived_path": "appendix.html",
                "sha256": original_index_sha256,
            },
            "source_files_sha256": {
                name: digest(directory / name)
                for name in ("render_visual_report.py", "visual-template.html", "visual.css", "visual.js")
            },
        },
    }


def export_overview(data: dict, directory: Path) -> None:
    """Export standalone SVG, PNG and PDF figures from the same display values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "genesis-isaaclab-visual-v1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#cbd5e1",
            "text.color": "#192333",
            "axes.labelcolor": "#536174",
            "xtick.color": "#536174",
            "ytick.color": "#536174",
        }
    )
    columns = (
        ("Pure physics", "physics", "none", 4096, "physics_steps_per_s", 1e6, "M steps/s ↑"),
        ("RGB rollouts", "rollout", "rgb", 64, "env_steps_per_s", 1e3, "k transitions/s ↑"),
        ("RGB startup", "rollout", "rgb", 64, "startup_s", 1, "seconds ↓"),
        ("RGB PPO budget", "learning", "rgb", 512, "training_s", 1, "seconds ↓"),
    )
    figure, axes = plt.subplots(3, 4, figsize=(13.4, 6.9), layout="constrained")
    figure.set_constrained_layout_pads(w_pad=0.08, h_pad=0.12, wspace=0.1, hspace=0.25)
    for column, (title, mode, vision, count, metric, scale, unit) in enumerate(columns):
        selected = [
            group
            for group in data["groups"]
            if (group["mode"], group["vision"], group["num_envs"]) == (mode, vision, count)
        ]
        maximum = max(group["metrics"][metric]["max"] / scale for group in selected)
        for row, task in enumerate(TASKS):
            axis = axes[row, column]
            axis.set_axisbelow(True)
            axis.grid(axis="x", color="#e8edf2", linewidth=0.7)
            axis.set_xlim(0, maximum * (1.50 if mode == "learning" else 1.32))
            axis.set_ylim(-0.6, 1.6)
            axis.set_yticks([])
            axis.tick_params(axis="x", labelsize=9, length=0)
            axis.locator_params(axis="x", nbins=3)
            for y, backend in zip((1, 0), BACKENDS):
                group = next(group for group in selected if group["task"] == task and group["backend"] == backend)
                stats = group["metrics"][metric]
                median, low, high = (stats[key] / scale for key in ("median", "min", "max"))
                axis.barh(y, median, height=0.46, color=COLORS[backend], alpha=0.85, zorder=2)
                axis.hlines(y, low, high, color="#192333", linewidth=1.3, zorder=3)
                axis.vlines((low, high), y - 0.1, y + 0.1, color="#192333", linewidth=1.1, zorder=3)
                axis.scatter(
                    [point["value"] / scale for point in stats["values"]],
                    [y] * 3,
                    s=12,
                    facecolors="white",
                    edgecolors=COLORS[backend],
                    linewidths=0.8,
                    zorder=4,
                )
                value = f"{median:.2f}" if scale > 1 else f"{median:.0f}"
                if mode == "learning":
                    value += f" · {group['learning']['reached']}/3 reached"
                axis.annotate(
                    value,
                    (high, y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    annotation_clip=False,
                )
            if column == 0:
                axis.text(-0.03, 1.27, LABELS[task], transform=axis.transAxes, fontsize=13, weight="bold")
            if row == 0:
                axis.set_title(f"{title}\n{unit}", loc="left", pad=32, fontsize=12, weight="bold")
    figure.legend(
        handles=[
            Patch(facecolor=COLORS[backend], label=label)
            for backend, label in zip(BACKENDS, ("Genesis", "Isaac Lab MJWarp"))
        ],
        loc="outside upper right",
        ncols=2,
        frameon=False,
    )
    figure.suptitle("Genesis vs Isaac Lab MJWarp", x=0.03, ha="left", fontsize=20, weight="bold")
    figure.supxlabel(
        "Three-seed medians; whiskers show min–max. Physics: 4,096 environments, no RGB. "
        "RGB rollout/startup: 64 environments.\n"
        "PPO: 512 environments, 500 iterations, including missed targets and evaluation. "
        "Native solvers and renderers differ.",
        x=0.03,
        ha="left",
        fontsize=9,
        color="#536174",
    )
    target = directory / "visuals"
    target.mkdir(exist_ok=True)
    for extension in ("svg", "png", "pdf"):
        metadata = {"Date": None} if extension == "svg" else None
        if extension == "pdf":
            metadata = {"CreationDate": None, "ModDate": None}
        figure.savefig(target / f"overview.{extension}", dpi=180, metadata=metadata, facecolor="white")
    plt.close(figure)


def render(directory: Path) -> None:
    """Write the presentation and checksums, preserving the original evidence."""
    data = presentation_data(directory)
    appendix = directory / "appendix.html"
    if not appendix.exists():
        shutil.copyfile(directory / "index.html", appendix)
    encoded = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    (directory / "data/presentation.json").write_text(encoded)
    template = (directory / "visual-template.html").read_text()
    if template.count("__REPORT_DATA__") != 1:
        raise ValueError("Visual template must contain exactly one report data slot")
    page = template.replace("__REPORT_DATA__", encoded.replace("<", "\\u003c"))
    (directory / "index.html").write_text(page)
    export_overview(data, directory)
    readme = (
        "Visual presentation source (MIT)\n\n"
        "Extract these files into the published report directory, alongside data/ and figures/.\n"
        "Run: uv run --no-project --with matplotlib python render_visual_report.py\n\n"
        "The renderer uses data/measurements.json and preserves all measured evidence.\n"
        "The first run archives the original index.html as appendix.html. Further runs keep it.\n"
        "Original data/provenance.json refers to that archived presentation; data/presentation.json\n"
        "records the new renderer and input hashes. SHA256SUMS covers the current published files.\n"
        "The three original measured/publisher source ZIPs are separate, unchanged snapshots.\n"
    )
    with zipfile.ZipFile(directory / "presentation-source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        files = {name: (directory / name).read_bytes() for name in data["presentation"]["source_files_sha256"]}
        files["README.txt"] = readme.encode()
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    checksums = "".join(
        f"{digest(path)}  {path.relative_to(directory)}\n"
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS" and "__pycache__" not in path.parts
    )
    (directory / "SHA256SUMS").write_text(checksums)
    print(
        json.dumps(
            {
                "page": str(directory / "index.html"),
                "runs": data["scope"]["runs"],
                "groups": len(data["groups"]),
                "checksums": len(checksums.splitlines()),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_dir", type=Path, default=Path(__file__).resolve().parent)
    render(parser.parse_args().report_dir.resolve())
