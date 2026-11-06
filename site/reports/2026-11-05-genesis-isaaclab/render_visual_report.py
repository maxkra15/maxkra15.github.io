# Copyright (c) 2026 Maximilian Krause
# SPDX-License-Identifier: MIT

"""Render the rollout, startup and learning comparison, including its 8k extension.

Run with ``uv run --no-project --with matplotlib python render_visual_report.py``.
The original technical index is retained as original-report.html. Measured records,
source archives, original figures and checkpoints are never rewritten. The current
appendix and presentation omit the historical pure stepping measurements.
"""

import argparse
import csv
import hashlib
import html
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


def _validate_extension_runs(directory: Path, extension: dict, original: list[dict]) -> None:
    """Bind the 18 new execution records to their raw bytes and original task settings."""
    expected = {(task, backend, seed) for task in TASKS for backend in BACKENDS for seed in (0, 1, 2)}
    actual = [(run["config"]["task"], run["config"]["backend"], run["config"]["seed"]) for run in extension["runs"]]
    if len(actual) != 18 or set(actual) != expected:
        raise ValueError("The 8k extension requires exactly three seeds per task and backend")
    for run in extension["runs"]:
        config, evidence = run["config"], run["provenance"]
        if (config["mode"], config["vision"], config["num_envs"]) != ("rollout", "none", 8192):
            raise ValueError("The 8k extension must contain state-only rollouts")
        if run["status"] != "ok" or run["returncode"] != 0:
            raise ValueError("Unsuccessful extension execution")
        path = Path(run["download"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe extension record path")
        if digest(directory / path) != run["sha256"]:
            raise ValueError(f"Extension raw-record hash differs: {path}")
        raw = json.loads((directory / path).read_text())
        if raw != {key: value for key, value in run.items() if key not in {"download", "sha256"}}:
            raise ValueError(f"Extension record differs from its raw evidence: {path}")
        reference = next(
            item
            for item in original
            if all(item["config"][key] == config[key] for key in ("task", "backend", "seed", "mode", "vision"))
            and item["config"]["num_envs"] == 4096
        )
        previous_config = reference["config"]
        comparable = (set(config) | set(previous_config)) - {"asset_path", "num_envs"}
        if any(config.get(key) != previous_config.get(key) for key in comparable):
            raise ValueError("8k settings differ from their original 4k workload")
        previous = reference["provenance"]
        for key in ("source_revisions", "harness_sha256", "asset_manifest"):
            if evidence[key] != previous[key]:
                raise ValueError(f"Extension source or physical inputs differ: {key}")
        versions = [
            dict(line.split("==", 1) for line in item["package_versions"].splitlines() if "==" in line)
            for item in (previous, evidence)
        ]
        for package in ("torch", "warp-lang", "mujoco", "mujoco-warp", "newton", "genesis-world"):
            if versions[0].get(package) != versions[1].get(package):
                raise ValueError(f"Extension dependency version changed: {package}")
        for key in (
            "cuda_device_uuid",
            "cuda_device",
            "torch",
            "torch_cuda",
            "torch_matmul_precision",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "kit_modules_loaded",
        ):
            if evidence["runtime"][key] != previous["runtime"][key]:
                raise ValueError(f"Extension hardware/runtime differs: {key}")
        if evidence["hardware"]["gpu"] != previous["hardware"]["gpu"]:
            raise ValueError("GPU model, driver, inventory or power limit changed")
        if evidence["other_active_compute_detected"] or not evidence["gpu_activity_samples"]:
            raise ValueError("Extension lacks clean selected-GPU activity evidence")
        if any(sample["external_compute"] for sample in evidence["gpu_activity_samples"]):
            raise ValueError("An extension execution overlapped external GPU compute")
        if config["backend"] == "isaaclab":
            diagnostics = run["backend_diagnostics"]
            if not diagnostics["graph_active"] or any(diagnostics["fatal_capacity_flags_at_end"]):
                raise ValueError("MJWarp graph or final capacity check failed")


def _validate_extension_audits(directory: Path, extension: dict) -> None:
    """Bind six high-index validation records to measured configurations and sources."""
    validations = extension.get("validations", [])
    if {(item["task"], item["backend"]) for item in validations} != {
        (task, backend) for task in TASKS for backend in BACKENDS
    } or len(validations) != 6:
        raise ValueError("The extension requires six end-of-batch validation records")
    audit_source = extension["audit_script"]
    audit_path = Path(audit_source["path"])
    if audit_path.is_absolute() or ".." in audit_path.parts:
        raise ValueError("Unsafe audit source path")
    if digest(directory / audit_path) != audit_source["sha256"]:
        raise ValueError("Batch audit source hash differs")
    required_checks = {
        "last_environment_subset_reset_preserves_all_other_state_bits",
        "last_environment_receives_requested_reset",
        "last_environment_control_preserves_other_worlds",
        "last_environment_control_reaches_last_actuated_coordinate",
    }
    for item in validations:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe extension validation path")
        if digest(directory / path) != item["sha256"]:
            raise ValueError("Extension validation artifact hash differs")
        validation = json.loads((directory / path).read_text())
        config = validation["config"]
        if (validation["kind"], validation["status"], config["num_envs"], config["mode"], config["vision"]) != (
            "batch_scale_validation",
            "ok",
            8192,
            "rollout",
            "none",
        ):
            raise ValueError("The extension has unsuccessful batch validation")
        if (config["task"], config["backend"]) != (item["task"], item["backend"]):
            raise ValueError("Declared batch validation belongs to another task/backend")
        reference = next(
            run
            for run in extension["runs"]
            if all(run["config"][key] == config[key] for key in ("task", "backend", "seed"))
        )
        if {key: value for key, value in config.items() if key != "provenance"} != reference["config"]:
            raise ValueError("Batch validation used a different measured configuration")
        evidence, previous = validation["provenance"], reference["provenance"]
        if validation["audit_sha256"] != audit_source["sha256"]:
            raise ValueError("Validation used a different audit script")
        if (
            evidence["harness_sha256"] != previous["harness_sha256"]
            or evidence["source_revisions"] != previous["source_revisions"]
        ):
            raise ValueError("Validation used different runtime sources")
        if evidence["cuda_device_uuid"] != previous["runtime"]["cuda_device_uuid"]:
            raise ValueError("Validation used a different physical GPU")
        expected_packages = {"genesis"} if config["backend"] == "genesis" else {"isaaclab", "newton"}
        if not expected_packages.issubset(evidence["loaded_packages"]):
            raise ValueError("Validation lacks required loaded source packages")
        if any(
            package["git_head"] != evidence["source_revisions"][name]
            for name, package in evidence["loaded_packages"].items()
        ):
            raise ValueError("Loaded audit packages differ from declared upstream pins")
        if not required_checks.issubset(validation["checks"]) or not all(
            check["passed"] is True for check in validation["checks"].values()
        ):
            raise ValueError("A required batch-index validation check did not pass")
        if config["backend"] == "isaaclab":
            diagnostics = validation["backend_diagnostics_after_index_audit"]
            if not diagnostics["graph_active"] or any(diagnostics["fatal_capacity_flags_at_end"]):
                raise ValueError("Post-audit native graph or final capacity check failed")


def load_runs(directory: Path) -> tuple[list[dict], dict]:
    """Load complete matched rollout/learning evidence and verify the 8k extension."""
    source = directory / "data/measurements.json"
    original = json.loads(source.read_text())["runs"]
    provenance = json.loads((directory / "data/provenance.json").read_text())
    if digest(source) != provenance["derivatives_sha256"]["data/measurements.json"]:
        raise ValueError("Measurements differ from the original published evidence")
    if len(original) != 126 or any(run["status"] != "ok" for run in original):
        raise ValueError("The original measurement archive is incomplete")
    archive = directory / "original-report.html"
    if digest(archive) != provenance["derivatives_sha256"]["index.html"]:
        raise ValueError("The archived original presentation differs from its provenance")

    extension_path = directory / "data/measurements-8192.json"
    extension = json.loads(extension_path.read_text())
    if extension["original_measurements_sha256"] != digest(source):
        raise ValueError("The extension belongs to a different original study")
    _validate_extension_runs(directory, extension, original)
    _validate_extension_audits(directory, extension)
    selected = [run for run in original if run["config"]["mode"] != "physics"] + extension["runs"]
    if len(selected) != 108:
        raise ValueError("Expected 90 original rollout/learning cases plus 18 new rollouts")
    return selected, extension


def presentation_data(directory: Path) -> dict:
    """Derive display values directly from the immutable measurement records."""
    source = directory / "data/measurements.json"
    original_provenance = json.loads((directory / "data/provenance.json").read_text())
    original_index_sha256 = original_provenance["derivatives_sha256"]["index.html"]
    runs, extension = load_runs(directory)
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
            for metric in ("startup_s", "env_steps_per_s", "training_s")
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
            "extension_measured_date": "2026-11-06",
            "presentation_date": "2026-11-06",
            "archived_original_runs": 126,
            "omitted_physics_runs": 36,
            "extension_runs": 18,
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
            "extension_measurement_sha256": digest(directory / "data/measurements-8192.json"),
            "extension_execution_note": extension["execution_note"],
            "original_presentation": {
                "original_path": "index.html",
                "archived_path": "original-report.html",
                "sha256": original_index_sha256,
            },
            "source_files_sha256": {
                name: digest(directory / name)
                for name in (
                    "render_visual_report.py",
                    "plot_report.py",
                    "visual-template.html",
                    "appendix-template.html",
                    "visual.css",
                    "visual.js",
                )
            },
        },
    }


def write_report_data(directory: Path, data: dict, runs: list[dict]) -> None:
    """Export only the rollout and learning records shown in the current report."""
    current = {
        "schema_version": 1,
        "scope": data["scope"],
        "original_measurements_sha256": data["presentation"]["measurement_sha256"],
        "extension_measurements_sha256": data["presentation"]["extension_measurement_sha256"],
        "runs": runs,
    }
    (directory / "data/report-measurements.json").write_text(json.dumps(current, indent=2, allow_nan=False) + "\n")
    columns = (
        "task",
        "backend",
        "mode",
        "vision",
        "num_envs",
        "seed",
        "startup_s",
        "rollout_transitions_per_s",
        "training_s",
        "training_transitions_per_s",
        "first_target_s",
        "target_reached",
        "final_target_reached",
        "threshold_censored",
        "final_error",
        "error_unit",
        "survival_fraction",
        "process_wall_s",
        "cuda_device_uuid",
        "runtime_harness_sha256",
        "raw_sha256",
        "raw_download",
    )
    with (directory / "data/report-measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            config, metrics = run["config"], run["metrics"]
            evaluation = metrics.get("evaluation", {})
            first_target = metrics.get("time_to_threshold")
            writer.writerow(
                {
                    **{key: config[key] for key in columns[:6]},
                    "startup_s": metrics["startup_s"],
                    "rollout_transitions_per_s": metrics["env_steps_per_s"] if config["mode"] == "rollout" else None,
                    "training_s": metrics.get("training_s"),
                    "training_transitions_per_s": metrics["env_steps_per_s"] if config["mode"] == "learning" else None,
                    "first_target_s": first_target["elapsed_s"] if first_target else None,
                    "target_reached": first_target is not None if config["mode"] == "learning" else None,
                    "final_target_reached": evaluation.get("threshold_reached"),
                    "threshold_censored": metrics.get("threshold_censored"),
                    "final_error": evaluation.get(ERROR_KEYS[config["task"]]),
                    "error_unit": {"anymal": "m/s", "cartpole": "rad", "franka": "m"}[config["task"]],
                    "survival_fraction": evaluation.get("survival_fraction"),
                    "process_wall_s": run["process_wall_s"],
                    "cuda_device_uuid": run["provenance"]["runtime"]["cuda_device_uuid"],
                    "runtime_harness_sha256": run["provenance"]["harness_sha256"],
                    "raw_sha256": run["sha256"],
                    "raw_download": run["download"],
                }
            )


def write_appendix(directory: Path, data: dict, runs: list[dict], extension: dict) -> None:
    """Write the current methods and summaries without archived stepping cases."""

    def escape(value: object) -> str:
        return html.escape(str(value))

    def cell(stats: dict | None) -> str:
        if stats is None:
            return "—"
        return f"{stats['median']:.4g} <small>[{stats['min']:.4g}, {stats['max']:.4g}]</small>"

    rows = []
    for group in data["groups"]:
        metrics = group["metrics"]
        learning = group.get("learning")
        first_target = cell(learning["first_target_s"]) if learning else "—"
        reached = f"{learning['reached']}/3; {learning['censored']} censored" if learning else "—"
        values = (
            LABELS[group["task"]],
            group["mode"],
            "State + RGB" if group["vision"] == "rgb" else "State",
            f"{group['num_envs']:,}",
            "Genesis" if group["backend"] == "genesis" else "Isaac Lab MJWarp",
        )
        rows.append(
            "<tr>"
            + "".join(f"<td>{escape(value)}</td>" for value in values)
            + f"<td>{cell(metrics['startup_s'])}</td>"
            + f"<td>{cell(metrics.get('env_steps_per_s')) if group['mode'] == 'rollout' else '—'}</td>"
            + f"<td>{cell(metrics.get('training_s'))}</td><td>{reached}</td><td>{first_target}</td></tr>"
        )
    raw_links = "".join(
        f'<li><a href="{escape(run["download"])}">{escape(run["config"]["task"])} · '
        f"{escape(run['config']['backend'])} · {escape(run['config']['mode'])} · "
        f"{escape(run['config']['vision'])} · N={run['config']['num_envs']} · "
        f"seed {run['config']['seed']}</a></li>"
        for run in runs
    )
    revisions = "".join(
        f"<tr><td>{escape(name)}</td><td><code>{escape(revision)}</code></td></tr>"
        for name, revision in data["source_revisions"].items()
    )
    validators = "".join(
        f'<li><a href="{escape(item["path"])}">{escape(item.get("label", item["path"]))}</a></li>'
        for item in extension.get("validations", [])
    )
    page = (directory / "appendix-template.html").read_text()
    slots = {
        "__RUN_COUNT__": str(len(runs)),
        "__REVISIONS__": revisions,
        "__EXECUTION_NOTE__": escape(extension["execution_note"]),
        "__ROWS__": "".join(rows),
        "__VALIDATION_LINKS__": validators,
        "__RAW_LINKS__": raw_links,
    }
    for slot, value in slots.items():
        if page.count(slot) != 1:
            raise ValueError(f"Appendix template slot missing or repeated: {slot}")
        page = page.replace(slot, value)
    (directory / "appendix.html").write_text(page)


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
        ("8k state rollouts", "rollout", "none", 8192, "env_steps_per_s", 1e6, "M transitions/s ↑"),
        ("RGB rollouts", "rollout", "rgb", 64, "env_steps_per_s", 1e3, "k transitions/s ↑"),
        ("8k state startup", "rollout", "none", 8192, "startup_s", 1, "seconds ↓"),
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
        "Three-seed medians; whiskers show min–max. State rollout/startup: 8,192 environments. "
        "RGB rollout: 64 environments.\n"
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
    archive = directory / "original-report.html"
    if not archive.exists():
        shutil.copyfile(directory / "appendix.html", archive)
    data = presentation_data(directory)
    runs, extension = load_runs(directory)
    write_report_data(directory, data, runs)
    write_appendix(directory, data, runs, extension)
    export_overview(data, directory)
    from plot_report import export_plots

    data["plots"] = export_plots(directory, runs, data["groups"])
    data["presentation"]["plot_data_sha256"] = digest(directory / "data/plots.json")
    encoded = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    (directory / "data/presentation.json").write_text(encoded)
    template = (directory / "visual-template.html").read_text()
    if template.count("__REPORT_DATA__") != 1:
        raise ValueError("Visual template must contain exactly one report data slot")
    page = template.replace("__REPORT_DATA__", encoded.replace("<", "\\u003c"))
    (directory / "index.html").write_text(page)
    readme = (
        "Visual presentation source (MIT)\n\n"
        "Extract these files into the published report directory, alongside data/ and figures/.\n"
        "Run: uv run --no-project --with matplotlib python render_visual_report.py\n\n"
        "The renderer reads data/measurements.json plus data/measurements-8192.json and their raw records.\n"
        "It preserves historical evidence, selects rollout/learning cases, and builds the current appendix.\n"
        "plot_report.py generates scaling, paired RGB-cost and per-seed learning figures.\n"
        "Original data/provenance.json refers to original-report.html; data/presentation.json\n"
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
