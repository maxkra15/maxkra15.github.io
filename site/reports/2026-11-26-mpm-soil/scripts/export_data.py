# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Derive the report's compact data files from the newton-calibration MPM soil case.

Run from a newton-calibration checkout that contains ``examples/mpm_soil`` and the run
directories of the two calibrations, for example::

    python export_data.py --repo . --rft-run workspace/mpm-rft/<run> \\
        --rake-run workspace/mpm-rake/<run> --out <report>/data

Measured plate-intrusion slopes of Li, Zhang & Goldman (2013) are not copied into the
outputs; they remain available from the article's supplementary table S5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _match(points, target, tol=1e-3) -> int:
    distance = np.abs(np.asarray(points) - np.asarray(target)).max(axis=1)
    index = int(np.argmin(distance))
    if distance[index] > tol:
        raise ValueError(f"point {target} not found")
    return index


def rft_points(archive: Path, train_points) -> dict:
    literature = _load(archive / "rft_sweep_literature.json")
    calibrated = _load(archive / "rft_sweep_calibrated.json")
    train = np.asarray(train_points)
    points = []
    for (beta, gamma), lit in zip(literature["points"], literature["predicted"], strict=True):
        cal = calibrated["predicted"][_match(calibrated["points"], (beta, gamma))]
        role = "train" if np.isclose(train, (beta, gamma), atol=1e-3).all(axis=1).any() else "validation"
        points.append(
            {
                "beta_rad": round(float(beta), 6),
                "gamma_rad": round(float(gamma), 6),
                "role": role,
                "literature_alpha_z_x": [round(float(v), 6) for v in lit],
                "calibrated_alpha_z_x": [round(float(v), 6) for v in cal],
            }
        )
    return {
        "schema": "mpm-soil-report/rft-points/1",
        "medium": literature["medium"],
        "units": {"alpha": "N/cm^3", "angles": "rad"},
        "measured_values": "not redistributed; Li, Zhang & Goldman (2013), Science 339:1408, table S5",
        "parameters": {"literature": literature["parameters"], "calibrated": calibrated["parameters"]},
        "numeric_settings": literature["numeric_settings"],
        "rmse_all_78_points": {"literature": literature["rmse_N_per_cm3"], "calibrated": calibrated["rmse_N_per_cm3"]},
        "points": points,
    }


def rft_calibration(archive: Path) -> dict:
    record = _load(archive / "rft_calibration.json")
    for experiment in record.get("experiments", {}).values():
        experiment.pop("reference", None)
    record["measured_values"] = "removed; Li, Zhang & Goldman (2013), table S5"
    return record


def search(run: Path, names) -> dict:
    candidates = []
    generation = 0
    for path in sorted((run / "batches").glob("batch-*.json")):
        batch = _load(path)
        purpose = batch["purpose"]
        if purpose == "train":
            generation += 1
        for trial, values, objective, valid, metrics in zip(
            batch["trial_ids"],
            batch["parameter_values"],
            batch["objectives"],
            batch["valid_mask"],
            batch["metrics"],
            strict=True,
        ):
            candidates.append(
                {
                    "batch": path.stem,
                    "purpose": purpose,
                    "generation": generation if purpose == "train" else None,
                    "trial": trial,
                    "parameters": dict(zip(names, values, strict=True)),
                    "objective": objective,
                    "max_normalized_error": metrics[1] if metrics and len(metrics) > 1 else None,
                    "valid": bool(valid),
                }
            )
    state = _load(run / "run_state.json")
    return {
        "schema": "mpm-soil-report/search/1",
        "termination_reason": state["termination_reason"],
        "elapsed_s": state["elapsed_s"],
        "budget_usage": state["budget_usage"],
        "identifiability": {
            k: state["analysis_reports"].get("identifiability", {}).get(k)
            for k in ("rank", "singular_values", "conclusive", "per_anchor_response_correlation")
        },
        "numerical": state["analysis_reports"].get("numerical"),
        "candidates": candidates,
    }


def clay(archive: Path, dataset: dict) -> dict:
    record = _load(archive / "clay_sweep_laboratory.json")
    tests = {}
    for name, test in record["tests"].items():
        spec = dataset["tests"][name]
        tests[name] = {
            "plate_width_m": spec["plate_width"],
            "boundary": spec["boundary"],
            "clay_height_m": spec["clay_height"],
            "batch": spec["batch"],
            "depth_mm": test["depth_mm"],
            "measured_kPa": [round(v, 3) for v in test["measured_kPa"]],
            "predicted_kPa": [round(v, 3) for v in test["predicted_kPa"]],
            "rmse_kPa": test["rmse_kPa"],
            "history_every_10th_step": test["history"],
            "wall_seconds": test["wall_seconds"],
            "measured_curve": {
                "penetration_mm": spec["penetration_mm"],
                "net_pressure_kPa": spec["net_pressure_kPa"],
            },
        }
    return {
        "schema": "mpm-soil-report/clay-laboratory/1",
        "measured_data": "Zwanenburg, Wittekoek, Alderlieste & Martinelli (2024), doi:10.5281/zenodo.8066899, CC BY 4.0; "
        "net pressure after contact detection and buoyancy correction",
        "profile": record["profile"],
        "parameters": record["parameters"],
        "numeric_settings": record["numeric_settings"],
        "setup": dataset["setup"],
        "tests": tests,
    }


def rake_traces(repo: Path, run: Path) -> dict:
    sys.path.insert(0, str(repo))
    from examples.mpm_soil.franka_rake.logs import (  # noqa: PLC0415
        experiments_from_logs,
        load_logs,
        stroke_slices,
    )
    from examples.mpm_soil.franka_rake.run import DEMO_LOGS, SCALE  # noqa: PLC0415

    logs = load_logs(DEMO_LOGS)
    experiments, settings = experiments_from_logs(logs, scale=SCALE)
    comparison = _load(run / "comparison.json")["experiments"]
    holdout = None
    for path in sorted((run / "batches").glob("batch-*.json")):
        batch = _load(path)
        if batch["purpose"] != "validation":
            continue
        for episode in batch["episode_records"]:
            if episode["experiment_id"] == "rake_holdout" and episode["trace"] and holdout is None:
                holdout = np.asarray(episode["trace"]["values"])
    strokes = {}
    for experiment in experiments:
        slices = stroke_slices(settings[experiment.id])
        reference = np.asarray(experiment.reference)
        mask = np.asarray(experiment.mask)[:, 0]
        if experiment.id == "rake_train":
            baseline = np.asarray(comparison["rake_train"]["baseline"])
            calibrated = np.asarray(comparison["rake_train"]["calibrated"])
        else:
            baseline, calibrated = None, holdout
        for entry in settings[experiment.id]["strokes"]:
            rows = slices[entry["name"]]
            strokes[entry["name"]] = {
                "group": experiment.group,
                "time_s": [round(t, 4) for t in entry["times"]],
                "scored": mask[rows].astype(bool).tolist(),
                "logged_N": np.round(reference[rows], 4).tolist(),
                "baseline_N": None if baseline is None else np.round(baseline[rows], 4).tolist(),
                "calibrated_N": np.round(calibrated[rows], 4).tolist(),
            }
    meta = _load(DEMO_LOGS / "recipe.json")
    return {
        "schema": "mpm-soil-report/rake-traces/1",
        "synthetic_logs": True,
        "recipe": meta,
        "force_frame": "robot base; force of the soil on the tool; columns fx, fy, fz",
        "strokes": strokes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--rft-run", type=Path, required=True)
    parser.add_argument("--rake-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    archive = repo / "examples" / "mpm_soil" / "archive"
    sys.path.insert(0, str(repo))
    from examples.mpm_soil.clay_plate.protocol import load_dataset  # noqa: PLC0415
    from examples.mpm_soil.rft_plate.run import TRAIN_POINTS  # noqa: PLC0415

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "rft_points.json", rft_points(archive, TRAIN_POINTS))
    _write(out / "rft_calibration.json", rft_calibration(archive))
    _write(out / "rft_search.json", search(args.rft_run, ("friction", "plate_friction")))
    _write(out / "clay_laboratory.json", clay(archive, load_dataset()))
    _write(out / "rake_result.json", _load(archive / "franka_rake_result.json"))
    _write(out / "rake_traces.json", rake_traces(repo, args.rake_run))
    _write(out / "rake_search.json", search(args.rake_run, ("friction", "cohesion", "tool_friction")))
    runs = [args.rft_run / "run_state.json", args.rake_run / "run_state.json", args.rake_run / "comparison.json"]
    sources = {
        str(p.resolve().relative_to(repo)): _sha256(p) for p in sorted(archive.glob("*.json")) + [r.resolve() for r in runs]
    }
    _write(out / "source_hashes.json", {"schema": "mpm-soil-report/source-hashes/1", "sha256": sources})


if __name__ == "__main__":
    main()
