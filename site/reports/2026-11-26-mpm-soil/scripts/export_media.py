# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collect the five-media plate-intrusion results into data/rft_media.json.

Loosely packed poppy seeds come from the archived local runs of the newton-calibration MPM
soil case; the other four media come from the OSMO results folder (one sweep and one
calibration per medium). Measured slopes are not copied. Run from a newton-calibration
checkout::

    PYTHONPATH=. python export_media.py --repo . --results <osmo>/results --out <report>/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

MEDIA = {
    "poppy_seeds_LP": "poppy-lp",
    "poppy_seeds_CP": "poppy-cp",
    "glass_0p3mm_LP": "glass03-lp",
    "glass_0p3mm_CP": "glass03-cp",
    "glass_3mm_CP": "glass3-cp",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _order(points, reference_points, values):
    """Reorder ``values`` given at ``points`` to the order of ``reference_points``."""
    points = np.asarray(points)
    out = []
    for p in reference_points:
        index = int(np.argmin(np.abs(points - np.asarray(p)).max(axis=1)))
        if np.abs(points[index] - np.asarray(p)).max() > 1e-3:
            raise ValueError(f"point {p} missing")
        out.append([round(float(v), 6) for v in values[index]])
    return out


def _calibration(summary: dict, dataset, medium, train_points):
    from examples.mpm_soil.rft_plate.protocol import measured_grid  # noqa: PLC0415

    parameters = summary["parameters"]
    record = {
        "dt_s": summary["numeric_settings"]["dt"],
        "acceptance": summary["acceptance"],
        "termination_reason": summary["termination_reason"],
        "parameters": parameters,
        "friction_angle_deg": math.degrees(math.asin(parameters["friction"])) if parameters else None,
        "baseline_parameters": summary["baseline_parameters"],
        "parameter_definitions": summary["parameter_definitions"],
        "identifiability": summary["identifiability"],
        "numerical": summary["numerical"],
        "training_retest": summary["training_retest"],
        "validation": summary["validation"],
        "budget_usage": summary["budget_usage"],
        "elapsed_s": summary["elapsed_s"],
        "run_id": summary["run_id"],
    }
    experiments = summary["experiments"]
    predicted = None
    if "calibrated" in experiments.get("rft_holdout", {}):
        train = [tuple(p) for p in train_points]
        holdout = [p for p in measured_grid(dataset, medium) if not any(np.allclose(p, q) for q in train)]
        points = train + holdout
        values = np.concatenate([experiments["rft_train"]["calibrated"], experiments["rft_holdout"]["calibrated"]])
        predicted = (points, values)
    return record, predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from examples.mpm_soil.rft_plate.protocol import load_dataset, measured_grid  # noqa: PLC0415
    from examples.mpm_soil.rft_plate.run import (  # noqa: PLC0415
        TRAIN_POINTS,
        friction_bounds,
        literature_parameters,
        observation_scale,
    )

    dataset = load_dataset()
    archive = args.repo / "examples" / "mpm_soil" / "archive"
    output = {
        "schema": "mpm-soil-report/rft-media/1",
        "units": {"alpha": "N/cm^3", "angles": "rad"},
        "measured_values": "not redistributed; Li, Zhang & Goldman (2013), Science 339:1408, table S5",
        "training_points_rad": [list(p) for p in TRAIN_POINTS],
        "media": {},
        "sources_sha256": {},
    }
    for medium, tag in MEDIA.items():
        points = [list(p) for p in measured_grid(dataset, medium)]
        entry = {
            "angle_deg": dataset["media"][medium]["repose_deg"],
            "bulk_density": dataset["media"][medium]["bulk_density"],
            "observation_scale": observation_scale(dataset, medium),
            "friction_bounds": list(friction_bounds(dataset, medium)),
            "literature_parameters": literature_parameters(dataset, medium),
            "points_rad": [[round(b, 6), round(g, 6)] for b, g in points],
        }
        if medium == "poppy_seeds_LP":
            sweep_path = archive / "rft_sweep_literature.json"
            entry["source"] = "local RTX PRO 6000, Warp 1.20.0.dev20261117 (published run)"
        else:
            sweep_path = args.results / f"rft-sweep-{tag}" / "sweep.json"
            entry["source"] = "OSMO L40S, Warp 1.20.0.dev20261120"
        sweep = json.loads(sweep_path.read_text())
        entry["literature"] = _order(sweep["points"], points, np.asarray(sweep["predicted"]))
        output["sources_sha256"][str(sweep_path.name if medium == "poppy_seeds_LP" else sweep_path.relative_to(args.results))] = _sha(sweep_path)
        calibrations = {}
        if medium == "poppy_seeds_LP":
            summary_path = archive / "rft_calibration.json"
            summary = json.loads(summary_path.read_text())
            summary.setdefault("medium", medium)
            record, predicted = _calibration(summary, dataset, medium, TRAIN_POINTS)
            calibrations["dt_1ms"] = record
            output["sources_sha256"][summary_path.name] = _sha(summary_path)
            if predicted:
                entry["calibrated"] = _order(predicted[0], points, predicted[1])
        else:
            for key, folder in (("dt_1ms", f"rft-cal-{tag}"), ("dt_0.5ms", f"rft-cal-{tag}-dt05")):
                summary_path = args.results / folder / "calibration_summary.json"
                if not summary_path.is_file():
                    continue
                record, predicted = _calibration(json.loads(summary_path.read_text()), dataset, medium, TRAIN_POINTS)
                calibrations[key] = record
                output["sources_sha256"][str(summary_path.relative_to(args.results))] = _sha(summary_path)
                if predicted and record["acceptance"] in ("accepted", "rejected", "rejected_validation"):
                    entry["calibrated"] = _order(predicted[0], points, predicted[1])
                    entry["calibrated_from"] = key
        entry["calibrations"] = calibrations
        output["media"][medium] = entry
    path = args.out / "rft_media.json"
    path.write_text(json.dumps(output, indent=1) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
