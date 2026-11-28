# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Summarize the sand and soil experiment jobs (rod intrusion, plate sinkage) into summary.json.

Run with the newton-calibration environment from its repository root:
    PYTHONPATH=. .venv/bin/python <this folder>/analyze.py
"""

import json
import math
from pathlib import Path

import numpy as np

from examples.mpm_soil.plate_sinkage.protocol import METHODS, lab_parameters, reference as plate_reference
from examples.mpm_soil.plate_sinkage.protocol import load_dataset as load_plate
from examples.mpm_soil.rod_intrusion.protocol import (
    load_dataset as load_rod,
    observation_scale,
    reference as rod_reference,
    slope,
    split,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MEDIA = {"sand": "sand", "glass": "glass_beads", "lhs1": "lhs1"}


def _first(*paths):
    for path in paths:
        if path.is_file():
            return path
    return paths[-1]


def _calibration_record(path, roles):
    record = json.loads(path.read_text())
    out = {k: record.get(k) for k in ("acceptance", "termination_reason", "parameters", "baseline_parameters",
                                       "validation", "training_retest", "identifiability", "numerical")}
    out["experiments"] = {
        name: {k: e.get(k) for k in ("role", "times", "reference", "scale", "baseline", "calibrated")}
        for name, e in record.get("experiments", {}).items() if any(name.startswith(r) for r in roles)
    }
    return out


def _search_counts(run_dir):
    """Search trials of a calibration run and how many were invalid (for example outside the bounds)."""
    total = invalid = 0
    for path in sorted(run_dir.glob("workspace/*/batches/*.json")):
        batch = json.loads(path.read_text())
        if batch["purpose"] == "train":
            total += len(batch["trial_ids"])
            invalid += sum(not v for v in batch["valid_mask"])
    return {"search_trials": total, "invalid_search_trials": invalid}


def pullout_summary():
    out = {}
    sweep = RESULTS / "pullout-lit" / "sweep_base.json"
    if sweep.is_file():
        r = json.loads(sweep.read_text())
        out["literature"] = {k: r[k] for k in ("parameters", "geometries", "measured_N", "std_N", "predicted_N",
                                                "normalized_error", "rmse_N", "normalized_rmse")}
        out["literature"]["details"] = {g: {k: v[k] for k in ("voxel_size", "particles", "resting_force_N",
                                                               "material_peak_N", "peak_rise_m", "rod_weight_N")}
                                        for g, v in r["details"].items()}
    grid = RESULTS / "pullout-cal" / "calibration_summary.json"
    if grid.is_file():
        out["calibration_grid"] = _calibration_record(grid, ("pullout",))
    calib = _first(RESULTS / "pullout-cal3" / "calibration_summary.json", RESULTS / "pullout-cal2" / "calibration_summary.json")
    if calib.is_file():
        out["calibration"] = _calibration_record(calib, ("pullout",))
        out["calibration"].update(_search_counts(calib.parent))
    corner = RESULTS / "pullout-corner" / "sweep_base.json"
    if corner.is_file():
        r = json.loads(corner.read_text())
        out["corner_check"] = {k: r[k] for k in ("parameters", "geometries", "measured_N", "predicted_N", "normalized_error",
                                                  "rmse_N", "normalized_rmse")}
        out["corner_check"]["ratio"] = [p / m for p, m in zip(r["predicted_N"], r["measured_N"], strict=True)]
    return out


def probe_summary():
    out = {}
    sweep = _first(RESULTS / "probe-lit2" / "sweep_base.json", RESULTS / "probe-lit" / "sweep_base.json")
    if sweep.is_file():
        r = json.loads(sweep.read_text())
        out["generic"] = {"parameters": r["parameters"], "report_depths_mm": r["report_depths_mm"],
                          "probes": {k: {f: v[f] for f in ("shapes", "voxel_size", "particles_per_world", "start_gap_m",
                                                           "measured_N", "predicted_N", "scale_N", "rmse_N",
                                                           "normalized_rmse")}
                                     for k, v in r["probes"].items()}}
    calib = _first(RESULTS / "probe-cal2" / "calibration_summary.json", RESULTS / "probe-cal" / "calibration_summary.json")
    if calib.is_file():
        out["calibration"] = _calibration_record(calib, ("probe",))
    return out


def rod_sweep_slopes(path):
    """Slopes [N/m] of every simulated curve of a sweep file (works for either observation format)."""
    record = json.loads(path.read_text())
    out = {}
    for key, curve in record["curves"].items():
        out[float(key)] = slope(np.asarray(curve["depth_m"]), np.asarray(curve["force_N"]))
    return record, out


def rod_summary():
    data = load_rod()
    summary = {}
    for tag, medium in MEDIA.items():
        entry = {"scale_N_per_m": observation_scale(data, medium),
                 "critical_fraction": data["media"][medium]["critical_fraction"],
                 "friction_deg": data["media"][medium]["friction_deg"],
                 "grain_density": data["media"][medium]["grain_density"]}
        sweep_path = _first(RESULTS / f"rod-{tag}-lit3" / "sweep_base.json", RESULTS / f"rod-{tag}-lit" / "sweep_base.json")
        if sweep_path.is_file():
            record, slopes = rod_sweep_slopes(sweep_path)
            phis = sorted(slopes)
            measured = rod_reference(data, medium, phis)[:, 0]
            predicted = np.array([slopes[p] for p in phis])
            stds = {round(v["packing_fraction"], 4): v["slope_std_N_per_m"] for v in data["media"][medium]["packings"].values()}
            counts = {round(v["packing_fraction"], 4): v["n"] for v in data["media"][medium]["packings"].values()}
            entry["literature"] = {
                "parameters": record["parameters"],
                "packings": phis,
                "measured": measured.round(2).tolist(),
                "measured_std": [stds[round(p, 4)] for p in phis],
                "runs": [counts[round(p, 4)] for p in phis],
                "predicted": predicted.round(2).tolist(),
                "ratio": (predicted / measured).round(3).tolist(),
                "rmse_N_per_m": float(np.sqrt(np.mean((predicted - measured) ** 2))),
            }
        runs = {}
        for name in (f"rod-{tag}-cal", f"rod-{tag}-cal3", f"rod-{tag}-cal4", f"rod-{tag}-cal5"):
            path = RESULTS / name / "calibration_summary.json"
            if path.is_file():
                record = json.loads(path.read_text())
                numerical = record.get("numerical") or {}
                runs[name] = {
                    "acceptance": record["acceptance"],
                    "termination_reason": record.get("termination_reason"),
                    "max_iterations": record.get("numeric_settings", {}).get("max_iterations"),
                    "devices": record.get("numeric_settings", {}).get("devices"),
                    "time_step_check": {
                        "passed": numerical.get("passed"),
                        "tolerance": numerical.get("tolerance"),
                        "difference_max": [p.get("feature_difference_max") for p in numerical.get("probes", [])],
                    },
                    "budget_usage": record.get("budget_usage"),
                    "elapsed_s": record.get("elapsed_s"),
                    "best_candidate": record.get("best_candidate"),
                    **_search_counts(path.parent),
                }
        if runs:
            entry["calibration_runs"] = runs
        calib = _first(*(RESULTS / f"rod-{tag}-{s}" / "calibration_summary.json" for s in ("cal5", "cal4", "cal3", "cal")))
        if calib.is_file():
            record = json.loads(calib.read_text())
            train, holdout = split(data, medium)
            exp = record["experiments"]
            entry["calibration"] = {
                "run": calib.parent.name,
                "acceptance": record["acceptance"],
                "termination_reason": record.get("termination_reason"),
                "parameters": record["parameters"],
                "best_candidate": record.get("best_candidate"),
                "identifiability": record.get("identifiability"),
                "validation": record.get("validation"),
                "training_retest": record.get("training_retest"),
                "train_packings": train,
                "holdout_packings": holdout,
                "train_measured": np.asarray(exp["rod_train"]["reference"])[:, 0].round(2).tolist(),
                "train_calibrated": np.asarray(exp["rod_train"].get("calibrated", [[np.nan]] * len(train)))[:, 0].round(2).tolist(),
                "holdout_measured": np.asarray(exp["rod_holdout"]["reference"])[:, 0].round(2).tolist(),
                "holdout_calibrated": np.asarray(exp["rod_holdout"].get("calibrated", [[np.nan]] * len(holdout)))[:, 0].round(2).tolist(),
            }
        best = RESULTS / f"rod-{tag}-best" / "sweep_base.json"
        if best.is_file():
            record, slopes = rod_sweep_slopes(best)
            phis = sorted(slopes)
            measured = rod_reference(data, medium, phis)[:, 0]
            predicted = np.array([slopes[p] for p in phis])
            train, holdout = split(data, medium)
            scale = observation_scale(data, medium)
            entry["best_law_sweep"] = {
                "note": "diagnostic outside the lifecycle: the best training candidate of the rejected calibration, "
                        "simulated at every packing with the sweep seed",
                "parameters": record["parameters"],
                "packings": phis,
                "measured": measured.round(2).tolist(),
                "predicted": predicted.round(2).tolist(),
                "error_scales": ((predicted - measured) / scale).round(3).tolist(),
                "train_packings": train,
                "holdout_packings": holdout,
            }
        summary[medium] = entry
    checks = RESULTS / "rod-sand-checks"
    if checks.is_dir():
        base = None
        rows = {}
        for path in sorted(checks.glob("sweep_*.json")):
            name = path.stem.removeprefix("sweep_")
            _, slopes = rod_sweep_slopes(path)
            rows[name] = slopes
        base = rows.get("base")
        summary["sand_checks"] = {
            name: {f"{p:.3f}": round(v, 2) for p, v in s.items()}
            | ({"change_pct": {f"{p:.3f}": round(100 * (s[p] / base[p] - 1), 1) for p in s}} if base else {})
            for name, s in rows.items()
        }
    return summary


def plate_summary():
    data = load_plate()
    summary = {}
    lab = _first(RESULTS / "plate-lab2" / "sweep_base.json", RESULTS / "plate-lab" / "sweep_base.json")
    if lab.is_file():
        record = json.loads(lab.read_text())
        summary["lab"] = {
            m: {"parameters": record["parameters"][m], "measured": np.round(record["measured"][m], 1).tolist(),
                "predicted": np.round(record["predicted"][m], 1).tolist()}
            for m in record["predicted"]
        }
    calib = _first(RESULTS / "plate-cal2" / "calibration_summary.json", RESULTS / "plate-cal" / "calibration_summary.json")
    if calib.is_file():
        record = json.loads(calib.read_text())
        exp = record["experiments"]
        summary["calibration"] = {
            "acceptance": record["acceptance"],
            "parameters": record["parameters"],
            "validation": record.get("validation"),
            "train_measured": np.round(exp["plate_train"]["reference"], 1).tolist(),
            "train_calibrated": np.round(exp["plate_train"].get("calibrated", []), 1).tolist(),
            "holdout_measured": np.round(exp["plate_holdout"]["reference"], 1).tolist(),
            "holdout_calibrated": np.round(exp["plate_holdout"].get("calibrated", []), 1).tolist(),
        }
    checks = RESULTS / "plate-checks" / "variants.json"
    if checks.is_file():
        record = json.loads(checks.read_text())
        base = np.asarray(record["base"]["predicted"]["4"]) if "base" in record and "predicted" in record["base"] else None
        summary["checks"] = {}
        for name, v in record.items():
            if "predicted" not in v:
                summary["checks"][name] = {"error": v.get("error")}
                continue
            p = np.asarray(v["predicted"]["4"])
            summary["checks"][name] = {"predicted": p.round(1).tolist(),
                                       "change_pct": (100 * (p / base - 1)).round(1).tolist() if base is not None else None}
    grid = {}
    for m in METHODS:
        path = RESULTS / f"plate-grid-m{m}" / "variants.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        measured = plate_reference(data, [m])[0]
        rows = []
        for name, v in record.items():
            if "predicted" not in v:
                continue
            p = np.asarray(v["predicted"][str(m)])
            mu = float(name.split("-")[0][2:])
            c = float(name.split("-")[1][1:])
            rows.append({"mu": mu, "cohesion_kPa": c, "predicted": p.round(1).tolist(),
                         "rmse_kPa": float(np.sqrt(np.mean((p - measured) ** 2)))})
        rows.sort(key=lambda r: r["rmse_kPa"])
        grid[m] = {"measured": measured.round(1).tolist(), "best": rows[:3], "all": rows}
    summary["grid"] = grid
    summary["lab_parameters"] = {m: lab_parameters(data, m) for m in METHODS}
    summary["measured_curves"] = {
        m: {"n": v["n"], "dry_density": v["dry_density"], "pressure_mean_kPa": v["pressure_mean_kPa"][::4],
            "pressure_std_kPa": v["pressure_std_kPa"][::4]}
        for m, v in data["plate_sinkage"]["methods"].items()
    }
    summary["measured_curves_sinkage_mm"] = data["sinkage_grid_mm"][::4]
    return summary


def main():
    summary = {"rod": rod_summary(), "plate": plate_summary(), "pullout": pullout_summary(), "probe": probe_summary()}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    rod = summary["rod"]
    for medium in MEDIA.values():
        e = rod.get(medium, {})
        if "literature" in e:
            lit = e["literature"]
            print(f"{medium}: literature K ratio sim/measured {lit['ratio'][0]:.2f} (loosest) .. {lit['ratio'][-1]:.2f} (densest), "
                  f"RMSE {lit['rmse_N_per_m']:.0f} N/m (scale {e['scale_N_per_m']})")
        if "calibration" in e:
            c = e["calibration"]
            print(f"   calibration {c['acceptance']} {c['parameters']}")
    if "sand_checks" in rod:
        for name, v in rod["sand_checks"].items():
            print(f"   check {name}: {v.get('change_pct')}")
    plate = summary["plate"]
    for m, v in plate.get("lab", {}).items():
        print(f"plate method {m} lab: measured {v['measured']} predicted {v['predicted']}")
    if "calibration" in plate:
        print("plate calibration", plate["calibration"]["acceptance"], plate["calibration"]["parameters"])
    for name, v in plate.get("checks", {}).items():
        print(f"   plate check {name}: {v.get('change_pct')}")
    pull = summary["pullout"]
    if "literature" in pull:
        print("pullout literature", round(pull["literature"]["normalized_rmse"], 2), "normalized RMSE;",
              "predicted/measured", [round(p / m, 2) for p, m in zip(pull["literature"]["predicted_N"], pull["literature"]["measured_N"])])
    if "calibration" in pull:
        print("pullout calibration", pull["calibration"]["acceptance"], pull["calibration"]["parameters"])
    probe = summary["probe"]
    for k, v in probe.get("generic", {}).get("probes", {}).items():
        print(f"probe {k} generic normalized RMSE {v['normalized_rmse']}")
    if "calibration" in probe:
        print("probe calibration", probe["calibration"]["acceptance"], probe["calibration"]["parameters"])
    for m, g in plate.get("grid", {}).items():
        b = g["best"][0]
        print(f"   grid method {m}: best mu {b['mu']} c {b['cohesion_kPa']} kPa RMSE {b['rmse_kPa']:.0f} kPa; measured {g['measured']}")


if __name__ == "__main__":
    main()
