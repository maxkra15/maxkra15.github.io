# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Summarize the MPM soil resolution study from results/<job>/variants.json into summary.json.

Granular: for each medium, parameter set and variant, the predictions at the 11 diagnostic
motions, their change against the 6 mm baseline, the change caused by a second particle
arrangement (the noise floor), normalized errors against the measurements, the objective over the
five training motions and the shallow-to-broadside lift ratio. Clay: RMSE and mean error per test
and variant.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SCALES = {"poppy_seeds_LP": 0.03, "poppy_seeds_CP": 0.048, "glass_0p3mm_LP": 0.021, "glass_0p3mm_CP": 0.034,
          "glass_3mm_CP": 0.025}
TRAIN = 5  # the first five diagnostic motions are the training motions
ORDER = ["v6", "v6-seed14", "v4", "v3", "v6-ppc3", "v6-p1d", "v6-pic27"]


def _key(p):
    return f"({p[0]:g}, {p[1]:.1f})"


def granular():
    media = {}
    for path in sorted(RESULTS.glob("rft-*/variants.json")):
        record = json.loads(path.read_text())
        entry = media.setdefault(record["medium"], {"points_deg": record["points_deg"],
                                                    "measured": record["measured"], "variants": {}})
        for name, variant in record["variants"].items():
            entry["variants"][name] = variant
    summary = {}
    for medium, entry in media.items():
        measured = np.asarray(entry["measured"])
        scale = SCALES[medium]
        points = entry["points_deg"]
        broadside, shallow = points.index([0, 90]), points.index([0, 18.435])
        out = {"scale": scale, "points_deg": points, "measured": measured.round(4).tolist(), "variants": {}}
        for name in [n for n in ORDER if n in entry["variants"]] + sorted(set(entry["variants"]) - set(ORDER)):
            variant = entry["variants"][name]
            if "error" in variant:
                out["variants"][name] = {"error": variant["error"], "wall_seconds": variant["wall_seconds"]}
                continue
            row = {"wall_seconds": round(variant["wall_seconds"]), "particles_per_world": round(variant["particles_per_world"]),
                   "peak_gpu_memory_MiB": variant.get("peak_gpu_memory_MiB"), "sets": {}}
            for set_name, predicted in variant["predicted"].items():
                predicted = np.asarray(predicted)
                error = (predicted - measured) / scale
                stats = {
                    "predicted": predicted.round(4).tolist(),
                    "error_scales": error.round(2).tolist(),
                    "train_objective": float((error[:TRAIN] ** 2).mean()),
                    "diagnostic_objective": float((error[TRAIN:] ** 2).mean()),
                    "max_error_scales": float(np.abs(error).max()),
                    "worst_point": points[int(np.abs(error).max(axis=1).argmax())],
                    "shallow_to_broadside_lift": float(predicted[shallow, 0] / predicted[broadside, 0]),
                }
                base = entry["variants"].get("v6", {}).get("predicted", {}).get(set_name)
                if base is not None:
                    change = (predicted - np.asarray(base)) / scale
                    stats["change_vs_v6_scales"] = change.round(2).tolist()
                    stats["max_change_vs_v6_scales"] = float(np.abs(change).max())
                    stats["rms_change_vs_v6_scales"] = float(np.sqrt((change**2).mean()))
                    stats["broadside_lift_change_pct"] = float(100 * (predicted[broadside, 0] / base[broadside][0] - 1))
                row["sets"][set_name] = stats
            out["variants"][name] = row
        out["measured_shallow_to_broadside_lift"] = float(measured[shallow, 0] / measured[broadside, 0])
        summary[medium] = out
    return summary


def clay():
    tests = {}
    for path in sorted(RESULTS.glob("clay-*/variants.json")):
        record = json.loads(path.read_text())
        for name, variant in record["variants"].items():
            if "error" in variant:
                tests.setdefault("errors", {})[name] = variant["error"]
                continue
            for test, entry in variant["tests"].items():
                predicted, measured = np.asarray(entry["predicted_kPa"]), np.asarray(entry["measured_kPa"])
                tests.setdefault(test, {})[name] = {
                    "rmse_kPa": round(float(entry["rmse_kPa"]), 2),
                    "mean_error_kPa": round(float((predicted - measured).mean()), 2),
                    "wall_seconds": round(entry["wall_seconds"]),
                    "voxel_size": variant["numeric_settings"]["voxel_size"],
                    "dt": variant["numeric_settings"]["dt"],
                    "collider_basis": variant["numeric_settings"].get("collider_basis", "S2"),
                    "peak_gpu_memory_MiB": variant.get("peak_gpu_memory_MiB"),
                }
    return tests


def main():
    summary = {"granular": granular(), "clay": clay()}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for medium, out in summary["granular"].items():
        print(f"\n{medium} (scale {out['scale']}): measured shallow/broadside lift {out['measured_shallow_to_broadside_lift']:.2f}")
        for name, row in out["variants"].items():
            if "error" in row:
                print(f"  {name:10s} ERROR {row['error'][:80]}")
                continue
            for set_name, s in row["sets"].items():
                print(f"  {name:10s} {set_name:10s} train J {s['train_objective']:5.2f}  diag J {s['diagnostic_objective']:5.2f}"
                      f"  max {s['max_error_scales']:4.1f} at {_key(s['worst_point']):12s} ratio {s['shallow_to_broadside_lift']:.2f}"
                      f"  broadside {s.get('broadside_lift_change_pct', 0):+5.1f}%  max change {s.get('max_change_vs_v6_scales', 0):4.2f}"
                      f"  ({row['wall_seconds']} s)")
    for test, variants in summary["clay"].items():
        print(f"\nclay {test}")
        for name, s in (variants.items() if isinstance(variants, dict) else []):
            print(f"  {name:18s} {s}" if not isinstance(s, dict) else
                  f"  {name:18s} RMSE {s['rmse_kPa']:6.1f}  mean {s['mean_error_kPa']:+6.1f}  ({s['wall_seconds']} s)")


if __name__ == "__main__":
    main()
