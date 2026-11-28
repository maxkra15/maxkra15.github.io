# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collect the resolution study and the sand and soil experiments into report data files.

Inputs are the summary files written by the analysis scripts of the two OSMO workspaces
(``analyze.py``), plus the processed evidence of newton-calibration. Usage:

    python export_studies.py --resolution <resolution workspace> --sand <sand workspace> --out <report>/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolution(root: Path) -> dict:
    summary = json.loads((root / "summary.json").read_text())
    granular = {}
    for medium, entry in summary["granular"].items():
        variants = {}
        for name, row in entry["variants"].items():
            if "error" in row:
                variants[name] = {"error": row["error"]}
                continue
            variants[name] = {
                "particles_per_world": row["particles_per_world"],
                "wall_seconds": row["wall_seconds"],
                "sets": {
                    s: {k: v[k] for k in ("predicted", "error_scales", "train_objective", "diagnostic_objective",
                                          "max_error_scales", "worst_point", "shallow_to_broadside_lift",
                                          "change_vs_v6_scales", "max_change_vs_v6_scales", "broadside_lift_change_pct")
                        if k in v}
                    for s, v in row["sets"].items()
                },
            }
        granular[medium] = {k: entry[k] for k in ("scale", "points_deg", "measured", "measured_shallow_to_broadside_lift")} | {"variants": variants}
    return {
        "schema": "mpm-soil-report/resolution-study/1",
        "description": "Discretization study: 11 diagnostic plate motions per medium (literature and calibrated parameters) "
                       "at 6, 4 and 3 mm voxels, 27 particles per voxel, P1d strain, pic27 contact basis and a second "
                       "particle arrangement; clay tests 1 and 3 at 2.5 and 1.25 mm voxels and with pic27, at 0.5 and 0.25 ms. "
                       "OSMO L40S, Warp 1.20.0.dev20261120.",
        "granular": granular,
        "clay": summary["clay"],
        "sources_sha256": {"summary.json": _sha(root / "summary.json")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--sand", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = args.out / "resolution_study.json"
    path.write_text(json.dumps(resolution(args.resolution), indent=1) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    if args.sand and (args.sand / "summary.json").is_file():
        sand = json.loads((args.sand / "summary.json").read_text())
        for name, key in (("rod_intrusion.json", "rod"), ("plate_sinkage.json", "plate"), ("rod_pullout.json", "pullout"),
                          ("probe_penetration.json", "probe")):
            path = args.out / name
            record = {"schema": f"mpm-soil-report/{key}/1", **sand[key],
                      "sources_sha256": {"summary.json": _sha(args.sand / "summary.json")}}
            path.write_text(json.dumps(record, indent=1) + "\n")
            print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
