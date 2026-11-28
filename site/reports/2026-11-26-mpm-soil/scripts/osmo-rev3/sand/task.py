# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one job of the MPM sand and soil experiments on an OSMO GPU task (see job.json).

Kinds:
  rod_sweep / plate_sweep          predictions with one parameter set (numeric overrides allowed)
  rod_calibrate / plate_calibrate  the standard Tuning lifecycle of the case
  rod_variants / plate_variants    the same prediction under several numerical variants
Results are written to the output folder after every step.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path


def _visible(devices):
    """Keep the requested CUDA devices that this task can see."""
    import warp as wp

    count = wp.get_cuda_device_count()
    kept = [d for d in devices if int(str(d).split(":")[1]) < count]
    print("DEVICES", count, "visible, using", kept, flush=True)
    return kept


def rod(job, out):
    from examples.mpm_soil.rod_intrusion import run as case
    from examples.mpm_soil.rod_intrusion.protocol import literature_parameters, load_dataset

    dataset = load_dataset()
    medium = job["medium"]
    values = job.get("parameters", "literature")
    if values == "literature":
        values = literature_parameters(dataset, medium)
    if job["kind"] == "rod_calibrate":
        case.NUMERIC_SETTINGS.update(job.get("numeric", {}))
        if "devices" in case.NUMERIC_SETTINGS:
            case.NUMERIC_SETTINGS["devices"] = _visible(case.NUMERIC_SETTINGS["devices"])
        tuning = case.build_tuning(medium, workspace=out / "workspace", max_seconds=3600.0 * job.get("max_hours", 6.0))
        return _calibrate(tuning, out)
    variants = job.get("variants", [{"name": "base", "numeric": job.get("numeric", {})}])
    summary = {}
    for variant in variants:
        numeric = {**case.NUMERIC_SETTINGS, **variant.get("numeric", {})}
        began = time.monotonic()
        try:
            result = case.sweep(medium, variant.get("parameters", values), out / f"sweep_{variant['name']}.json",
                                numeric=numeric, selected=job.get("packings"))
            summary[variant["name"]] = {"rmse": result.get("rmse_N_per_m", result.get("rmse_N")), "wall_seconds": time.monotonic() - began}
        except Exception as error:
            summary[variant["name"]] = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
            print(summary[variant["name"]]["traceback"], flush=True)
        print("VARIANT_DONE", variant["name"], json.dumps(summary[variant["name"]])[:300], flush=True)
        (out / "variants.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary, 0


def plate(job, out):
    from examples.mpm_soil.plate_sinkage import run as case
    from examples.mpm_soil.plate_sinkage.protocol import METHODS, lab_parameters, load_dataset

    dataset = load_dataset()
    if job["kind"] == "plate_calibrate":
        case.NUMERIC_SETTINGS.update(job.get("numeric", {}))
        return _calibrate(case.build_tuning(workspace=out / "workspace"), out)
    values = job.get("parameters", "lab")
    sets = {m: lab_parameters(dataset, m) for m in METHODS} if values == "lab" else {m: dict(values) for m in METHODS}
    if "methods" in job:
        sets = {m: sets[m] for m in job["methods"]}
    variants = job.get("variants", [{"name": "base", "numeric": job.get("numeric", {})}])
    summary = {}
    for variant in variants:
        numeric = {**case.NUMERIC_SETTINGS, **variant.get("numeric", {})}
        began = time.monotonic()
        try:
            chosen = sets if "parameters" not in variant else {m: dict(variant["parameters"]) for m in sets}
            result = case.sweep(chosen, out / f"sweep_{variant['name']}.json", numeric=numeric)
            summary[variant["name"]] = {"predicted": result["predicted"], "wall_seconds": time.monotonic() - began}
        except Exception as error:
            summary[variant["name"]] = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
            print(summary[variant["name"]]["traceback"], flush=True)
        print("VARIANT_DONE", variant["name"], json.dumps(summary[variant["name"]])[:300], flush=True)
        (out / "variants.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary, 0


def pullout(job, out):
    from examples.mpm_soil.rod_pullout import run as case
    from examples.mpm_soil.rod_pullout.protocol import literature_parameters, load_dataset

    numeric = {**case.NUMERIC_SETTINGS, **job.get("numeric", {})}
    if job["kind"] == "pullout_calibrate":
        return _calibrate(case.build_tuning(workspace=out / "workspace", numeric=numeric,
                                            sand_friction=bool(job.get("sand_friction", False))), out)
    values = job.get("parameters", "literature")
    if values == "literature":
        values = literature_parameters(load_dataset())
    began = time.monotonic()
    result = case.sweep(values, out / "sweep_base.json", numeric=numeric, selected=job.get("geometries"))
    return {"wall_seconds": time.monotonic() - began, "keys": sorted(result)}, 0


def probe(job, out):
    from examples.mpm_soil.probe_penetration import run as case
    from examples.mpm_soil.probe_penetration.protocol import diameter_key, generic_parameters

    case.NUMERIC_SETTINGS.update(job.get("numeric", {}))
    if job["kind"] == "probe_calibrate":
        holdout = tuple(diameter_key(d) for d in job.get("holdout_diameters", [11.0, 9.0, 7.0]))
        tuning = case.build_tuning(workspace=out / "workspace", max_seconds=3600.0 * job.get("max_hours", 12.0),
                                   holdout_diameters=holdout)
        return _calibrate(tuning, out)
    values = job.get("parameters", "generic")
    if values == "generic":
        values = generic_parameters()
    began = time.monotonic()
    result = case.sweep(values, out / "sweep_base.json", diameters=[diameter_key(d) for d in job["diameters"]],
                        shapes=job.get("shapes", case.FOUR_TIPS))
    return {"wall_seconds": time.monotonic() - began,
            "probes": {k: {f: v.get(f) for f in ("rmse_N", "normalized_rmse")} for k, v in result["probes"].items()}}, 0


def _calibrate(tuning, out):
    from examples.mpm_soil.verify import summarize

    result = tuning.run()
    record = summarize(result.run_directory)
    (out / "calibration_summary.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    summary = {
        "acceptance": result.acceptance,
        "termination_reason": result.termination_reason,
        "parameters": result.parameters,
        "run_directory": str(Path(result.run_directory).relative_to(out)),
    }
    return summary, 0


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    out = Path(os.environ["OSMO_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "job.json").write_text(json.dumps(job, indent=1) + "\n")
    began = time.time()
    if job["kind"].startswith("rod_"):
        summary, code = rod(job, out)
    elif job["kind"].startswith("plate_"):
        summary, code = plate(job, out)
    elif job["kind"].startswith("pullout_"):
        summary, code = pullout(job, out)
    elif job["kind"].startswith("probe_"):
        summary, code = probe(job, out)
    else:
        raise SystemExit(f"unknown job kind {job['kind']}")
    record = {"job": job, "summary": summary, "wall_seconds": time.time() - began}
    (out / "result.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print("RESULT", json.dumps(record, default=str)[:2000], flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
