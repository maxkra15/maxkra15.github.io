# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one MPM soil job of the five-media study on an OSMO GPU task (see job.json)."""
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    out = Path(os.environ["OSMO_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "job.json").write_text(json.dumps(job, indent=1) + "\n")
    began = time.time()
    kind = job["kind"]
    code = 0
    if kind in ("rft_sweep", "rft_calibrate"):
        from examples.mpm_soil.rft_plate import run as rft
        from examples.mpm_soil.rft_plate.protocol import load_dataset

        if "dt" in job:
            rft.NUMERIC_SETTINGS["dt"] = float(job["dt"])
        medium = job["medium"]
        if kind == "rft_sweep":
            dataset = load_dataset()
            values = job["parameters"]
            if values == "literature":
                values = rft.literature_parameters(dataset, medium)
            result = rft.sweep(medium, values, out / "sweep.json")
            summary = {k: result[k] for k in ("medium", "parameters", "rmse_N_per_cm3", "wall_seconds")}
        else:
            from examples.mpm_soil.verify import summarize

            tuning = rft.build_tuning(medium, workspace=out / "workspace")
            result = tuning.run()
            record = summarize(result.run_directory)
            (out / "calibration_summary.json").write_text(json.dumps(record, indent=1) + "\n")
            summary = {
                "acceptance": result.acceptance,
                "termination_reason": result.termination_reason,
                "parameters": result.parameters,
                "run_directory": str(Path(result.run_directory).relative_to(out)),
            }
            code = 0 if result.acceptance in ("accepted", "rejected", "rejected_validation", "rejected_numerics") else 1
    elif kind == "clay_sweep":
        from examples.mpm_soil.clay_plate import run as clay

        clay.NUMERIC_SETTINGS["dt"] = float(job["dt"])
        clay.TRAIN_TESTS = tuple(job["tests"])
        clay.VALIDATION_TESTS = ()
        values = clay.laboratory_parameters(clay.load_dataset())
        result = clay.sweep(values, out / "sweep.json", profile="shansep")
        summary = {test: {"rmse_kPa": entry["rmse_kPa"]} for test, entry in result["tests"].items()}
    else:
        raise SystemExit(f"unknown job kind {kind}")
    record = {"job": job, "summary": summary, "wall_seconds": time.time() - began}
    (out / "result.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print("RESULT", json.dumps(record, default=str), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
