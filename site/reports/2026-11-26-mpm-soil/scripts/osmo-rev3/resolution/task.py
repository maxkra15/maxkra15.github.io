# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one job of the MPM soil resolution study on a GPU (see job.json).

Kinds:
  rft_variants   diagnostic plate motions of one medium for several parameter sets, repeated for each
                 numerical variant (voxel size, particles per cell, strain basis, collider basis, seed)
  clay_variants  centrifuge plate tests with the laboratory strength, repeated for each numerical variant

Each variant runs on its own; a failing variant is recorded and the job continues. Results are
written after every variant to variants.json in the output folder.
"""

import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


class GPUMemory:
    """Poll nvidia-smi for the peak memory used on the device while a variant runs."""

    def __init__(self, period=2.0):
        self.period, self.peak, self._stop = period, 0, threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            try:
                used = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.split()
                self.peak = max(self.peak, int(used[0]))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=15)


def _closest(grid, target):
    best = min(grid, key=lambda p: max(abs(p[0] - target[0]), abs(p[1] - target[1])))
    if max(abs(best[0] - target[0]), abs(best[1] - target[1])) > 1e-3:
        raise ValueError(f"point {target} is not on the measured grid")
    return (float(best[0]), float(best[1]))


def rft_variants(job, out):
    import numpy as np

    from examples.mpm_soil.rft_plate import run as rft
    from examples.mpm_soil.rft_plate.backend import RFTAsset
    from examples.mpm_soil.rft_plate.protocol import load_dataset, measured_grid, reference
    from newton_calibration import Template

    dataset = load_dataset()
    medium = job["medium"]
    grid = [tuple(p) for p in measured_grid(dataset, medium)]
    points = [_closest(grid, (math.radians(b), math.radians(g))) for b, g in job["points_deg"]]
    fixed = {"bulk_density": dataset["media"][medium]["bulk_density"], "gravity": 9.81}
    names = list(job["parameter_sets"])
    sets = {
        n: rft.literature_parameters(dataset, medium) if v == "literature" else {k: float(x) for k, x in v.items()}
        for n, v in job["parameter_sets"].items()
    }
    candidates = [{**sets[n], **fixed} for n in names]
    results = {
        "kind": "rft_variants",
        "medium": medium,
        "points": [list(p) for p in points],
        "points_deg": job["points_deg"],
        "measured": reference(dataset, medium, points).tolist(),
        "parameter_sets": {n: c for n, c in zip(names, candidates)},
        "variants": {},
    }
    for variant in job["variants"]:
        numeric = {**rft.NUMERIC_SETTINGS, "dt": float(job["dt"]), **variant.get("numeric", {})}
        discretization = {"particles_per_cell_axis": int(variant.get("particles_per_cell_axis", 2))}
        seed = int(variant.get("seed", 13))
        settings = {"seed": seed, "medium": medium, "points": [list(p) for p in points]}
        record = {"numeric_settings": numeric, "discretization": discretization, "seed": seed}
        print(f"VARIANT {variant['name']} {json.dumps(record)}", flush=True)
        began = time.monotonic()
        try:
            with GPUMemory() as memory:
                template = Template(
                    numeric_settings=numeric,
                    discretization=discretization,
                    fixed_parameters=fixed,
                    experiment_settings={"sweep": settings},
                    qualification_mode="repeatability",
                ).freeze()
                environment = RFTAsset(sets[names[0]], config={"medium": medium}).create_batch_environment(
                    template, candidates, template.experiment_settings["sweep"]
                )
                record["build_seconds"] = time.monotonic() - began
                worlds = sum(len(layout["layout"]) for _, layout in environment._models)
                record["particles_per_world"] = sum(m.particle_count for m, _ in environment._models) / worlds
                traces = environment.run(np.arange(len(points), dtype=np.float64), seed=0, deadline=None)
            record["predicted"] = {n: np.asarray(t.values).tolist() for n, t in zip(names, traces)}
            record["curves"] = {
                n: {
                    f"{b:+.4f},{g:+.4f}": [
                        {"depth_m": c["depth"][::5].tolist(), "force_N": c["force"][::5, [0, 2]].tolist()}
                        for c in environment.curves[ci, pi]
                    ]
                    for pi, (b, g) in enumerate(points)
                }
                for ci, n in enumerate(names)
            }
            record["peak_gpu_memory_MiB"] = memory.peak
            del environment
        except Exception as error:  # record and continue with the next variant
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()
            print(record["traceback"], flush=True)
        record["wall_seconds"] = time.monotonic() - began
        results["variants"][variant["name"]] = record
        (out / "variants.json").write_text(json.dumps(results) + "\n")
        print(f"VARIANT_DONE {variant['name']} {record['wall_seconds']:.0f} s {record.get('error', 'ok')}", flush=True)
    return results


def clay_variants(job, out):
    from examples.mpm_soil.clay_plate import run as clay

    base = dict(clay.NUMERIC_SETTINGS)
    values = clay.laboratory_parameters(clay.load_dataset())
    clay.TRAIN_TESTS = tuple(job["tests"])
    clay.VALIDATION_TESTS = ()
    results = {"kind": "clay_variants", "tests": job["tests"], "parameters": values, "variants": {}}
    for variant in job["variants"]:
        # sweep() and its template read this dictionary, so update it in place.
        clay.NUMERIC_SETTINGS.clear()
        clay.NUMERIC_SETTINGS.update({**base, **variant["numeric"]})
        record = {"numeric_settings": dict(clay.NUMERIC_SETTINGS)}
        print(f"VARIANT {variant['name']} {json.dumps(record)}", flush=True)
        began = time.monotonic()
        try:
            with GPUMemory() as memory:
                result = clay.sweep(values, out / f"sweep_{variant['name']}.json", profile="shansep")
            record["tests"] = result["tests"]
            record["peak_gpu_memory_MiB"] = memory.peak
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()
            print(record["traceback"], flush=True)
        record["wall_seconds"] = time.monotonic() - began
        results["variants"][variant["name"]] = record
        (out / "variants.json").write_text(json.dumps(results) + "\n")
        print(f"VARIANT_DONE {variant['name']} {record['wall_seconds']:.0f} s {record.get('error', 'ok')}", flush=True)
    return results


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    out = Path(os.environ["OSMO_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "job.json").write_text(json.dumps(job, indent=1) + "\n")
    began = time.time()
    if job["kind"] == "rft_variants":
        results = rft_variants(job, out)
    elif job["kind"] == "clay_variants":
        results = clay_variants(job, out)
    else:
        raise SystemExit(f"unknown job kind {job['kind']}")
    failed = [name for name, record in results["variants"].items() if "error" in record]
    record = {"job": job, "failed_variants": failed, "wall_seconds": time.time() - began}
    (out / "result.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print("RESULT", json.dumps(record, default=str), flush=True)
    return 1 if len(failed) == len(results["variants"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
