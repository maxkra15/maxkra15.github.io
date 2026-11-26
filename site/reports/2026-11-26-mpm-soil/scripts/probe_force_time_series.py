# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record the plate force at every step for one path of each validation case.

Writes ``data/clay_force_series_test1.json`` (centrifuge test 1, laboratory SHANSEP strength, first
22 mm of penetration) and ``data/rft_force_series.json`` (plate intrusion at beta = 0,
gamma = 90 deg with literature parameters). Run with a CUDA GPU from a newton-calibration
checkout that contains ``examples/mpm_soil``::

    PYTHONPATH=<newton-calibration> python probe_force_time_series.py --out <report>/data
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from newton_calibration import Template


def _running_median(values: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    return np.median(sliding_window_view(np.pad(values, half, mode="edge"), width), axis=1)


def _dips(values: np.ndarray, position: np.ndarray, width: int, fraction: float, start: float):
    """Contiguous runs where ``values`` fall below ``fraction`` of the running median."""
    below = (values < fraction * _running_median(values, width)) & (position > start)
    index = np.flatnonzero(below)
    if index.size == 0:
        return []
    return np.split(index, np.flatnonzero(np.diff(index) > 1) + 1)


def clay_probe() -> dict:
    from examples.mpm_soil.clay_plate.backend import ClayAsset  # noqa: PLC0415
    from examples.mpm_soil.clay_plate.protocol import load_dataset  # noqa: PLC0415
    from examples.mpm_soil.clay_plate.run import NUMERIC_SETTINGS, _template, laboratory_parameters  # noqa: PLC0415

    dataset = load_dataset()
    values = laboratory_parameters(dataset)
    fixed = {"shansep_exponent": 0.8}
    depths = np.arange(5.0, 20.01, 2.5)
    template = _template(("test1",), numeric_settings=NUMERIC_SETTINGS, fixed=fixed).freeze()
    environment = ClayAsset(values, config={"profile": "shansep"}).create_batch_environment(
        template, [{**values, **fixed}], template.experiment_settings["test1"]
    )
    environment.run(depths, seed=0, deadline=None)
    penetration, pressure = (np.asarray(v) for v in environment.histories[0])
    groups = _dips(pressure, penetration, width=21, fraction=0.5, start=1.0)
    window = [np.abs(penetration - d) <= 0.5 for d in depths]
    mean = np.array([pressure[w].mean() for w in window])
    median = np.array([np.median(pressure[w]) for w in window])
    test = dataset["tests"]["test1"]
    measured = np.interp(depths, test["penetration_mm"], test["net_pressure_kPa"])
    clay_height_mm = test["clay_height"] * 1000.0
    spacing_mm = NUMERIC_SETTINGS["voxel_size"] * 1000.0 / 2
    layers = clay_height_mm - (0.5 * spacing_mm + spacing_mm * np.arange(int(round(clay_height_mm / spacing_mm))))
    layers = np.sort(layers[(layers > 0) & (layers < penetration.max() + 0.5)])
    return {
        "schema": "mpm-soil-report/clay-dips/1",
        "description": "Centrifuge test 1 with the laboratory SHANSEP strength: plate pressure at every 0.5 ms step",
        "numeric_settings": {
            "voxel_size_m": NUMERIC_SETTINGS["voxel_size"],
            "particles_per_cell_axis": 2,
            "dt_s": NUMERIC_SETTINGS["dt"],
            "speed_m_per_s": NUMERIC_SETTINGS["speed"],
            "strain_basis": NUMERIC_SETTINGS["strain_basis"],
            "collider_basis": "S2",
            "max_iterations": NUMERIC_SETTINGS["max_iterations"],
            "tolerance": NUMERIC_SETTINGS["tolerance"],
        },
        "penetration_mm": np.round(penetration, 4).tolist(),
        "pressure_kPa": np.round(pressure, 3).tolist(),
        "dip_starts_mm": [round(float(penetration[g[0]]), 3) for g in groups],
        "dip_widths_steps": [int(len(g)) for g in groups],
        "particle_layers_mm": np.round(layers, 4).tolist(),
        "depths_mm": depths.tolist(),
        "window_mean_kPa": np.round(mean, 3).tolist(),
        "window_median_kPa": np.round(median, 3).tolist(),
        "measured_kPa_interpolated": np.round(measured, 3).tolist(),
        "note": "Clay particles sit on an unjittered lattice (two per voxel axis). particle_layers_mm are the "
        "penetrations at which the plate underside reaches a layer of particle centres; dips are runs below half "
        "the 21-step running median.",
    }


def rft_probe() -> dict:
    from examples.mpm_soil.rft_plate.backend import RFTAsset  # noqa: PLC0415
    from examples.mpm_soil.rft_plate.protocol import load_dataset, plate_runs  # noqa: PLC0415
    from examples.mpm_soil.rft_plate.run import NUMERIC_SETTINGS, literature_parameters  # noqa: PLC0415

    medium = "poppy_seeds_LP"
    dataset = load_dataset()
    values = literature_parameters(dataset, medium)
    fixed = {"bulk_density": dataset["media"][medium]["bulk_density"], "gravity": 9.81}
    template = Template(
        numeric_settings=NUMERIC_SETTINGS,
        discretization={"particles_per_cell_axis": 2},
        fixed_parameters=fixed,
        experiment_settings={"probe": {"seed": 13, "medium": medium, "points": [[0.0, math.pi / 2]]}},
        qualification_mode="repeatability",
    ).freeze()
    environment = RFTAsset(values, config={"medium": medium}).create_batch_environment(
        template, [{**values, **fixed}], template.experiment_settings["probe"]
    )
    trace = environment.run(np.zeros(1), seed=0, deadline=None)[0]
    curve = environment.curves[0, 0][0]
    depth, force_z = np.asarray(curve["depth"]), np.asarray(curve["force"])[:, 2]
    run = plate_runs(0.0, math.pi / 2, dataset)[0]
    fit = (depth >= run.fit_depth[0]) & (depth <= run.fit_depth[1])
    voxel = float(NUMERIC_SETTINGS["voxel_size"])

    def slope(mask):
        return float(np.sum(force_z[mask] * depth[mask]) / np.sum(depth[mask] ** 2))

    k = slope(fit)
    ratio = force_z / (k * np.maximum(depth, 1e-9))
    phase = (depth % voxel) / voxel
    edges = np.linspace(0.0, 1.0, 13)
    which = np.clip(np.digitize(phase[fit], edges) - 1, 0, 11)
    profile = [float(ratio[fit][which == i].mean()) for i in range(12)]
    low = force_z < 0.85 * _running_median(force_z, 61)
    return {
        "schema": "mpm-soil-report/rft-force-series/2",
        "description": "Plate intrusion beta = 0, gamma = 90 deg with literature parameters: vertical plate force "
        "at every 1 ms step (0.1 mm of travel)",
        "numeric_settings": {
            "voxel_size_m": voxel,
            "particles_per_cell_axis": 2,
            "lattice_jitter": 0.25,
            "dt_s": NUMERIC_SETTINGS["dt"],
            "speed_m_per_s": NUMERIC_SETTINGS["speed"],
            "strain_basis": "P0",
            "collider_basis": "S2",
        },
        "depth_m": np.round(depth, 6).tolist(),
        "force_z_N": np.round(force_z, 5).tolist(),
        "alpha_z_N_per_cm3": float(np.asarray(trace.values)[0, 0]),
        "fit_window_m": list(run.fit_depth),
        "voxel_phase_profile": {
            "description": "mean of force / (fitted slope x depth) in 12 bins of the depth modulo one voxel, fit window only",
            "values": np.round(profile, 3).tolist(),
        },
        "low_phase": {
            "description": "samples below 85 % of the 61-step running median",
            "fraction_of_fit_window": round(float(low[fit].mean()), 3),
            "slope_change_when_excluded": round(slope(fit & ~low) / k - 1.0, 4),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name, record in (("clay_force_series_test1.json", clay_probe()), ("rft_force_series.json", rft_probe())):
        path = args.out / name
        path.write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
