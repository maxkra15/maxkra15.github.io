# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate one OSMO workflow per job of the MPM soil resolution study.

Snapshots the example code from the newton-calibration branch into payload/examples, records
SHA-256 hashes, and writes jobs/<name>/job.json and jobs/<name>/workflow.yaml.

Granular: 11 diagnostic plate motions (the five training motions and the worst held-out ones of
the five media) for literature and calibrated parameters, at 6, 4 and 3 mm voxels, with 27
instead of 8 particles per voxel, with P1d strain, with the pic27 collider basis and with a second
particle arrangement. Clay: tests 1 and 3 at 2.5 and 1.25 mm voxels and with the pic27 collider
basis, each at 0.5 and 0.25 ms.
"""

import base64
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get("NEWTON_CALIBRATION", "."))  # checkout with examples/mpm_soil
PLATFORM = os.environ.get("OSMO_PLATFORM")  # GPU platform of the target pool; the pool default if unset
PAYLOAD = ROOT / "payload"  # bootstrap.sh, fetch_sources.py, sources.json, requirements.txt as in ../osmo/; task.py from here
EXAMPLE_FILES = [
    "__init__.py",
    "verify.py",
    "common/__init__.py",
    "common/mpm.py",
    "common/presentation.py",
    "rft_plate/__init__.py",
    "rft_plate/protocol.py",
    "rft_plate/backend.py",
    "rft_plate/run.py",
    "clay_plate/__init__.py",
    "clay_plate/protocol.py",
    "clay_plate/backend.py",
    "clay_plate/run.py",
    "data/__init__.py",
    "data/li2013_rft.json",
    "data/zwanenburg2024_clay.json",
]
# (beta, gamma) in degrees: the five training motions, the worst held-out motions of the five
# media and the face-on drag.
POINTS_DEG = [
    [0, 90], [30, 45], [-30, 45], [90, 90], [60, -45],
    [0, 18.435], [30, 18.435], [30, -18.435], [60, 45], [-60, 71.565], [90, 0],
]
MEDIA = {
    "poppy-lp": ("poppy_seeds_LP", 0.001, {"friction": 0.5639441225542097, "plate_friction": 0.23046052848993623}),
    "poppy-cp": ("poppy_seeds_CP", 0.0005, {"friction": 0.6241421938762877, "plate_friction": 0.3727581084021456}),
    "glass03-lp": ("glass_0p3mm_LP", 0.001, {"friction": 0.39612379471270104, "plate_friction": 0.22904492490535697}),
    "glass03-cp": ("glass_0p3mm_CP", 0.001, {"friction": 0.46627578770596734, "plate_friction": 0.18084985100634457}),
    "glass3-cp": ("glass_3mm_CP", 0.001, {"friction": 0.3888456753703826, "plate_friction": 0.18274503679485926}),
}
COARSE_VARIANTS = [
    {"name": "v6", "numeric": {}},
    {"name": "v6-seed14", "numeric": {}, "seed": 14},
    {"name": "v6-pic27", "numeric": {"collider_basis": "pic27"}},
    {"name": "v6-p1d", "numeric": {"strain_basis": "P1d"}},
    {"name": "v6-ppc3", "numeric": {"max_worlds": 8}, "particles_per_cell_axis": 3},
    {"name": "v4", "numeric": {"voxel_size": 0.004, "max_worlds": 8}},
]
FINE_VARIANTS = [{"name": "v3", "numeric": {"voxel_size": 0.003, "max_worlds": 4}}]
JOBS = {}
for tag, (medium, dt, calibrated) in MEDIA.items():
    for suffix, variants, hours in (("coarse", COARSE_VARIANTS, 10), ("v3", FINE_VARIANTS, 12)):
        JOBS[f"rft-{tag}-{suffix}"] = {
            "kind": "rft_variants",
            "medium": medium,
            "dt": dt,
            "points_deg": POINTS_DEG,
            "parameter_sets": {"literature": "literature", "calibrated": calibrated},
            "variants": variants,
            "hours": hours,
        }
JOBS["clay-v25"] = {
    "kind": "clay_variants",
    "tests": ["test1", "test3"],
    "variants": [
        {"name": "v25-dt05", "numeric": {}},
        {"name": "v25-pic27-dt05", "numeric": {"collider_basis": "pic27"}},
        {"name": "v25-pic27-dt025", "numeric": {"collider_basis": "pic27", "dt": 0.00025}},
    ],
    "hours": 10,
}
JOBS["clay-v125-dt05"] = {
    "kind": "clay_variants",
    "tests": ["test1", "test3"],
    "variants": [{"name": "v125-dt05", "numeric": {"voxel_size": 0.00125}}],
    "hours": 12,
}
for test in ("test1", "test3"):
    JOBS[f"clay-v125-dt025-{test}"] = {
        "kind": "clay_variants",
        "tests": [test],
        "variants": [{"name": "v125-dt025", "numeric": {"voxel_size": 0.00125, "dt": 0.00025}}],
        "hours": 14,
    }
DATE = "20261127"


def snapshot() -> dict:
    target = PAYLOAD / "examples" / "mpm_soil"
    if target.exists():
        shutil.rmtree(target)
    hashes = {}
    for relative in EXAMPLE_FILES:
        source = REPO / "examples" / "mpm_soil" / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[f"examples/mpm_soil/{relative}"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (PAYLOAD / "payload-sha256.json").write_text(json.dumps(hashes, indent=1) + "\n")
    # OSMO renders the whole spec, including inlined files, with Jinja; base64 carries no braces.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(PAYLOAD / "examples", arcname="examples")
    (PAYLOAD / "code.tar.gz.b64").write_text(base64.encodebytes(buffer.getvalue()).decode())
    return hashes


def workflow_yaml(name: str, job: dict, job_path: Path) -> str:
    files = [
        ("bootstrap.sh", PAYLOAD / "bootstrap.sh"),
        ("task.py", PAYLOAD / "task.py"),
        ("fetch_sources.py", PAYLOAD / "fetch_sources.py"),
        ("sources.json", PAYLOAD / "sources.json"),
        ("requirements.txt", PAYLOAD / "requirements.txt"),
        ("payload-sha256.json", PAYLOAD / "payload-sha256.json"),
        ("job.json", job_path),
        ("code.tar.gz.b64", PAYLOAD / "code.tar.gz.b64"),
    ]
    lines = [
        "version: 2",
        "workflow:",
        f"  name: mpm-soil-res-{name}-{DATE}",
        "  resources:",
        "    default:",
        "      cpu: 8",
        "      gpu: 1",
        "      memory: 64Gi",
        "      storage: 64Gi",
        *([f"      platform: {PLATFORM}"] if PLATFORM else []),
        "  timeout:",
        f"    exec_timeout: {job['hours']}h",
        "    queue_timeout: 6h",
        "  tasks:",
        "  - name: run",
        "    image: ghcr.io/astral-sh/uv:python3.12-bookworm",
        "    command:",
        "    - bash",
        "    - /tmp/mpmsoil/bootstrap.sh",
        "    environment:",
        "      OSMO_OUTPUT: '{{output}}'",
        "      PYTHONUNBUFFERED: '1'",
        "      PYTHONNOUSERSITE: '1'",
        "      OMP_NUM_THREADS: '4'",
        "    files:",
    ]
    for path, local in files:
        lines += [f"    - path: /tmp/mpmsoil/{path}", f"      localpath: {local}"]
    return "\n".join(lines) + "\n"


def main():
    hashes = snapshot()
    for name, job in JOBS.items():
        folder = ROOT / "jobs" / name
        folder.mkdir(parents=True, exist_ok=True)
        spec = {k: v for k, v in job.items() if k != "hours"}
        spec["name"] = name
        (folder / "job.json").write_text(json.dumps(spec, indent=1) + "\n")
        (folder / "workflow.yaml").write_text(workflow_yaml(name, job, folder / "job.json"))
    print(f"{len(JOBS)} jobs, {len(hashes)} example files snapshotted")


if __name__ == "__main__":
    main()
