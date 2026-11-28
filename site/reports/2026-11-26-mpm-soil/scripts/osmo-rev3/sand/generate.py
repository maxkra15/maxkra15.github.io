# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate one OSMO workflow per job of the MPM sand and soil experiments (rod intrusion, plate sinkage).

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
    "common/vertical_intrusion.py",
    "rft_plate/__init__.py",
    "rft_plate/protocol.py",
    "rft_plate/backend.py",
    "rft_plate/run.py",
    "clay_plate/__init__.py",
    "clay_plate/protocol.py",
    "clay_plate/backend.py",
    "clay_plate/run.py",
    "rod_intrusion/__init__.py",
    "rod_intrusion/protocol.py",
    "rod_intrusion/backend.py",
    "rod_intrusion/run.py",
    "plate_sinkage/__init__.py",
    "plate_sinkage/protocol.py",
    "plate_sinkage/backend.py",
    "plate_sinkage/run.py",
    "common/vertical_pullout.py",
    "rod_pullout/__init__.py",
    "rod_pullout/protocol.py",
    "rod_pullout/backend.py",
    "rod_pullout/run.py",
    "data/douglas2026_pullout.json",
    "common/revolved_mesh.py",
    "common/probe_intrusion.py",
    "probe_penetration/__init__.py",
    "probe_penetration/protocol.py",
    "probe_penetration/backend.py",
    "probe_penetration/run.py",
    "data/mishra2018_probes.json",
    "data/__init__.py",
    "data/li2013_rft.json",
    "data/zwanenburg2024_clay.json",
    "data/ruck2025_intrusion.json",
    "data/kruger2025_cullinan.json",
]
JOBS = {}
for medium, tag in (("sand", "sand"), ("glass_beads", "glass"), ("lhs1", "lhs1")):
    JOBS[f"rod-{tag}-lit"] = {"kind": "rod_sweep", "medium": medium, "parameters": "literature", "hours": 6}
    JOBS[f"rod-{tag}-cal"] = {"kind": "rod_calibrate", "medium": medium, "hours": 16}
JOBS["rod-sand-checks"] = {
    "kind": "rod_variants", "medium": "sand", "parameters": "literature", "packings": [0.55, 0.59, 0.615], "hours": 10,
    "variants": [
        {"name": "base", "numeric": {}},
        {"name": "bed26", "numeric": {"bed_depth": 0.26}},
        {"name": "width15", "numeric": {"half_width": 0.15}},
        {"name": "speed2", "numeric": {"speed": 0.02}},
        {"name": "dt05", "numeric": {"dt": 0.0005}},
        {"name": "v2", "numeric": {"voxel_size": 0.002, "max_worlds": 4}},
    ],
}
JOBS["plate-lab"] = {"kind": "plate_sweep", "parameters": "lab", "hours": 8}
JOBS["plate-cal"] = {"kind": "plate_calibrate", "hours": 20}
JOBS["plate-checks"] = {
    "kind": "plate_variants", "parameters": {"friction": 0.567, "cohesion": 10000.0}, "methods": [4], "hours": 12,
    "variants": [
        {"name": "base", "numeric": {}},
        {"name": "speed50", "numeric": {"speed": 0.05}},
        {"name": "dt05", "numeric": {"dt": 0.0005}},
        {"name": "bed40", "numeric": {"bed_depth": 0.4}},
        {"name": "bed65", "numeric": {"bed_depth": 0.65}},
        {"name": "v75", "numeric": {"voxel_size": 0.0075, "max_worlds": 2}},
    ],
}
for method in (3, 4, 5):
    JOBS[f"plate-grid-m{method}"] = {
        "kind": "plate_variants", "parameters": "lab", "methods": [method], "hours": 10,
        "variants": [
            {"name": f"mu{mu:.2f}-c{c // 1000}", "numeric": {"speed": 0.05},
             "parameters": {"friction": mu, "cohesion": float(c)}}
            for mu in (0.55, 0.65, 0.75) for c in (0, 5000, 10000, 15000)
        ],
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
        f"  name: mpm-soil-sand-{name}-{DATE}",
        "  resources:",
        "    default:",
        f"      cpu: {max(8, 4 * job.get('gpus', 1))}",
        f"      gpu: {job.get('gpus', 1)}",
        f"      memory: {max(64, 32 * job.get('gpus', 1))}Gi",
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
