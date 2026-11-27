# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate one OSMO workflow per job of the five-media MPM soil study.

Set NEWTON_CALIBRATION to a checkout with examples/mpm_soil and optionally OSMO_PLATFORM to an L40S platform of the
target pool (the runs of the report used L40S GPUs). Snapshots the example code into payload/examples, records
SHA-256 hashes, and writes jobs/<name>/job.json and jobs/<name>/workflow.yaml.
"""

import base64
import os
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get("NEWTON_CALIBRATION", "."))  # checkout with examples/mpm_soil
PLATFORM = os.environ.get("OSMO_PLATFORM")  # GPU platform of the target pool; the pool default if unset
PAYLOAD = ROOT  # this folder holds bootstrap.sh, task.py and the other payload files
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
CALIBRATED_POPPY_LP = {"friction": 0.5639441225542097, "plate_friction": 0.23046052848993623}
JOBS = {
    "rft-sweep-poppy-lp": {"kind": "rft_sweep", "medium": "poppy_seeds_LP", "parameters": "literature", "hours": 10},
    "rft-sweep-poppy-cp": {"kind": "rft_sweep", "medium": "poppy_seeds_CP", "parameters": "literature", "hours": 10},
    "rft-sweep-glass03-lp": {"kind": "rft_sweep", "medium": "glass_0p3mm_LP", "parameters": "literature", "hours": 10},
    "rft-sweep-glass03-cp": {"kind": "rft_sweep", "medium": "glass_0p3mm_CP", "parameters": "literature", "hours": 10},
    "rft-sweep-glass3-cp": {"kind": "rft_sweep", "medium": "glass_3mm_CP", "parameters": "literature", "hours": 10},
    "rft-cal-poppy-cp": {"kind": "rft_calibrate", "medium": "poppy_seeds_CP", "hours": 24},
    "rft-cal-glass03-lp": {"kind": "rft_calibrate", "medium": "glass_0p3mm_LP", "hours": 24},
    "rft-cal-glass03-cp": {"kind": "rft_calibrate", "medium": "glass_0p3mm_CP", "hours": 24},
    "rft-cal-glass3-cp": {"kind": "rft_calibrate", "medium": "glass_3mm_CP", "hours": 24},
    "rft-sweep-poppy-lp-dt05-lit": {
        "kind": "rft_sweep", "medium": "poppy_seeds_LP", "parameters": "literature", "dt": 0.0005, "hours": 20,
    },
    "rft-sweep-poppy-lp-dt05-cal": {
        "kind": "rft_sweep", "medium": "poppy_seeds_LP", "parameters": CALIBRATED_POPPY_LP, "dt": 0.0005, "hours": 20,
    },
    "rft-cal-poppy-cp-dt05": {"kind": "rft_calibrate", "medium": "poppy_seeds_CP", "dt": 0.0005, "hours": 24},
    "rft-sweep-poppy-cp-dt05": {
        "kind": "rft_sweep", "medium": "poppy_seeds_CP", "parameters": "literature", "dt": 0.0005, "hours": 12,
    },
    "clay-dt025-test1": {"kind": "clay_sweep", "tests": ["test1"], "dt": 0.00025, "hours": 8},
    "clay-dt025-test8": {"kind": "clay_sweep", "tests": ["test8"], "dt": 0.00025, "hours": 8},
    "clay-dt025-test6": {"kind": "clay_sweep", "tests": ["test6"], "dt": 0.00025, "hours": 8},
    "clay-dt025-test3": {"kind": "clay_sweep", "tests": ["test3"], "dt": 0.00025, "hours": 8},
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
        f"  name: mpm-soil-{name}-{DATE}",
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
