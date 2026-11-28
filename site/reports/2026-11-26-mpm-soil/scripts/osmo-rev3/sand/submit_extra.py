# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Write and submit extra jobs with the current payload: python submit_extra.py <jobs.json>."""
import json
import os
import subprocess
import sys
from pathlib import Path

import generate

for name, job in json.loads(Path(sys.argv[1]).read_text()).items():
    folder = generate.ROOT / "jobs" / name
    folder.mkdir(parents=True, exist_ok=True)
    spec = {k: v for k, v in job.items() if k != "hours"} | {"name": name}
    (folder / "job.json").write_text(json.dumps(spec, indent=1) + "\n")
    (folder / "workflow.yaml").write_text(generate.workflow_yaml(name, job, folder / "job.json"))
    out = subprocess.run(["osmo", "workflow", "submit", str(folder / "workflow.yaml"), "--pool", os.environ["OSMO_POOL"],
                          "-t", "json"], capture_output=True, text=True).stdout
    (generate.ROOT / "submissions" / f"{name}.json").write_text(out)
    print(json.loads(out).get("name"))
