#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT=/workspace/mpmsoil
mkdir -p "$ROOT/tmp" "$ROOT/cache/uv" "$ROOT/cache/warp" "$OSMO_OUTPUT"
export TMPDIR="$ROOT/tmp" UV_CACHE_DIR="$ROOT/cache/uv" WARP_CACHE_PATH="$ROOT/cache/warp" XDG_CACHE_HOME="$ROOT/cache"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
exec > >(tee -a "$OSMO_OUTPUT/task.log") 2>&1
finish() { code=$?; echo "$code" > "$OSMO_OUTPUT/exit-code.txt"; echo "EXIT_CODE $code"; }
trap finish EXIT
cp -a /tmp/mpmsoil/. "$ROOT/payload/"
cd "$ROOT"
base64 -d payload/code.tar.gz.b64 | tar -xz -C payload
python3 - <<'PY'
import hashlib, json
from pathlib import Path
expected = json.loads(Path("payload/payload-sha256.json").read_text())
for relative, digest in expected.items():
    actual = hashlib.sha256(Path("payload", relative).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"payload mismatch {relative}")
print("PAYLOAD_VERIFIED", len(expected), "files")
PY
date -u
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv venv --python 3.12 venv
uv pip install --python venv/bin/python -r payload/requirements.txt \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
venv/bin/python payload/fetch_sources.py "$ROOT"
cp -a payload/examples/mpm_soil src/newton-calibration/examples/
uv pip freeze --python venv/bin/python > "$OSMO_OUTPUT/dependencies.txt"
cd src/newton-calibration
export PYTHONPATH="$ROOT/src/newton-calibration:$ROOT/src/newton"
"$ROOT/venv/bin/python" -c "import warp, newton, newton_calibration; print('IMPORTS_OK', warp.__version__, newton.__version__)"
"$ROOT/venv/bin/python" "$ROOT/payload/task.py" "$ROOT/payload/job.json"
