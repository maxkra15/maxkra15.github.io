# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Download the pinned source archives, verify their SHA-256 and unpack them into src/<name>."""
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

root = Path(sys.argv[1])
sources = json.loads((Path(__file__).resolve().parent / "sources.json").read_text())
(root / "src").mkdir(parents=True, exist_ok=True)
for name, spec in sources.items():
    archive = root / f"{name}.tar.gz"
    with urllib.request.urlopen(spec["url"], timeout=300) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise SystemExit(f"checksum mismatch for {name}: {digest}")
    unpack = root / f"unpack-{name}"
    with tarfile.open(archive) as tar:
        tar.extractall(unpack, filter="data")
    (child,) = list(unpack.iterdir())
    child.rename(root / "src" / name)
    unpack.rmdir()
    print("SOURCE_READY", name, digest, flush=True)
