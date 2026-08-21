#!/usr/bin/env python3
"""Create a deterministic hash manifest for an ONNX external-data bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(path for path in args.bundle.iterdir() if path.is_file())
    entries = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in files
    ]
    bundle_digest = hashlib.sha256()
    for entry in entries:
        bundle_digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode()
        )
    manifest = {
        "schema_version": 1,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "bundle_sha256": bundle_digest.hexdigest(),
        "files": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
