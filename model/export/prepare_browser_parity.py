#!/usr/bin/env python3
"""Prepare ignored raw float32 files for local browser parity testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from vggt.utils.load_fn import load_and_preprocess_images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.fixture.read_text(encoding="utf-8"))
    image_paths = [args.fixture.parent / entry["path"] for entry in manifest["images"][:2]]
    images = load_and_preprocess_images([str(path) for path in image_paths], mode="pad").numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    images.astype("<f4", copy=False).tofile(args.output / "images.f32")
    outputs = {}
    with np.load(args.reference) as arrays:
        for name in ("pose_enc", "depth", "depth_conf"):
            value = arrays[name].astype("<f4", copy=False)
            value.tofile(args.output / f"{name}.f32")
            outputs[name] = {"shape": list(value.shape), "file": f"{name}.f32"}
    parity_manifest = {
        "schema_version": 1,
        "preprocessing": "pad",
        "input": {"shape": [1, *images.shape], "file": "images.f32"},
        "outputs": outputs,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(parity_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote browser parity files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
