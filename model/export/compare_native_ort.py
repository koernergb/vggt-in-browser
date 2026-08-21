#!/usr/bin/env python3
"""Compare native ONNX Runtime outputs with saved PyTorch reference arrays."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from vggt.utils.load_fn import load_and_preprocess_images


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    scale = np.maximum(np.abs(expected.astype(np.float64)), 1e-6)
    return {
        "shape_match": actual.shape == expected.shape,
        "finite": bool(np.isfinite(actual).all()),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "max_relative_at_1e-6_floor": float((difference / scale).max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.fixture.read_text(encoding="utf-8"))
    fixture_root = args.fixture.parent
    image_paths = [fixture_root / entry["path"] for entry in manifest["images"][:2]]
    images = load_and_preprocess_images([str(path) for path in image_paths], mode="pad")
    input_array = images.unsqueeze(0).numpy() if images.ndim == 4 else images.numpy()

    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    actual_values = session.run(None, {"images": input_array})
    inference_seconds = time.perf_counter() - started

    with np.load(args.reference) as expected_values:
        comparisons = {
            output.name: metrics(actual, expected_values[output.name])
            for output, actual in zip(session.get_outputs(), actual_values, strict=True)
        }
    report = {
        "schema_version": 1,
        "model": str(args.model),
        "fixture": str(args.fixture),
        "reference": str(args.reference),
        "preprocessing": "pad",
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "comparisons": comparisons,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
