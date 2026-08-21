#!/usr/bin/env python3
"""Smoke-test the fixed-shape camera+depth graph in native ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--report", type=Path, default=Path("bench/results/m1-native-ort-smoke.json")
    )
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    load_seconds = time.perf_counter() - started

    images = np.zeros((1, 2, 3, 518, 518), dtype=np.float32)
    started = time.perf_counter()
    outputs = session.run(None, {"images": images})
    inference_seconds = time.perf_counter() - started

    summaries = {}
    for metadata, value in zip(session.get_outputs(), outputs, strict=True):
        summaries[metadata.name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(np.isfinite(value).all()),
            "min": float(value.min()),
            "max": float(value.max()),
            "mean": float(value.mean()),
        }

    report = {
        "schema_version": 1,
        "model": str(args.model),
        "provider": session.get_providers()[0],
        "input": "deterministic-all-zero-smoke-input",
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "outputs": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
