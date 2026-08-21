#!/usr/bin/env python3
"""Render FP32 and quantized depth predictions for human validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def colorize(values: np.ndarray, low: float, high: float) -> Image.Image:
    normalized = np.clip((values - low) / max(high - low, 1e-8), 0, 1)
    stops = np.array(
        [[35, 30, 120], [25, 120, 210], [30, 200, 170], [245, 220, 60], [210, 45, 35]],
        dtype=np.float32,
    )
    position = normalized * (len(stops) - 1)
    lower = np.floor(position).astype(np.int32)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (position - lower)[..., None]
    return Image.fromarray((stops[lower] * (1 - fraction) + stops[upper] * fraction).astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.reference) as ref_file, np.load(args.candidate) as candidate_file:
        reference = ref_file["depth"][0, :, :, :, 0]
        candidate = candidate_file["depth"][0, :, :, :, 0]
    low, high = np.percentile(reference, [2, 98])
    canvas = Image.new("RGB", (1554, 1120), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "VGGT depth: FP32 PyTorch vs aggregator INT8 ONNX Runtime", fill=(15, 20, 30))
    for view in range(2):
        y = 55 + view * 525
        panels = [
            ("FP32 reference", colorize(reference[view], low, high)),
            ("INT8 candidate", colorize(candidate[view], low, high)),
            ("absolute difference", colorize(np.abs(candidate[view] - reference[view]), 0, 0.25)),
        ]
        for column, (label, panel) in enumerate(panels):
            x = 24 + column * 510
            draw.text((x, y), f"view {view} — {label}", fill=(20, 20, 20))
            canvas.paste(panel.resize((486, 486), Image.Resampling.BILINEAR), (x, y + 22))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
