#!/usr/bin/env python3
"""Render FP32 and INT8 unprojected two-view geometry with shared framing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def derive(arrays: np.lib.npyio.NpzFile):
    pose = torch.from_numpy(arrays["pose_enc"])
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose, (518, 518))
    points = unproject_depth_map_to_point_map(
        arrays["depth"][0], extrinsic.numpy()[0], intrinsic.numpy()[0]
    )
    confidence = arrays["depth_conf"][0]
    centers = -np.einsum(
        "...ji,...j->...i", extrinsic.numpy()[0, :, :, :3], extrinsic.numpy()[0, :, :, 3]
    )
    return points, confidence, centers


def selected(points: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    threshold = np.percentile(confidence, 55)
    xyz, view = [], []
    for index in range(points.shape[0]):
        chosen = points[index][confidence[index] >= threshold]
        stride = max(1, len(chosen) // 70_000)
        xyz.append(chosen[::stride])
        view.append(np.full(len(chosen[::stride]), index))
    return np.concatenate(xyz), np.concatenate(view)


def panel(
    points: np.ndarray,
    view: np.ndarray,
    centers: np.ndarray,
    basis: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> Image.Image:
    width, height = 740, 700
    projected = points @ basis.T
    camera_projected = centers @ basis.T
    x0, x1, y0, y1 = bounds
    x = np.clip((projected[:, 0] - x0) / (x1 - x0) * (width - 1), 0, width - 1).astype(int)
    y = np.clip((1 - (projected[:, 1] - y0) / (y1 - y0)) * (height - 1), 0, height - 1).astype(int)
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    colors = np.array([[20, 135, 210], [235, 105, 35]], dtype=np.uint8)
    for order in (0, 1):
        mask = view == order
        canvas[y[mask], x[mask]] = colors[order]
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for index, camera in enumerate(camera_projected):
        cx = int(np.clip((camera[0] - x0) / (x1 - x0) * (width - 1), 0, width - 1))
        cy = int(np.clip((1 - (camera[1] - y0) / (y1 - y0)) * (height - 1), 0, height - 1))
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(10, 10, 10))
        draw.text((cx + 9, cy - 8), f"C{index}", fill=(10, 10, 10))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.reference) as ref_file, np.load(args.candidate) as cand_file:
        ref_points, ref_conf, ref_centers = derive(ref_file)
        cand_points, cand_conf, cand_centers = derive(cand_file)
    ref_xyz, ref_view = selected(ref_points, ref_conf)
    cand_xyz, cand_view = selected(cand_points, cand_conf)
    center = np.median(ref_xyz, axis=0)
    _, _, basis = np.linalg.svd(ref_xyz[:: max(1, len(ref_xyz) // 20_000)] - center, full_matrices=False)
    ref_xyz, cand_xyz = ref_xyz - center, cand_xyz - center
    ref_centers, cand_centers = ref_centers - center, cand_centers - center
    projected = ref_xyz @ basis.T
    x0, x1 = np.percentile(projected[:, 0], [1, 99])
    y0, y1 = np.percentile(projected[:, 1], [1, 99])
    bounds = (x0, x1, y0, y1)
    canvas = Image.new("RGB", (1520, 770), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), "Unprojected two-view geometry — blue=view 0, orange=view 1", fill=(15, 20, 30))
    draw.text((20, 40), "FP32 reference", fill=(15, 20, 30))
    draw.text((780, 40), "Aggregator INT8 candidate", fill=(15, 20, 30))
    canvas.paste(panel(ref_xyz, ref_view, ref_centers, basis, bounds), (20, 65))
    canvas.paste(panel(cand_xyz, cand_view, cand_centers, basis, bounds), (780, 65))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
