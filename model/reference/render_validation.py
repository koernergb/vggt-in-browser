#!/usr/bin/env python3
"""Render a dependency-light M0 depth/point-cloud validation PNG from saved arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reference_utils import validate_fixture


def colorize(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Apply a compact blue→cyan→yellow→red map without matplotlib."""

    result = np.full((*values.shape, 3), 245, dtype=np.uint8)
    finite = values[valid]
    if not finite.size:
        return result
    low, high = np.percentile(finite, [2, 98])
    normalized = np.clip((values - low) / max(high - low, 1e-8), 0, 1)
    stops = np.array(
        [[35, 30, 120], [25, 120, 210], [30, 200, 170], [245, 220, 60], [210, 45, 35]],
        dtype=np.float32,
    )
    position = normalized * (len(stops) - 1)
    lower = np.floor(position).astype(np.int32)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (position - lower)[..., None]
    colors = stops[lower] * (1 - fraction) + stops[upper] * fraction
    result[valid] = colors[valid].astype(np.uint8)
    return result


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[..., :3]
    translation = extrinsic[..., 3]
    return -np.einsum("...ji,...j->...i", rotation, translation)


def remove_batch(array: np.ndarray) -> np.ndarray:
    return array[0] if array.ndim and array.shape[0] == 1 else array


def point_cloud_panel(
    points: np.ndarray, confidence: np.ndarray, extrinsic: np.ndarray, size: tuple[int, int]
) -> Image.Image:
    width, height = size
    points = points.reshape(-1, 3).astype(np.float64)
    confidence = confidence.reshape(-1).astype(np.float64)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    threshold = np.percentile(confidence[valid], 55)
    points = points[valid & (confidence >= threshold)]
    if len(points) > 120_000:
        indices = np.linspace(0, len(points) - 1, 120_000, dtype=np.int64)
        points = points[indices]

    center = np.median(points, axis=0)
    centered = points - center
    _, _, basis = np.linalg.svd(centered[:: max(1, len(centered) // 20_000)], full_matrices=False)
    projected = centered @ basis.T
    cameras = (camera_centers(extrinsic).reshape(-1, 3) - center) @ basis.T

    x_low, x_high = np.percentile(projected[:, 0], [1, 99])
    y_low, y_high = np.percentile(projected[:, 1], [1, 99])
    x_pad = max((x_high - x_low) * 0.08, 1e-6)
    y_pad = max((y_high - y_low) * 0.08, 1e-6)
    x_low, x_high = x_low - x_pad, x_high + x_pad
    y_low, y_high = y_low - y_pad, y_high + y_pad
    px = np.clip((projected[:, 0] - x_low) / (x_high - x_low) * (width - 1), 0, width - 1).astype(int)
    py = np.clip((1 - (projected[:, 1] - y_low) / (y_high - y_low)) * (height - 1), 0, height - 1).astype(int)
    z = projected[:, 2]
    z_valid = np.isfinite(z)
    colors = colorize(z.reshape(1, -1), z_valid.reshape(1, -1))[0]

    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    # Far-to-near assignment provides a simple point z-buffer.
    order = np.argsort(z)
    canvas[py[order], px[order]] = colors[order]
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for index, camera in enumerate(cameras):
        cx = int(np.clip((camera[0] - x_low) / (x_high - x_low) * (width - 1), 0, width - 1))
        cy = int(np.clip((1 - (camera[1] - y_low) / (y_high - y_low)) * (height - 1), 0, height - 1))
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(15, 15, 15), outline=(255, 255, 255), width=2)
        draw.text((cx + 9, cy - 8), f"C{index}", fill=(10, 10, 10))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrays", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path("bench/fixtures/manifest.json"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    _, image_paths = validate_fixture(args.fixture)
    arrays = np.load(args.arrays)
    depth = remove_batch(arrays["depth"]).squeeze(-1)
    depth_conf = remove_batch(arrays["depth_conf"])
    points = remove_batch(arrays["world_points"])
    point_conf = remove_batch(arrays["world_points_conf"])
    extrinsic = remove_batch(arrays["extrinsic"])

    width, height = 1600, 1100
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((40, 25), "VGGT M0 — MPS four-view validation", fill=(15, 30, 55), font=font)
    draw.text(
        (40, 48),
        "Top: ordered source views and predicted depth. Bottom: confidence-filtered point map (PCA view); C0–C3 are camera centers.",
        fill=(60, 70, 80),
        font=font,
    )

    tile_width = 360
    tile_height = 240
    gap = 20
    left = 40
    for index, image_path in enumerate(image_paths[: depth.shape[0]]):
        x = left + index * (tile_width + gap)
        source = Image.open(image_path).convert("RGB")
        source.thumbnail((tile_width, tile_height // 2 - 8))
        canvas.paste(source, (x, 88))
        valid = np.isfinite(depth[index]) & np.isfinite(depth_conf[index])
        depth_image = Image.fromarray(colorize(depth[index], valid)).resize(
            (tile_width, tile_height // 2), Image.Resampling.BILINEAR
        )
        canvas.paste(depth_image, (x, 88 + tile_height // 2))
        draw.text((x, 74), f"view {index}", fill=(20, 20, 20), font=font)

    panel = point_cloud_panel(points, point_conf, extrinsic, (1520, 700))
    canvas.paste(panel, (40, 360))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote validation image: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
