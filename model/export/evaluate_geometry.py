#!/usr/bin/env python3
"""Measure task-level camera, geometry, and confidence drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def cameras(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            torch.from_numpy(pose), (518, 518)
        )
    return extrinsic.numpy(), intrinsic.numpy()


def centers(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[..., :3]
    translation = extrinsic[..., 3]
    return -np.einsum("...ji,...j->...i", rotation, translation)


def rotation_degrees(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ji,...jk->...ik", reference, candidate)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values.reshape(-1), kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return (ranks / max(order.size - 1, 1)).reshape(values.shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.reference) as reference_file, np.load(args.candidate) as candidate_file:
        ref = {name: reference_file[name] for name in ("pose_enc", "depth", "depth_conf")}
        cand = {name: candidate_file[name] for name in ("pose_enc", "depth", "depth_conf")}

    ref_extri, ref_intri = cameras(ref["pose_enc"])
    cand_extri, cand_intri = cameras(cand["pose_enc"])
    ref_centers, cand_centers = centers(ref_extri), centers(cand_extri)
    baseline = np.linalg.norm(ref_centers[:, 1] - ref_centers[:, 0], axis=-1)
    center_error = np.linalg.norm(cand_centers - ref_centers, axis=-1)
    angles = rotation_degrees(ref_extri[..., :3], cand_extri[..., :3])
    focal_relative = np.abs(
        cand_intri[..., (0, 1), (0, 1)] / ref_intri[..., (0, 1), (0, 1)] - 1
    )

    ref_points = unproject_depth_map_to_point_map(ref["depth"][0], ref_extri[0], ref_intri[0])
    cand_points = unproject_depth_map_to_point_map(cand["depth"][0], cand_extri[0], cand_intri[0])
    point_error = np.linalg.norm(cand_points - ref_points, axis=-1)
    scene_scale = float(np.median(np.linalg.norm(ref_points, axis=-1)))

    ref_rank = rank_percentiles(ref["depth_conf"])
    cand_rank = rank_percentiles(cand["depth_conf"])
    rank_correlation = float(np.corrcoef(ref_rank.reshape(-1), cand_rank.reshape(-1))[0, 1])
    ref_keep, cand_keep = ref_rank >= 0.55, cand_rank >= 0.55
    retained_intersection_over_union = float(
        np.logical_and(ref_keep, cand_keep).sum() / np.logical_or(ref_keep, cand_keep).sum()
    )

    report = {
        "schema_version": 1,
        "camera": {
            "rotation_error_degrees_per_view": angles.reshape(-1).tolist(),
            "rotation_error_degrees_max": float(angles.max()),
            "center_error_per_view": center_error.reshape(-1).tolist(),
            "reference_camera_baseline": float(baseline[0]),
            "center_error_over_baseline_max": float(center_error.max() / max(baseline[0], 1e-8)),
            "focal_relative_error_max": float(focal_relative.max()),
        },
        "geometry": {
            "corresponding_point_error_mean": float(point_error.mean()),
            "corresponding_point_error_p95": float(np.percentile(point_error, 95)),
            "reference_scene_scale_median_radius": scene_scale,
            "point_error_mean_over_scene_scale": float(point_error.mean() / scene_scale),
        },
        "confidence": {
            "rank_correlation": rank_correlation,
            "top_45_percent_mask_iou": retained_intersection_over_union,
            "policy": "rank/percentile filtering; do not reuse FP32 absolute thresholds",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
