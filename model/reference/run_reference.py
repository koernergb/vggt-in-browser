#!/usr/bin/env python3
"""Run pinned VGGT inference and emit a compact, reproducible M0 summary."""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

import numpy as np

from reference_utils import (
    flatten_tensor_outputs,
    sha256_file,
    summarize_array,
    validate_fixture,
    write_json,
)


PINNED_UPSTREAM_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument("--checkpoint", type=Path, help="Optional local state_dict checkpoint")
    parser.add_argument("--preprocess-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-views",
        type=int,
        choices=(2, 3, 4),
        help="Use the first N ordered fixture views for a constrained exploratory run",
    )
    parser.add_argument("--disable-point-head", action="store_true")
    parser.add_argument("--save-arrays", type=Path, help="Optional local NPZ path; ignored by Git")
    parser.add_argument("--allow-unpinned-upstream", action="store_true")
    return parser.parse_args()


def git_value(arguments: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def choose_dtype(torch: Any, device: str, requested: str) -> Any:
    if requested != "auto":
        return getattr(torch, requested)
    if device.startswith("cuda"):
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def environment_metadata(torch: Any, device: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": None,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_requested": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    try:
        import torchvision

        metadata["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device() if device == "cuda" else torch.device(device).index
        props = torch.cuda.get_device_properties(index)
        metadata["gpu"] = {
            "name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": int(props.total_memory),
        }
    return metadata


def assert_upstream_revision(allow_unpinned: bool) -> dict[str, Any]:
    import vggt

    package_file = getattr(vggt, "__file__", None)
    package_paths = list(getattr(vggt, "__path__", []))
    package_path = Path(package_file or package_paths[0]).resolve()
    direct_url = None
    try:
        direct_url_text = distribution("vggt").read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text else None
    except (PackageNotFoundError, json.JSONDecodeError):
        pass
    actual = (direct_url or {}).get("vcs_info", {}).get("commit_id")
    dirty = None
    if actual is None:
        repository = package_path
        while repository.parent != repository and not (repository / ".git").exists():
            repository = repository.parent
        actual = (
            git_value(["rev-parse", "HEAD"], repository)
            if (repository / ".git").exists()
            else None
        )
        dirty = git_value(["status", "--porcelain"], repository) if actual else None
    if not allow_unpinned and actual != PINNED_UPSTREAM_REVISION:
        raise RuntimeError(
            "Installed VGGT is not the pinned source checkout. "
            f"Expected {PINNED_UPSTREAM_REVISION}, found {actual or 'unknown'}. "
            "Install the pinned editable checkout or pass --allow-unpinned-upstream for an explicitly non-golden run."
        )
    return {
        "package_path": str(package_path),
        "direct_url": direct_url,
        "git_revision": actual,
        "git_dirty": bool(dirty) if dirty is not None else None,
        "pinned_revision": PINNED_UPSTREAM_REVISION,
        "pinned_match": actual == PINNED_UPSTREAM_REVISION,
    }


def synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device == "mps":
        torch.mps.synchronize()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    manifest, image_paths = validate_fixture(args.fixture)
    if args.max_views:
        image_paths = image_paths[: args.max_views]

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; do not substitute CPU output as the golden CUDA reference")
    if args.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    upstream = assert_upstream_revision(args.allow_unpinned_upstream)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dtype = choose_dtype(torch, args.device, args.dtype)
    load_start = time.perf_counter()
    if args.checkpoint:
        checkpoint_path = args.checkpoint.resolve()
        model = VGGT(enable_track=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        checkpoint = {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)}
    else:
        model = VGGT.from_pretrained(args.model_id)
        checkpoint = {"path": None, "sha256": None}
    if args.disable_point_head:
        model.point_head = None
    model.eval().to(args.device)
    synchronize(torch, args.device)
    load_seconds = time.perf_counter() - load_start

    preprocess_start = time.perf_counter()
    images = load_and_preprocess_images(
        [str(path) for path in image_paths], mode=args.preprocess_mode
    ).to(args.device)
    synchronize(torch, args.device)
    preprocess_seconds = time.perf_counter() - preprocess_start

    inference_start = time.perf_counter()
    autocast = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if args.device.startswith("cuda") and dtype != torch.float32
        else torch.autocast(device_type="mps", dtype=dtype)
        if args.device == "mps" and dtype != torch.float32
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        predictions = model(images)
    synchronize(torch, args.device)
    inference_seconds = time.perf_counter() - inference_start

    postprocess_start = time.perf_counter()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    selected = {
        key: value
        for key, value in predictions.items()
        if key in {"pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"}
    }
    selected["extrinsic"] = extrinsic
    selected["intrinsic"] = intrinsic
    tensors = {name: summarize_array(value) for name, value in flatten_tensor_outputs(selected)}
    input_summary = summarize_array(images)
    postprocess_seconds = time.perf_counter() - postprocess_start

    if any(summary.get("non_finite", 0) for summary in tensors.values()):
        bad = [name for name, summary in tensors.items() if summary.get("non_finite", 0)]
        raise RuntimeError(f"Non-finite values found in outputs: {', '.join(bad)}")

    if args.save_arrays:
        arrays = {}
        for name, value in flatten_tensor_outputs(selected):
            tensor = value.detach().cpu()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            arrays[name] = tensor.numpy()
        args.save_arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_arrays, **arrays)

    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": 1,
        "classification": "measured",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "revision": git_value(["rev-parse", "HEAD"], repo_root),
            "dirty": bool(git_value(["status", "--porcelain"], repo_root)),
        },
        "upstream": upstream,
        "model": {
            "id": args.model_id,
            "checkpoint": checkpoint,
            "heads": ["camera", "depth"] + ([] if args.disable_point_head else ["point"]),
        },
        "configuration": {
            "seed": args.seed,
            "device": args.device,
            "dtype": str(dtype).removeprefix("torch."),
            "preprocess_mode": args.preprocess_mode,
            "batch_size": 1,
            "view_count": len(image_paths),
        },
        "environment": environment_metadata(torch, args.device),
        "fixture": {
            "scene_id": manifest["scene_id"],
            "manifest_sha256": sha256_file(args.fixture.resolve()),
            "images": [
                {
                    "order": entry["order"],
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "width": entry["width"],
                    "height": entry["height"],
                    "bytes": entry["bytes"],
                }
                for entry in manifest["images"][: len(image_paths)]
            ],
        },
        "input": input_summary,
        "tensors": tensors,
        "timings_seconds": {
            "model_load": load_seconds,
            "preprocessing": preprocess_seconds,
            "inference": inference_seconds,
            "postprocessing_and_summary": postprocess_seconds,
            "total": time.perf_counter() - started,
        },
    }
    write_json(args.output, result)
    print(f"Wrote reference summary: {args.output}")
    print(json.dumps(result["timings_seconds"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
