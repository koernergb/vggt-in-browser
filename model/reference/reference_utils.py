"""Shared deterministic fixture and tensor-summary utilities for M0."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class FixtureError(ValueError):
    """Raised when a fixture is incomplete or has drifted from its manifest."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FixtureError(f"Expected a JSON object in {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_fixture(manifest_path: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    errors: list[str] = []

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not str(manifest.get("scene_id", "")).strip():
        errors.append("scene_id is required")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    else:
        for field in ("source", "license", "permission_record"):
            if not str(provenance.get(field, "")).strip():
                errors.append(f"provenance.{field} is required")
        if provenance.get("contains_sensitive_content") is not False:
            errors.append("provenance.contains_sensitive_content must be false")

    images = manifest.get("images")
    if not isinstance(images, list) or not 2 <= len(images) <= 4:
        errors.append("images must contain 2 to 4 entries")
        images = []

    expected_orders = list(range(len(images)))
    actual_orders = [entry.get("order") for entry in images if isinstance(entry, dict)]
    if actual_orders != expected_orders:
        errors.append(f"image order must be exactly {expected_orders}, got {actual_orders}")

    resolved_paths: list[Path] = []
    for index, entry in enumerate(images):
        label = f"images[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be an object")
            continue
        relative = Path(str(entry.get("path", "")))
        if not relative.as_posix() or relative.is_absolute() or ".." in relative.parts:
            errors.append(f"{label}.path must be a safe path relative to the manifest")
            continue
        image_path = (manifest_path.parent / relative).resolve()
        resolved_paths.append(image_path)
        if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            errors.append(f"{label}.path has unsupported image suffix: {image_path.suffix}")
        if not image_path.is_file():
            errors.append(f"{label}.path does not exist: {image_path}")
            continue

        actual_hash = sha256_file(image_path)
        expected_hash = str(entry.get("sha256", ""))
        if actual_hash != expected_hash:
            errors.append(f"{label}.sha256 mismatch: expected {expected_hash}, got {actual_hash}")
        actual_bytes = image_path.stat().st_size
        if actual_bytes != entry.get("bytes"):
            errors.append(f"{label}.bytes mismatch: expected {entry.get('bytes')}, got {actual_bytes}")
        try:
            with Image.open(image_path) as image:
                actual_size = image.size
                image.verify()
        except Exception as exc:  # Pillow raises format-specific exceptions.
            errors.append(f"{label} is not a valid image: {exc}")
            continue
        expected_size = (entry.get("width"), entry.get("height"))
        if actual_size != expected_size:
            errors.append(f"{label} dimensions mismatch: expected {expected_size}, got {actual_size}")

    if len(set(resolved_paths)) != len(resolved_paths):
        errors.append("fixture contains duplicate image paths")
    if errors:
        raise FixtureError("Fixture validation failed:\n- " + "\n- ".join(errors))
    return manifest, resolved_paths


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def summarize_array(value: Any, sample_count: int = 16) -> dict[str, Any]:
    """Return a compact, deterministic summary for an array or torch tensor."""

    declared_dtype: str | None = None
    declared_bytes: int | None = None
    raw_bytes: bytes | None = None
    if hasattr(value, "detach"):
        import torch

        tensor = value.detach().cpu().contiguous()
        declared_dtype = str(tensor.dtype).removeprefix("torch.")
        declared_bytes = tensor.numel() * tensor.element_size()
        raw_bytes = tensor.view(-1).view(torch.uint8).numpy().tobytes()
        # NumPy cannot represent torch.bfloat16. Use FP32 for descriptive
        # statistics while hashing the original tensor bytes above.
        if declared_dtype == "bfloat16":
            tensor = tensor.float()
        value = tensor.numpy()
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    flat = contiguous.reshape(-1)
    numeric = np.issubdtype(contiguous.dtype, np.number)
    finite_mask = np.isfinite(flat) if numeric else np.ones(flat.shape, dtype=bool)
    finite_values = flat[finite_mask] if numeric else np.array([], dtype=np.float64)
    if flat.size:
        sample_indices = np.linspace(0, flat.size - 1, min(sample_count, flat.size), dtype=np.int64)
        samples = [_json_number(flat[i]) if numeric else str(flat[i]) for i in sample_indices]
    else:
        sample_indices = np.array([], dtype=np.int64)
        samples = []

    summary: dict[str, Any] = {
        "shape": list(contiguous.shape),
        "dtype": declared_dtype or str(contiguous.dtype),
        "elements": int(contiguous.size),
        "bytes": declared_bytes if declared_bytes is not None else int(contiguous.nbytes),
        "sha256": hashlib.sha256(raw_bytes or contiguous.tobytes(order="C")).hexdigest(),
        "sample_indices": sample_indices.tolist(),
        "samples": samples,
    }
    if numeric:
        summary.update(
            {
                "finite": int(finite_mask.sum()),
                "non_finite": int(flat.size - finite_mask.sum()),
                "min": _json_number(finite_values.min()) if finite_values.size else None,
                "max": _json_number(finite_values.max()) if finite_values.size else None,
                "mean": _json_number(finite_values.astype(np.float64).mean()) if finite_values.size else None,
                "std": _json_number(finite_values.astype(np.float64).std()) if finite_values.size else None,
            }
        )
    return summary


def flatten_tensor_outputs(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    """Yield stable names for tensors/arrays nested in mappings and sequences."""

    if hasattr(value, "detach") or isinstance(value, np.ndarray):
        yield prefix or "output", value
    elif isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_tensor_outputs(value[key], child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from flatten_tensor_outputs(item, child)
