#!/usr/bin/env python3
"""Compare two compact M0 run summaries and characterize determinism."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reference_utils import load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def numeric_delta(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return abs(float(first) - float(second))


def main() -> int:
    args = parse_args()
    first = load_json(args.first)
    second = load_json(args.second)
    compatibility_fields = (
        ("fixture", "manifest_sha256"),
        ("upstream", "git_revision"),
        ("model", "id"),
        ("configuration", "dtype"),
        ("configuration", "preprocess_mode"),
        ("configuration", "view_count"),
    )
    mismatches = []
    for section, field in compatibility_fields:
        left = first.get(section, {}).get(field)
        right = second.get(section, {}).get(field)
        if left != right:
            mismatches.append({"field": f"{section}.{field}", "first": left, "second": right})
    if mismatches:
        raise ValueError(f"Runs are not comparable: {mismatches}")

    first_tensors = first.get("tensors", {})
    second_tensors = second.get("tensors", {})
    names = sorted(set(first_tensors) | set(second_tensors))
    tensors: dict[str, Any] = {}
    all_hashes_equal = True
    for name in names:
        left = first_tensors.get(name)
        right = second_tensors.get(name)
        if left is None or right is None:
            tensors[name] = {"present_in_both": False}
            all_hashes_equal = False
            continue
        hashes_equal = left.get("sha256") == right.get("sha256")
        all_hashes_equal &= hashes_equal
        tensors[name] = {
            "present_in_both": True,
            "shape_equal": left.get("shape") == right.get("shape"),
            "dtype_equal": left.get("dtype") == right.get("dtype"),
            "hash_equal": hashes_equal,
            "mean_abs_delta": numeric_delta(left.get("mean"), right.get("mean")),
            "std_abs_delta": numeric_delta(left.get("std"), right.get("std")),
            "min_abs_delta": numeric_delta(left.get("min"), right.get("min")),
            "max_abs_delta": numeric_delta(left.get("max"), right.get("max")),
            "samples_equal": left.get("samples") == right.get("samples"),
        }

    payload = {
        "schema_version": 1,
        "classification": "measured",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [str(args.first), str(args.second)],
        "exactly_deterministic": all_hashes_equal,
        "note": "Summary deltas characterize drift but do not replace elementwise comparison of saved arrays when exact hashes differ.",
        "tensors": tensors,
    }
    write_json(args.output, payload)
    print(f"Wrote determinism comparison: {args.output}")
    print(f"Exactly deterministic: {str(all_hashes_equal).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

