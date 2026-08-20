#!/usr/bin/env python3
"""Fetch the pinned official VGGT kitchen fixture and verify every file."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from reference_utils import FixtureError, load_json, sha256_file, validate_fixture


REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
RAW_ROOT = (
    f"https://raw.githubusercontent.com/facebookresearch/vggt/{REVISION}"
    "/examples/kitchen/images"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "bench/fixtures/manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true", help="Replace a mismatched local image")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    for entry in manifest.get("images", []):
        destination = (args.manifest.resolve().parent / entry["path"]).resolve()
        expected = entry["sha256"]
        if destination.exists() and sha256_file(destination) == expected:
            print(f"Already verified: {destination}")
            continue
        if destination.exists() and not args.force:
            raise FixtureError(
                f"Refusing to replace mismatched file {destination}; inspect it or pass --force"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        url = f"{RAW_ROOT}/{destination.name}"
        print(f"Fetching {url}")
        urllib.request.urlretrieve(url, temporary)
        actual = sha256_file(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise FixtureError(
                f"Downloaded hash mismatch for {destination.name}: expected {expected}, got {actual}"
            )
        temporary.replace(destination)

    _, paths = validate_fixture(args.manifest)
    print(f"Fixture ready: {len(paths)} verified images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
