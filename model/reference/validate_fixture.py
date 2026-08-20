#!/usr/bin/env python3
"""Validate fixture provenance metadata, image order, dimensions, and hashes."""

from __future__ import annotations

import argparse
from pathlib import Path

from reference_utils import FixtureError, validate_fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        manifest, paths = validate_fixture(args.manifest)
    except FixtureError as exc:
        parser.error(str(exc))
    publication = manifest["provenance"].get("approved_for_public_repository", False)
    print(f"Fixture OK: {manifest['scene_id']} ({len(paths)} ordered images)")
    print(f"Approved for public repository: {str(bool(publication)).lower()}")
    for index, path in enumerate(paths):
        print(f"  {index}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

