# 0001 — Reference baseline

- Status: accepted for fixture/checkpoint selection; blocked on CUDA execution and visual validation
- Date: 2026-08-19

## Context

Browser conversion needs a trustworthy native reference. Visual plausibility alone is insufficient: later graphs must be compared at tensor and task levels using identical ordered inputs and preprocessing.

The upstream VGGT project has continued to change, including memory behavior and checkpoint licensing. A floating dependency on its default branch would make results difficult to reproduce.

## Decision

Pin upstream VGGT revision `a288dd0f14786c93483e45524328726ab7b1b4ce` for the first baseline. Use Python 3.10, PyTorch 2.3.1, torchvision 0.18.1, upstream crop preprocessing, batch size 1, and the camera, depth, and point-map heads. Tracks are excluded from the first baseline because they require query-point policy and additional work that is not needed to validate MVP geometry.

Run on CUDA with BF16 on compute capability 8+ and FP16 otherwise, matching the upstream recommendation. Record the actual dtype and full device environment rather than assuming it.

Use official kitchen views `00.png` through `03.png` from the pinned upstream repository as the ordered four-image golden fixture. Fetch and hash them locally; do not duplicate them in this repository. Preserve compact tensor summaries and hashes by default; do not commit model weights or raw full-resolution tensors.

The repository owner confirmed the project is strictly non-commercial, so the original `facebook/VGGT-1B` checkpoint under CC BY-NC 4.0 is the approved M0 reference.

## Rationale

Camera plus depth/point maps are the minimum outputs needed to validate the intended interactive geometry demo. Pinning the upstream revision and preprocessing eliminates major sources of accidental drift. Compact summaries are sufficient for smoke tests and first-pass parity while avoiding large repository artifacts.

## Human decisions still required

1. Provide a representative CUDA run if the agent environment lacks appropriate hardware.
2. Visually accept the camera/depth/point-map reconstruction before marking M0 complete.

An Apple Silicon MPS run is approved as an exploratory feasibility check, beginning with two views in FP16. It does not replace the canonical CUDA evidence unless a later decision explicitly changes the reference definition.

## Consequences

- The initial baseline does not validate tracking.
- Different upstream revisions or preprocessing modes must create a new result series.
- A checkpoint change invalidates golden hashes and requires rerunning M0.
- Quantized checkpoints are derived experiments and must be compared against this full-precision reference.
- Browser work must not claim parity until it is compared to this accepted baseline.
