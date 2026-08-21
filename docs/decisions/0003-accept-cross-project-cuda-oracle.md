# 0003 — Accept the vggt-mlx CUDA oracle for M0

- Status: accepted
- Date: 2026-08-20

## Decision

Accept the CUDA T4 oracle fixtures and parity evidence in sibling project `vggt-mlx` as the independent CUDA evidence for M0. Mark M0 complete and treat the kitchen-specific Colab notebook as an optional regression tool rather than a blocking prerequisite.

Evidence was inspected at `vggt-mlx` commit `29281e4` on branch `agent/add-vggt-license`; the relevant implementation history is also present on its `main` branch at `4c5bd64` and earlier commits.

## Evidence

The sibling project contains official-PyTorch FP32 CUDA oracle fixtures for one-view and three-view inputs, with autocast disabled. Its parity suite checks:

- DINO patch tokens;
- aggregator layers 4, 11, 17, and 23;
- camera pose, extrinsics, and intrinsics;
- depth and depth confidence;
- full MLX forward outputs and finite-value requirements;
- strict conversion/loading of all 1,403 in-scope non-tracking checkpoint tensors.

Reported worst-case differences are `1.38e-4` for patch tokens, `1.41e-3` at aggregator layer 23, `3.88e-7` for camera pose, and `1.04e-5` for depth, with depth-confidence correlation greater than `0.99999999999`.

This repository separately produced two clean, byte-identical four-view PyTorch-MPS runs on the official kitchen fixture and received human visual acceptance.

## Limitation

The CUDA oracle uses different one-view and three-view fixtures from this repository's four-view kitchen MPS fixture. It establishes cross-runtime implementation parity, but is not an exact same-input MPS-versus-CUDA comparison.

The Colab workflow remains available if ONNX parity is ambiguous, if tolerances need tightening, or if a regression appears device-specific.

## Consequence

M0 is complete. M1 begins with a fixed-shape camera-plus-depth export. Point maps remain a subsequent export slice, while tracking is out of scope for the first browser graph.
