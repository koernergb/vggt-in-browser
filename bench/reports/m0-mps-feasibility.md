# M0 Apple MPS feasibility report

- Status: accepted local MPS baseline; M0 complete with cross-project CUDA evidence
- Date: 2026-08-19
- Classification: exploratory, not canonical CUDA reference

## Environment

- Device: 2024 MacBook Pro, Apple M4 (10-core GPU), 16 GB unified memory
- OS/kernel: Darwin 25.1.0, arm64
- PyTorch: 2.12.1
- torchvision: 0.27.1
- Backend: MPS
- Precision: FP16 autocast
- Model: `facebook/VGGT-1B`
- Upstream source: `a288dd0f14786c93483e45524328726ab7b1b4ce`
- Fixture: official kitchen views `00.png`–`03.png`
- Preprocessing: upstream crop mode, 518-pixel width

## Initial measurements

| Views | Run | Model load | Inference | Total | Outcome |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 1 | 19.18 s | 11.30 s | 31.96 s | Success |
| 2 | 2 | 24.96 s | 9.46 s | 36.72 s | Success |
| 4 | 1 | 19.77 s | 21.47 s | 45.30 s | Success |
| 4 | 2 | 16.76 s | 18.75 s | 37.81 s | Success |

Both two-view runs were byte-for-byte identical across pose encoding, extrinsics, intrinsics, depth, depth confidence, world points, and world-point confidence. Both four-view runs were also byte-for-byte identical across the same seven tensors.

## Clean-commit evidence

Commit `60f1a8743345a04ffc7c02895d93f5a1f23a58bb` was measured twice with an explicitly clean worktree:

| Views | Run | Model load | Inference | Total | Outcome |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | clean 1 | 27.02 s | 32.58 s | 63.92 s | Success |
| 4 | clean 2 | 19.50 s | 34.36 s | 57.12 s | Success |

The two clean runs were byte-for-byte identical across all seven recorded output tensors. The observed inference range is 32.58–34.36 seconds; earlier faster runs are retained as exploratory observations rather than substituted for the clean evidence.

## Interpretation

The 16 GB M4 is viable for local VGGT inference with four views. It is therefore suitable for preprocessing/postprocessing development, fixture iteration, visualization, and early export experiments.

This does not remove the CUDA reference requirement. The MPS environment uses a newer PyTorch release than upstream's pinned dependencies, and MPS/CUDA numeric behavior must be compared before a browser parity tolerance is frozen.

## Remaining validation

The repository owner reviewed the four-view validation image on 2026-08-20 and judged it plausible enough to accept as the local MPS baseline.

Remaining work:

1. Optional same-kitchen CUDA/Colab run if later ONNX drift is ambiguous.
2. Continue M1 using the accepted tolerances and evidence described in decision 0003.
