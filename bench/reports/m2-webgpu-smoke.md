# M2 WebGPU smoke report

Status: complete. Measured WebGPU execution and real-fixture comparison passed
human geometry review; strict numeric parity failed and is not claimed.

## Environment

- Machine: 16 GB M4 MacBook Pro
- Browser: Chromium 150 via the Codex in-app browser
- Adapter: Apple, Metal 3
- Reported maximum storage-buffer binding: 4096 MiB
- Reported maximum buffer: 4096 MiB
- ONNX Runtime Web: 1.27.0
- Execution providers requested: `webgpu` only; no WASM fallback configured
- Secure context and cross-origin isolation: enabled

## Results

The tiny known-good `Add` graph returned the exact expected output
`[3, 5, 7, 9]`. A subsequent fixed-shape, two-view run of the 1.8 GiB
aggregator-INT8 VGGT candidate also completed successfully:

- session creation and model loading: 9.59 s;
- zero-input inference: 72.64 s;
- pose output: `[1,2,9]`, all finite;
- depth output: `[1,2,518,518,1]`, all finite;
- confidence output: `[1,2,518,518]`, all finite.

This proves that the current graph can load and execute through the WebGPU-only
session configuration on the target Mac. It does not yet prove real-fixture
browser parity or production hosting behavior.

## Real-fixture browser comparison

The ignored parity bundle was generated from the same square-padded kitchen
input and native aggregator-INT8 outputs. Running it in the retained browser
session measured:

- warm WebGPU inference: 55.83 s;
- depth max/mean absolute difference versus native INT8: `0.6344` / `0.02455`;
- confidence max/mean absolute difference: `5.1439` / `0.39195`;
- pose encoding max/mean absolute difference: `0.02590` / `0.005581`;
- all browser output values finite.

The browser path therefore executes reliably but does not meet the raw-output
`1e-3` threshold. Its added mean drift is smaller than the previously approved
FP32-to-INT8 drift, but raw confidence thresholds must not be reused and strict
numeric parity must not be claimed.

## Human stop gate

The browser diagnostic now unprojects native INT8 and Browser WebGPU depth with
their respective predicted cameras and displays both point clouds with shared
framing. It supports top, front, and side projections plus an adjustable
per-model confidence percentile; its default 45% setting rendered 15,146 native
points and 15,151 browser points in the validation run. The two default top-view
panels appeared closely aligned during implementation, and projection switching
completed without browser errors, but that observation is not human approval.

The repository owner approved the M2 geometry match on 2026-09-02. See decision
record `docs/decisions/0006-approve-m2-webgpu-geometry.md`. If later scenes expose
unacceptable alignment, investigate precision-sensitive WebGPU nodes or retain
selected operations at higher precision rather than weakening this record's
numeric caveat.
