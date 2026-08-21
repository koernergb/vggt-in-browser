# M2 WebGPU smoke report

Status: measured WebGPU execution and real-fixture comparison completed; strict
numeric parity failed and task-level visual validation is required.

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

Before this artifact is called the demo-quality browser model, render the
browser-derived depth/camera geometry beside native INT8 and obtain human visual
approval. If alignment is unacceptable, investigate precision-sensitive WebGPU
nodes or retain selected operations at higher precision.
