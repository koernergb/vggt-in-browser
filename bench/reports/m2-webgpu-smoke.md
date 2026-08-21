# M2 WebGPU smoke report

Status: measured WebGPU execution passed; real-fixture parity pending.

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

## Next check

Run the real two-view kitchen tensor in the same retained session, compare the
three outputs with native INT8, and record warm inference time. The harness and
ignored parity-file preparation tool are implemented for this check.
