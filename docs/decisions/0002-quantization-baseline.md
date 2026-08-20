# 0002 — Quantization follows the full-precision baseline

- Status: accepted
- Date: 2026-08-19

## Context

The full `facebook/VGGT-1B` checkpoint is approximately 5.03 GB. A browser deliverable will almost certainly require compression, reduced precision, partitioning, or a smaller model. However, quantization changes weights and can damage camera pose, depth, confidence, or point-map quality in ways that visual inspection alone may miss.

WebGPU execution also depends on which quantized operators ONNX Runtime Web supports efficiently. Quantizing the original PyTorch checkpoint before establishing an export path could produce an artifact that is smaller but not browser-executable.

## Decision

Use the original full-precision `facebook/VGGT-1B` checkpoint for M0 reference evidence. Do not treat it as the eventual browser artifact.

Quantization begins only after a selected graph executes in native ONNX Runtime and its outputs match the M0 reference closely enough. Each quantized artifact must be derived reproducibly, versioned by configuration and hash, and compared against the same fixture.

Evaluate in this order:

1. exported FP32/native reference graph;
2. FP16 graph or weights where supported;
3. runtime-supported INT8 or weight-only quantization;
4. lower-bit weight formats only when the browser execution path is demonstrated;
5. mixed precision for layers that exceed camera/depth error tolerances.

## Required evidence per variant

- artifact and external-data size;
- conversion configuration and tool versions;
- native ONNX Runtime operator support;
- ORT Web/WebGPU operator placement and fallbacks;
- camera, depth, confidence, and point-map error against M0;
- cold load, compilation, inference, memory, and readback measurements;
- human side-by-side visual validation before becoming the default.

## Consequences

- M0 requires access to the large original checkpoint and representative CUDA hardware.
- Quantized files are derived VGGT materials and remain subject to the applicable upstream/model license.
- A smaller file is not considered progress if it silently falls back to CPU, fails in WebGPU, or crosses accepted quality thresholds.
- The browser default may ultimately be mixed precision, partitioned, or a smaller model rather than a single uniformly quantized VGGT-1B graph.

