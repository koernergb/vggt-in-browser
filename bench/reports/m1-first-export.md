# M1 first ONNX export report

Status: native runtime validation and PyTorch parity passed; packaging pending.

## Approved slice

- Model: `facebook/VGGT-1B`
- Input: fixed float32 `[1, 2, 3, 518, 518]`
- Graph: aggregator, camera head, and depth head
- Outputs: pose encoding `[1,2,9]`, depth `[1,2,518,518,1]`, and depth confidence `[1,2,518,518]`
- ONNX opset: 17

## Result

The legacy PyTorch exporter completed and `onnx.checker.check_model` passed.
The graph contains 12,813 nodes and 38 operator types. The initial blocker,
`aten::cartesian_prod` in VGGT's position grid, was replaced by an exactly
equivalent, unit-tested `meshgrid + stack` implementation.

The exported model is not committed. The corrected bundle is 4,634,612,161
bytes across 540 files, with deterministic manifest digest
`b8148c917c5777adcf83bc7c489fec0b1ada7721426649126a021e61dc8054ec`.
This fragmentation is unsuitable for web distribution; consolidate and
quantize it before browser model loading.

The first native ORT load exposed a mixed float32/float64 `Einsum` in the DPT
sinusoidal position embedding. An export-only helper now keeps the frequency
vector in the position grid's float32 dtype and is checked against upstream
within float32 tolerance. The regenerated graph passed ONNX checking and native
execution.

## Native execution and parity

Measured on the 16 GB M4 MacBook Pro with ONNX Runtime 1.22.1 CPU execution:

- deterministic zero-input smoke: 14.63 s session load, 39.71 s inference;
- real two-view kitchen fixture with square padding: 13.86 s load, 33.27 s inference;
- depth max/mean absolute error versus MPS PyTorch FP32: `5.865e-5` / `1.252e-6`;
- confidence max/mean absolute error: `5.083e-4` / `1.108e-5`;
- pose encoding max/mean absolute error: `9.686e-7` / `2.510e-7`;
- every output shape matched and every output value was finite.

For this spike, the accepted numeric threshold is max absolute error below
`1e-3` for each selected raw output. All outputs pass comfortably. Geometry
level validation will be repeated after quantization and browser execution.

## Human stop gate

No human input is needed yet. Stop for human judgment if quantization produces
materially different geometry, or if browser constraints force a
quality-versus-size decision.
