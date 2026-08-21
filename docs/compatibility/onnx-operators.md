# ONNX operator compatibility matrix

Initial M1 matrix. Status values must be updated from generated graph/runtime evidence, not expectation.

| Component | Status | Evidence / next check |
| --- | --- | --- |
| Image input and static reshape | exported | Fixed input is `[1,2,3,518,518]`; ONNX checker confirms it. Native runtime execution remains pending. |
| DINOv2 patch embedding | exported | Present in the structurally valid graph. Native runtime execution remains pending. |
| RoPE2D position grid | rewritten and exported | `aten::cartesian_prod` is unsupported in opset 17. `OnnxPositionGetter` uses exactly equivalent row-major `meshgrid + stack`; equality tests cover representative grid sizes. |
| Frame attention | exported | Attention is decomposed into primitive ONNX operations. Native runtime execution remains pending. |
| Global attention | exported | Static view/token reshapes passed export and structural validation. Native runtime execution remains pending. |
| Camera head iterations | exported | The wrapper selects the final iterative pose encoding, producing `[1,2,9]`. |
| Pose-to-camera conversion | deferred | Keep outside first graph; postprocess pose encoding separately. |
| DPT depth head | rewritten and supported natively | ORT found mixed float32/float64 inputs to a positional-embedding `Einsum`. The export patch preserves float32 with a tolerance test; the regenerated graph runs successfully. |
| Depth activation/confidence | supported natively | Real-fixture drift versus MPS PyTorch: depth max/mean absolute error `5.87e-5`/`1.25e-6`; confidence `5.08e-4`/`1.11e-5`. |
| Point head | deferred | Add only after camera+depth graph executes. |
| Tracking head | deferred | Outside first export scope. |

The first export contains 12,813 nodes across 38 operator types. It passes
`onnx.checker.check_model` when loaded by path with external data. The legacy
exporter currently emits roughly 4.3 GB across 540 files; consolidating those
weights into a browser-friendly package is required before distribution.

The corrected graph loads in native ONNX Runtime 1.22.1 on the M4 Mac CPU and
executes the two-view kitchen fixture. All selected output shapes match and all
values are finite. Pose encoding max/mean absolute error is `9.69e-7`/`2.51e-7`.
