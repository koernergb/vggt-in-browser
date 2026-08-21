# 0004 — First ONNX slice is fixed-shape camera plus depth

- Status: accepted
- Date: 2026-08-20

## Decision

Export a batch-one, two-view, 518×518 graph containing the VGGT aggregator, camera head, and depth head. Return pose encoding, depth, and depth confidence. Use ONNX opset 17, static shapes, and external tensor data.

Defer the point head until this graph executes in native ONNX Runtime. Defer tracking because it requires a query-point policy and is not required for the MVP geometry path.

## Rationale

Camera plus depth is the smallest slice that proves useful reconstruction rather than a toy backbone export. Static shapes remove dynamic-shape uncertainty from the first operator-compatibility experiment. Depth can be unprojected with camera parameters during postprocessing, so point-head export is not initially required to render geometry.

## Exit from this decision

If the camera-plus-depth graph cannot be exported or executed after three focused blocker rewrites, stop and present evidence for partitioning the aggregator and heads rather than silently reducing the slice further.
