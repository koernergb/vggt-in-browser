# M1 INT8 quantization spike

Status: rejected by the FP32 numeric threshold; accepted by the repository owner
on 2026-08-20 as a browser feasibility candidate after visual review.

## Candidates

| Candidate | Bundle size | Native load | Native inference | Conclusion |
| --- | ---: | ---: | ---: | --- |
| FP32 | 4,634,612,161 bytes | 13.86 s | 33.27 s | Accepted reference |
| All MatMul/Gemm INT8 | 2,134,234,472 bytes | 8.93 s | 30.58 s | Excessive drift |
| Aggregator-only INT8 | about 1.8 GiB | 8.88 s | 32.82 s | Excessive drift; no useful quality improvement |

The aggregator-only candidate quantizes 432 `MatMul`/`Gemm` nodes while keeping
the camera and depth heads in FP32. Its real-fixture drift versus PyTorch FP32 is:

- depth max/mean absolute error: `1.1067` / `0.06506`;
- confidence max/mean absolute error: `11.3535` / `1.4428`;
- pose encoding max/mean absolute error: `0.04645` / `0.01067`.

This fails the full-precision spike's `1e-3` raw-output threshold by a wide
margin. The depth maps remain recognizable, but meaningful background-depth and
camera differences are visible. Do not designate this candidate as the browser
model without explicit human approval. Recommended next technical experiment:
selective sensitivity testing or lower-bit weight-only WebGPU-compatible
quantization, rather than blanket dynamic INT8.

## Human stop gate

**RESOLVED:** The owner found the visual appearance acceptable and approved
continuing. FP32 remains the quality baseline. The next stop is required only if
camera-derived geometry is visibly misaligned or browser execution introduces
another material quality-versus-size decision.
