# 0005 — Accept aggregator INT8 as a browser feasibility candidate

- Status: accepted for feasibility testing
- Date: 2026-08-20
- Approved by: repository owner after reviewing the FP32/INT8 depth comparison

## Decision

Carry the aggregator-only dynamic INT8 model into browser feasibility work. It
is not the final quality baseline, and its larger raw-output drift does not
replace the accepted FP32 parity threshold.

## Conditions

- Keep FP32 as the correctness oracle.
- Validate camera-derived alignment and reconstructed geometry before calling
  INT8 suitable for the demo.
- Recalibrate confidence filtering rather than reusing FP32 scalar thresholds.
- Confirm that the quantized operators actually execute on WebGPU without a
  silent CPU fallback.
- Stop for renewed human judgment if reconstructed geometry is visibly
  misaligned or if the browser requires another material quality tradeoff.

## Rationale

The candidate reduces the local bundle from 4.63 GB to about 1.8 GiB, and the
reviewed foreground depth remains visually plausible. This warrants a browser
experiment even though depth, confidence, and pose encoding exceed the strict
raw-output threshold.
