# 0006 — Approve M2 WebGPU geometry

- Status: accepted
- Date: 2026-09-02
- Decision maker: repository owner

The repository owner explicitly approved the side-by-side native INT8 versus
Browser WebGPU geometry comparison after being given the local inspection flow.
This satisfies the subjective M2 geometry gate. The measured tensor drift in
the M2 report remains a known limitation and strict numeric parity is not
claimed.

The accepted candidate remains restricted to the project's non-commercial
research scope and fixed two-view, 518×518 input contract.
