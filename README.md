# VGGT in the Browser

Research project exploring useful multi-view 3D reconstruction entirely in a WebGPU-capable browser. The intended flow is 2–4 local images → camera/depth or point-map inference → interactive geometry, without an application inference server.

## Status

**M0 — reference baseline is complete. M1 — native ONNX export and parity are demonstrated. M2 — the INT8 candidate executes successfully in a WebGPU-only browser session, real-fixture drift is measured, and the diagnostic renders native/WebGPU geometry side by side; human visual approval is pending.** Everything described as a target is unmeasured until a benchmark report says otherwise.

An exploratory four-view MPS run succeeds on a 16 GB M4 MacBook Pro and has passed human visual validation. Independent CUDA oracle evidence from the sibling `vggt-mlx` project is accepted for M0, with the documented limitation that its one/three-view fixtures differ from this repository's four-view kitchen fixture. None of this demonstrates browser/WebGPU inference.

Current repository contents:

- detailed implementation and human stop gates in [`MILESTONES.md`](MILESTONES.md);
- reproducible reference-run scaffolding under `model/reference/`;
- a fixture manifest schema and validator under `bench/fixtures/`;
- pinned upstream/model configuration under `config/`;
- baseline decision documentation under `docs/decisions/`.

## Target architecture

```text
local images
  → deterministic preprocessing
  → converted VGGT backbone/heads
  → ONNX Runtime Web + WebGPU
  → camera/depth/point-map postprocessing
  → Three.js/WebGPU viewer
```

The repository now includes the M2 browser diagnostic and a lightweight geometry-comparison renderer. The upload workflow and interactive end-user viewer remain M3 work.

## M0 quick start

M0 deliberately does not auto-download a multi-gigabyte checkpoint or assume that fixture images may be published.

1. Clone the pinned upstream VGGT revision documented in `config/upstream-vggt.json`.
2. Create a Python 3.10 environment and install the reference requirements.
3. Fetch the selected official four-view kitchen fixture. The files are hash-verified and remain ignored locally.
4. Validate the fixture, then run the reference twice:

```bash
python -m pip install -r model/reference/requirements.txt
python model/reference/fetch_fixture.py
python model/reference/validate_fixture.py bench/fixtures/manifest.json
python model/reference/run_reference.py \
  --fixture bench/fixtures/manifest.json \
  --output bench/results/m0/run-01.json
python model/reference/run_reference.py \
  --fixture bench/fixtures/manifest.json \
  --output bench/results/m0/run-02.json
python model/reference/compare_runs.py \
  bench/results/m0/run-01.json \
  bench/results/m0/run-02.json \
  --output bench/results/m0/determinism.json
```

The reference runner defaults to `facebook/VGGT-1B`, camera/depth/point heads, preprocessing mode `crop`, and automatic BF16/FP16 selection on CUDA. These settings are recorded in every result and can be overridden explicitly. Model download may require several gigabytes and network access; do not start it unintentionally.

See [`docs/reference-environment.md`](docs/reference-environment.md) for exact environment and result-bundle guidance.

## Local WebGPU harness

The diagnostic page depends on a running local Vite process because model
weights and ignored parity tensors are intentionally not bundled into Git. Keep
this command running for as long as the page is open:

```bash
npm install
npm run generate:smoke-model
npm run dev -- --port 5174 --strictPort
```

If the page remains open after that process exits, the small smoke test may
still work from browser memory while VGGT reports that local assets cannot be
fetched. Restart the command and retry; that is a server-lifecycle error, not a
WebGPU model failure.

For the M4/MPS exploratory path, install `model/reference/requirements-mps.txt` instead of the CUDA-oriented requirements, then install the pinned VGGT source revision documented in `config/upstream-vggt.json`.

## Evidence policy

- Benchmarks must record commit, upstream revision, model identifier, artifact hash when available, browser/runtime, OS, GPU, precision, view count, resolution, and stage timings.
- Raw results are never overwritten. Published summaries must distinguish `measured`, `estimated`, `target`, and `not tested`.
- Model weights, raw tensors, and fixture images are ignored by default. Only manifests, checksums, compact summaries, and explicitly approved assets belong in Git.
- Uploaded images or derived tensors must not leave the device in the eventual browser application. This promise must be verified with production network inspection before release.

## Human input currently required

**STOP — visual approval required.** Run real-fixture parity in the local WebGPU harness, compare the native INT8 and Browser WebGPU panels across the top/front/side projections and confidence settings, and explicitly approve or reject the match. The agent must not declare M2 complete or begin treating this candidate as the demo-quality model until that judgment is recorded.

## Source material

- [Official VGGT repository](https://github.com/facebookresearch/vggt)
- [VGGT paper](https://arxiv.org/abs/2503.11651)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)

VGGT code and checkpoints have their own license and acceptable-use terms. This repository does not redistribute them at this stage.
