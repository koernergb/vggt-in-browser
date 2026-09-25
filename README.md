# VGGT in the Browser

Local multi-view 3D reconstruction in a WebGPU-capable browser: two overlapping photos → on-device VGGT inference → interactive point cloud and PLY export. Images and tensors never leave the device.

## Status

**Finished.** The delivered MVP reconstructs exactly two views entirely in the browser with ONNX Runtime Web + WebGPU. Upload, EXIF-aware preprocessing, inference, interactive orbit viewer, confidence filtering, and PLY export are implemented and runnable locally.

Scope that stayed out of the MVP (documented, not unfinished work): three/four-view models, hosted production deployment, and broader device-matrix optimization. Milestone history and stop gates remain in [`MILESTONES.md`](MILESTONES.md).

## What it does

```text
local images (exactly 2)
  → deterministic preprocess (EXIF, resize, white-pad)
  → INT8 VGGT ONNX on WebGPU (ORT Web worker)
  → camera / depth / point-map postprocessing
  → interactive canvas viewer + optional PLY export
```

Measured on a 16 GB M4 MacBook Pro (Chromium, Apple Metal): cached model load ~8 s, two-view inference ~51 s, ~27k sampled points at the default confidence cutoff. See `bench/reports/` for parity, privacy, and golden-scene notes.

## Quick start

Model weights and fixture tensors are intentionally not in Git. Keep a local Vite process running while the page is open:

```bash
npm install
npm run generate:smoke-model
npm run dev -- --port 5174 --strictPort
```

Then open the app, optionally run the WebGPU smoke test under diagnostics, and either:

- choose two overlapping JPEG/PNG/WebP images (≤25 MiB each), or
- run the installed golden fixture when local assets are present.

Drag to orbit, scroll to zoom, adjust confidence / point size, reset the view, or export PLY. Geometry conventions (relative scale, camera encoding, Y-down) are documented in [`docs/geometry-conventions.md`](docs/geometry-conventions.md).

If the page stays open after the Vite process exits, smoke tests may still work from memory while VGGT reports missing local assets—restart the server and retry.

## Repository layout

| Path | Role |
| --- | --- |
| `apps/web/` | Browser UI, preprocessing, worker inference, viewer |
| `model/reference/` | PyTorch reference runner and fixture tooling |
| `model/export/` | ONNX / smoke-model export helpers |
| `bench/fixtures/` | Fixture manifests and validators |
| `bench/reports/` | Milestone evidence (M0–M3) |
| `config/` | Pinned upstream / model configuration |
| `docs/decisions/` | Architecture and go/no-go decision records |

## Reference baseline (optional)

To reproduce the PyTorch reference path used for parity work:

1. Clone the pinned upstream VGGT revision from `config/upstream-vggt.json`.
2. Create a Python 3.10 environment and install `model/reference/requirements.txt` (or `requirements-mps.txt` on Apple Silicon).
3. Fetch and validate the four-view kitchen fixture, then run the reference twice:

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

Defaults are `facebook/VGGT-1B`, camera/depth/point heads, `crop` preprocessing, and automatic BF16/FP16 on CUDA. Model download is multi-gigabyte—see [`docs/reference-environment.md`](docs/reference-environment.md).

## Limitations

- **Two views only.** The shipped ONNX artifact has a fixed `[1,2,3,518,518]` input.
- **Relative scale.** Exported PLY is not metric.
- **WebGPU required.** No server-side inference fallback.
- **Large local model.** The INT8 candidate is ~1.8 GiB and must be served locally; it is not redistributed in this repository.
- **Privacy.** The app does not upload images or tensors; verify network traffic yourself before treating that as a release claim (`bench/reports/m3-privacy.md`).

## Evidence policy

- Benchmarks record commit, upstream revision, model id, artifact hash when available, browser/runtime, OS, GPU, precision, view count, resolution, and stage timings.
- Raw results are never overwritten. Summaries distinguish `measured`, `estimated`, `target`, and `not tested`.
- Weights, raw tensors, and fixture images stay out of Git by default.

## Source material

- [Official VGGT repository](https://github.com/facebookresearch/vggt)
- [VGGT paper](https://arxiv.org/abs/2503.11651)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)

VGGT code and checkpoints have their own license and acceptable-use terms. This repository does not redistribute them.
