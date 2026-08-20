# M0 reference environment

The reference environment exists to create evidence for later ONNX and browser comparisons. It is not the browser runtime.

## Pinned baseline

- Upstream repository: `facebookresearch/vggt`
- Revision: `a288dd0f14786c93483e45524328726ab7b1b4ce`
- Python: 3.10
- PyTorch: 2.3.1
- torchvision: 0.18.1
- Example CUDA wheel target from upstream documentation: CUDA 12.1
- Default preprocessing: upstream `load_and_preprocess_images(..., mode="crop")`
- Default model identifier in the runner: `facebook/VGGT-1B`

The machine-readable source of truth is `config/upstream-vggt.json`.

## Checkpoint selection

The repository owner confirmed on 2026-08-19 that this is strictly non-commercial work. The approved M0 reference is `facebook/VGGT-1B`, whose model card identifies CC BY-NC 4.0. The 5.03 GB full-precision artifact is intentionally not committed.

Do not commit or redistribute weights. Record the selected identifier and the downloaded artifact SHA-256 in the result bundle when it can be determined. If the project scope ever becomes commercial, stop using this checkpoint and make a new license/checkpoint decision before continuing.

## Environment setup

Clone the upstream project next to this repository or install it from the pinned revision. A direct editable clone is easiest while tracing modules:

```bash
git clone https://github.com/facebookresearch/vggt.git
cd vggt
git checkout a288dd0f14786c93483e45524328726ab7b1b4ce
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
python -m pip install pytest
```

Run this repository's scripts from its root while the environment is active. If the upstream clone is not installed, add its directory to `PYTHONPATH` explicitly rather than copying upstream code here.

## Fixture requirements

The golden fixture must contain 2–4 ordered images with meaningful overlap, texture, and viewpoint change. The manifest records:

- stable scene ID and description;
- source/provenance and usage permission;
- whether publication is approved;
- exact ordered paths;
- width, height, byte size, and SHA-256 for each image.

Fixture order is part of the test. The selected fixture is the first four kitchen views from the pinned official VGGT repository. Run `python model/reference/fetch_fixture.py` to retrieve only those files and verify their hashes. The images remain ignored locally rather than being duplicated in this repository.

The validator rejects missing files, hash/dimension drift, duplicate order values, and incomplete provenance metadata.

## Result bundle

Every run JSON includes:

- repository and upstream revisions;
- model ID, checkpoint path/hash when supplied, preprocessing, heads, seed, device, and dtype;
- Python, platform, PyTorch, CUDA, cuDNN, and GPU metadata;
- ordered fixture hashes and preprocessed tensor shape;
- wall-clock load, preprocessing, inference, postprocessing, and total times;
- tensor shape, dtype, element count, finite/non-finite counts, min/max/mean/std, deterministic sample values, and raw-byte SHA-256.

The normal result is a compact JSON summary. Use `--save-arrays` only for a controlled local comparison; generated arrays are ignored by Git.

## Running on human-provided hardware

If the coding environment has no representative NVIDIA GPU, a human should run:

```bash
python model/reference/validate_fixture.py bench/fixtures/manifest.json
python model/reference/run_reference.py --fixture bench/fixtures/manifest.json \
  --output bench/results/m0/run-01.json
python model/reference/run_reference.py --fixture bench/fixtures/manifest.json \
  --output bench/results/m0/run-02.json
python model/reference/compare_runs.py bench/results/m0/run-01.json \
  bench/results/m0/run-02.json --output bench/results/m0/determinism.json
```

Return the three unedited JSON files. Also preserve the terminal output if a run fails. Expected storage is several gigabytes for the downloaded checkpoint plus normal Python/CUDA dependencies; compact result JSON is typically small. Runtime depends strongly on GPU and view count and must be reported rather than estimated as measured.

## Apple Silicon exploratory run

An MPS run may identify operator or memory problems early, but it does not replace the canonical CUDA reference because the upstream inference path and precision recommendation target CUDA. On constrained unified-memory machines, start with the first two ordered views:

```bash
python model/reference/run_reference.py \
  --fixture bench/fixtures/manifest.json \
  --device mps --dtype float16 --max-views 2 \
  --output bench/results/m0/mps-2view-run-01.json
```

If it succeeds without swapping or memory failure, repeat for determinism and then attempt `--max-views 4`. Record MPS results as a separate series; never compare a two-view MPS run to a four-view CUDA run as if only the device changed.

The tested M4 environment uses `model/reference/requirements-mps.txt` with PyTorch 2.12.1. The upstream-pinned PyTorch 2.3.1 wheel is built with MPS but reports MPS unavailable on this M4/macOS combination because it predates the hardware. GPU access may also be hidden inside a restricted execution sandbox even when it works from a normal terminal.

Generate the human validation image after a run saved with `--save-arrays`:

```bash
python model/reference/render_validation.py \
  bench/reference/mps-4view-run-02.npz \
  --output bench/reference/mps-4view-validation.png
```
