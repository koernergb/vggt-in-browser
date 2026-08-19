# VGGT in the Browser — Implementation Milestones

This document turns the build brief into an execution plan for humans and coding agents. The goal is not merely to make a demo appear to work. Every milestone must leave behind reproducible evidence about correctness, browser compatibility, performance, and limitations.

The target outcome is a static web application that accepts 2–4 overlapping images, performs useful VGGT-compatible inference locally through WebGPU, and renders estimated cameras and geometry without sending the images to an inference server.

## How to use this document

Work through milestones in order. A later milestone may be explored early, but it must not be declared complete while an earlier prerequisite is unresolved.

An agent may proceed autonomously when it can make a reversible implementation choice, validate it with existing fixtures, and document the result. It must stop when a decision depends on human intent, subjective visual judgment, access to hardware or data it does not have, acceptance of a material scope reduction, or approval of a costly/irreversible action.

The labels below are normative:

- **AGENT:** Work that an agent can normally perform without interruption.
- **HUMAN INPUT:** Information or assets a person must provide.
- **HUMAN VALIDATION:** A person must inspect a result and report whether it is acceptable.
- **HUMAN DECISION:** A person must choose between materially different paths.
- **HUMAN APPROVAL:** A person must explicitly authorize an external, costly, destructive, security-sensitive, or release action.
- **STOP:** The agent must not guess or silently continue past this point. It must summarize the evidence, ask the specified question, and wait for the answer.

## Operating rules for agents

1. Preserve raw benchmark outputs. Derived summaries may be regenerated; source measurements must not be overwritten.
2. Separate measured facts from estimates and goals. Use labels such as `measured`, `estimated`, `target`, and `not tested` in reports.
3. Record the environment for every benchmark: commit, browser/version, OS, GPU/adapter, power mode when known, model artifact hash, image count, input resolution, precision, and runtime settings.
4. Use fixed fixtures for parity and performance comparisons. Do not substitute new images mid-comparison without starting a new result series.
5. Never claim browser privacy solely because inference is intended to be local. Verify network behavior in a production build and document every request the page makes.
6. Do not add a server-side inference fallback without a human scope decision. It changes the defining product promise.
7. Do not publish model weights, fixtures, or third-party assets until their licenses and redistribution terms are recorded.
8. Prefer a small, reproducible experiment over a large unverified integration.
9. Commit milestone evidence alongside the code when practical: compatibility matrices, numeric summaries, benchmark JSON, screenshots, and decision records.
10. When a stop gate is reached, keep the repository in a runnable state, record the blocker, and ask one concrete question with the available options and consequences.

## Cross-milestone evidence layout

The intended repository structure is:

```text
apps/web/                 Browser UI and capability reporting
packages/inference/       ORT Web sessions, workers, tensors, profiling
packages/geometry/        Camera, depth, point-map postprocessing
packages/viewer/          Three.js/WebGPU visualization
model/export/             PyTorch-to-ONNX export and graph inspection
model/patches/            Documented rewrites and custom operations
bench/fixtures/           Licensed fixed inputs and fixture manifests
bench/reference/          Golden summaries and hashes, not casual outputs
bench/results/            Machine-readable benchmark runs
bench/reports/            Human-readable comparisons and conclusions
docs/decisions/           Architecture and go/no-go decision records
docs/compatibility/       Operator, browser, GPU, and precision matrices
```

Large weights and raw tensors should normally live outside Git. Track manifests, checksums, download instructions, and compact numeric summaries instead.

## Project-wide human stop gates

These gates apply even when a milestone does not repeat them:

| Trigger | Required human involvement | Agent behavior while stopped |
| --- | --- | --- |
| A model, fixture, or dependency has unclear licensing or redistribution terms | **HUMAN APPROVAL** from the repository owner after the license evidence is summarized | Do not commit or publish the questionable artifact. Continue only with clearly licensed substitutes if that does not change the test. |
| A command requires credentials, paid compute, a large download, or publishing externally | **HUMAN APPROVAL** with the exact action, expected cost/size, and destination | Do not work around access controls or use unapproved credentials. |
| A proposed change sends uploaded images or derived user data off-device | **HUMAN DECISION** because this changes the core privacy promise | Do not implement or enable the transfer by default. |
| Progress requires deleting or replacing golden fixtures/results | **HUMAN APPROVAL** after explaining why versioning them is insufficient | Preserve the existing evidence. |
| Two technically viable paths differ mainly in product experience or visual quality | **HUMAN DECISION** with a runnable comparison when possible | Keep both paths isolated and avoid choosing based on assumed taste. |
| A milestone exit criterion cannot be met without reducing scope | **HUMAN DECISION** supported by measurements and a proposed revised criterion | Do not mark the milestone complete under a weaker, unrecorded definition. |
| Release claims exceed directly reproduced evidence | **HUMAN APPROVAL** of corrected wording or additional validation work | Block the release claim, not unrelated engineering work. |

---

## M0 — Reference baseline

### Objective

Create a reproducible PyTorch reference run for a fixed 2–4 image scene and preserve compact golden evidence for later ONNX and browser comparisons.

### Prerequisites

- The upstream VGGT revision or release is pinned.
- The intended checkpoint is identified.
- At least one overlapping multi-view scene is legally usable for development and publication.
- A CUDA-capable reference environment is available, or the limitation is documented and an alternative reference environment is explicitly accepted.

### Work plan

1. **AGENT:** Create the initial repository skeleton, environment documentation, `.gitignore`, and a root README that distinguishes current results from planned work.
2. **AGENT:** Pin the upstream source revision, Python version, PyTorch/CUDA versions, checkpoint identifier, and preprocessing configuration.
3. **AGENT:** Add a fixture manifest containing image filenames, hashes, dimensions, provenance, license, and expected view order.
4. **AGENT:** Implement a reference runner that:
   - loads a fixed fixture deterministically;
   - records preprocessing dimensions and normalization;
   - runs inference without gradients;
   - saves tensor names, shapes, dtypes, finite-value checks, min/max/mean/std, selected samples, and hashes;
   - records camera and depth/point-map outputs needed by downstream stages;
   - emits machine-readable JSON.
5. **AGENT:** Run the reference at least twice and compare summaries to characterize determinism.
6. **AGENT:** Add a smoke test that catches missing weights, incorrect fixture order, non-finite outputs, and gross output-shape drift.
7. **AGENT:** Write `docs/decisions/0001-reference-baseline.md` describing the exact baseline and known limitations.

### Human gates

- **HUMAN INPUT:** Provide or approve the 2–4 image fixture and its permitted usage. The fixture should have meaningful overlap, texture, and viewpoint change without containing sensitive material.
- **STOP — fixture missing or unsuitable:** Ask: “Which approved image set should be the golden fixture, and may it be committed publicly?” Do not invent provenance or commit personal photos by default.
- **HUMAN INPUT:** If the agent cannot access suitable reference hardware, a person must run the documented command and return the unedited result bundle plus environment metadata.
- **STOP — reference hardware unavailable:** Present the minimum required command, expected runtime/storage, and a result-bundle location. Do not fabricate reference numbers or substitute browser output as its own reference.
- **HUMAN VALIDATION:** Review a visualization of the reference cameras/depth/point map and confirm that the reconstruction is semantically plausible.
- **STOP — suspicious reference output:** If the visualization is clearly wrong, ambiguous, or highly sensitive to image order, ask the human whether to replace the fixture, alter preprocessing, or accept the limitation before freezing goldens.

### Deliverables

- Reproducible environment and reference-run instructions.
- Licensed fixture manifest and immutable file hashes.
- Reference runner and smoke tests.
- Compact golden summaries and output schema.
- A visual sanity-check artifact.
- Baseline decision record.

### Exit criteria

- A clean environment can reproduce the run from documented inputs.
- Repeated runs have characterized numeric variation.
- Required tensors have stable names or an explicit mapping, shapes, dtypes, finite values, summaries, and hashes.
- A human has confirmed that the baseline reconstruction is plausible.
- No unapproved large or restricted artifact is committed.

---

## M1 — ONNX export spike

### Objective

Export the smallest useful VGGT forward path, execute it in native ONNX Runtime, and produce an evidence-based operator compatibility report before UI work expands.

### Work plan

1. **AGENT:** Trace model modules, tensor shapes, dynamic axes, control flow, custom operators, and outputs used by the MVP.
2. **AGENT:** Define the smallest export slice that still proves meaningful model behavior. Begin with fixed batch, view count, and resolution unless dynamic shapes are already reliable.
3. **AGENT:** Add an export script with pinned opset, explicit inputs/outputs, constant-folding choice, and external-data handling.
4. **AGENT:** Validate the exported model structurally with ONNX tooling and record model/external-weight sizes and hashes.
5. **AGENT:** Execute the graph in native ONNX Runtime on the golden fixture.
6. **AGENT:** Compare each selected output with PyTorch using absolute error, relative error, cosine similarity where meaningful, and task-level camera/depth metrics where available.
7. **AGENT:** Classify every failure:
   - exporter limitation;
   - unsupported operator;
   - dynamic-shape issue;
   - precision/numeric drift;
   - excessive graph or weight size;
   - runtime memory failure;
   - incorrect rewrite.
8. **AGENT:** For each rewrite, add a focused unit test and document semantic equivalence, assumptions, and measured error.
9. **AGENT:** Publish an operator compatibility matrix with statuses: `supported`, `rewritten`, `partition candidate`, `custom kernel candidate`, `blocked`, or `not tested`.

### Human gates

- **HUMAN DECISION:** Approve the first “smallest useful forward path” if excluding outputs changes the product story. The agent should recommend a path and explain what it proves and does not prove.
- **STOP — path ambiguity:** If camera, depth, point maps, and tracks cannot all be exported together, ask which output family is the MVP priority. Default recommendation: preserve camera plus dense depth/point-map information before tracks.
- **HUMAN DECISION:** Choose whether a difficult block should be rewritten, partitioned, deferred, or replaced with a smaller model when the choice changes accuracy, schedule, or portfolio framing.
- **STOP — rewrite escalation:** Stop when any of these occurs:
  - three successive rewrites fail to move the same blocker;
  - rewrites would replace the majority of a major model block;
  - task-level quality falls materially despite acceptable elementwise metrics;
  - the estimated browser artifact remains operationally implausible after quantization assumptions are stated.
  Present the compatibility evidence and one recommended path before asking for a decision.
- **HUMAN VALIDATION:** Review side-by-side reference and ONNX visualizations when numeric thresholds do not clearly predict perceptual usefulness.

### Deliverables

- Deterministic export script and pinned export configuration.
- ONNX graph plus external-data manifest and hashes, stored or fetched according to license/size policy.
- Native ONNX Runtime comparison report.
- Operator compatibility matrix.
- Tests and documentation for every graph rewrite.
- Decision record for deferred or partitioned components.

### Exit criteria

- The selected graph executes in native ONNX Runtime on the golden fixture.
- Output drift is measured against the reference and accepted thresholds are recorded.
- Every unsupported operation has a documented disposition.
- Model size and expected browser memory implications are reported honestly.
- A human has approved any material reduction in the selected forward path.

---

## M2 — Browser WebGPU parity

### Objective

Run the selected ONNX graph in ONNX Runtime Web with WebGPU on one named target machine and demonstrate measured parity with the native baseline.

### Work plan

1. **AGENT:** Scaffold the TypeScript/Vite browser harness and a Web Worker boundary for model loading and inference.
2. **AGENT:** Add capability diagnostics for secure context, WebGPU availability, adapter/device acquisition, relevant limits/features, cross-origin isolation where used, browser/version, and runtime configuration.
3. **AGENT:** First run a tiny known-good ONNX model to prove ORT Web packaging, WASM asset routing, worker messaging, WebGPU session creation, and production-build behavior.
4. **AGENT:** Implement model artifact loading with progress, cancellation, clear error categories, checksums, and caching behavior that can be inspected and invalidated.
5. **AGENT:** Execute the VGGT export with the fixed input shape before adding general uploads.
6. **AGENT:** Capture browser output summaries using the same schema as M0/M1 and compare them offline without committing huge raw tensors.
7. **AGENT:** Measure separately:
   - artifact transfer and cache hit/miss;
   - session creation and shader/graph compilation;
   - warm-up;
   - inference;
   - output readback;
   - total wall time.
8. **AGENT:** Document all CPU fallbacks, warnings, unsupported nodes, and browser flags.
9. **AGENT:** Add graceful unsupported-device and out-of-memory states. Do not let the page hang indefinitely.

### Human gates

- **HUMAN INPUT:** Provide access to or results from the named target browser/GPU if the agent environment cannot expose WebGPU accurately.
- **STOP — no representative WebGPU hardware:** Ask the human to run a versioned diagnostic/benchmark page and return the downloadable result JSON. Do not infer GPU behavior from native ONNX Runtime.
- **HUMAN APPROVAL:** Required before downloading multi-gigabyte model artifacts or using paid/cloud GPU resources. State the expected size, time, and destination.
- **HUMAN VALIDATION:** On the target machine, confirm that the page remains responsive, progress reporting is understandable, cancellation works, and errors are actionable.
- **STOP — runtime incompatibility:** Stop after the compatibility report shows that proceeding requires a custom WebGPU kernel, a different runtime, a browser flag users would not normally enable, or a material model partition. Ask the human to approve the proposed technical direction.
- **STOP — unacceptable cold start:** If cold-start behavior is clearly outside the documented target, provide measured download/compile/inference components and ask whether to continue as a desktop research demo, prioritize quantization/sharding immediately, or change models.

### Deliverables

- Minimal production-build browser harness.
- Capability and error diagnostics.
- Browser parity report using the golden fixture.
- Timing breakdown and artifact/cache behavior report.
- Target device/browser record.
- Reproducible result-export mechanism.

### Exit criteria

- A production build executes the selected graph with WebGPU on one documented target machine.
- Browser outputs are compared numerically with native ONNX Runtime and PyTorch.
- No silent CPU fallback is presented as WebGPU success.
- Unsupported hardware fails clearly and safely.
- A human has validated the experience on representative hardware.

---

## M3 — End-to-end geometry demo

### Objective

Turn 2–4 local image uploads into estimated cameras and recognizable interactive geometry entirely in the browser.

### Work plan

1. **AGENT:** Implement local image validation, EXIF-orientation handling, resize/crop/letterbox behavior, color conversion, normalization, batching, and deterministic view ordering.
2. **AGENT:** Keep original images local. Add no analytics event, logging payload, or crash report that includes image bytes, filenames, thumbnails, or derived tensors.
3. **AGENT:** Implement postprocessing with explicit coordinate-system, handedness, camera-convention, scale, confidence, and invalid-value handling.
4. **AGENT:** Add unit tests against compact reference arrays for camera conversion, unprojection, confidence filtering, and point-cloud sampling.
5. **AGENT:** Build an interactive viewer with orbit controls, camera frusta, colored points, confidence threshold, point-size control, reset view, and useful loading/error states.
6. **AGENT:** Decimate or stream geometry so visualization does not require unnecessary full-resolution CPU copies.
7. **AGENT:** Add a built-in example or clearly documented test flow if licensing permits.
8. **AGENT:** Add result export only after coordinate conventions and units are documented. Prefer PLY first; GLB may follow when camera/scene conventions are stable.
9. **AGENT:** Verify production network traffic during upload and inference. Record all requests and confirm that user images and derived tensors are not transmitted.
10. **AGENT:** Add end-to-end tests for upload validation and deterministic fixture processing; use human checks for reconstruction quality rather than brittle screenshot-only assertions.

### Human gates

- **HUMAN VALIDATION:** Inspect the golden scene in the browser and verify:
  - the scene is recognizable;
  - camera locations/orientations are plausible;
  - controls feel usable;
  - confidence filtering improves rather than disguises the output;
  - the result is not mirrored, inverted, or incorrectly scaled in an obvious way.
- **STOP — visual acceptance required:** Provide a URL or exact local command, the fixture name, expected result, and a short checklist. Wait for explicit pass/fail notes; do not self-approve subjective reconstruction quality.
- **HUMAN INPUT:** Supply at least one additional real-world scene for generalization testing, with usage permission stated.
- **STOP — coordinate ambiguity:** If multiple conventions produce plausible but different renders, show labeled comparisons and ask which convention matches the reference. Do not tune transforms by eye without recording the math.
- **HUMAN APPROVAL:** Required before enabling telemetry or remote error reporting. The proposal must list collected fields and demonstrate that images/tensors are excluded.
- **HUMAN DECISION:** If only two views are usable, ask whether that satisfies the MVP or whether M3 remains open pending 3–4 view support.

### Deliverables

- Local upload and preprocessing pipeline.
- Geometry postprocessing package with convention documentation and tests.
- Interactive camera/point-cloud viewer.
- Network privacy verification report.
- Golden-scene and secondary-scene validation notes.
- Optional PLY export with documented coordinates.

### Exit criteria

- At least two approved overlapping images produce recognizable geometry locally.
- The displayed cameras and geometry agree with the selected reference convention.
- Production network inspection confirms no image/tensor upload.
- UI remains responsive or clearly communicates non-responsive phases.
- A human has explicitly accepted the visual result on the target machine.

---

## M4 — Profiling and optimization

### Objective

Measure the real bottlenecks and achieve at least one material improvement without hiding accuracy loss or changing the benchmark conditions.

### Work plan

1. **AGENT:** Freeze benchmark protocol and result schema before optimization.
2. **AGENT:** Benchmark cold and warm runs for 1, 2, and 4 views where supported, with repeat count and variability reported.
3. **AGENT:** Record model transfer size, cache behavior, session/compile time, peak memory proxy, preprocessing, inference, postprocessing, readback, render time, and end-to-end latency.
4. **AGENT:** Profile before introducing custom kernels or architectural complexity.
5. **AGENT:** Evaluate changes independently where possible:
   - lower resolution/view count;
   - FP16 paths;
   - INT8 or lower-bit weight quantization supported by the runtime;
   - external-data sharding and cache strategy;
   - graph partitioning and tensor-lifetime reduction;
   - output sampling/decimation;
   - reduced CPU/GPU transfers;
   - custom WGSL only for a demonstrated bottleneck.
6. **AGENT:** For every optimization, rerun parity metrics and the same visual fixtures. Report regressions, not just speedups.
7. **AGENT:** Maintain a before/after table with confidence intervals or at least min/median/max across repeated runs.
8. **AGENT:** Test cancellation, repeated runs, cache invalidation, and memory recovery after failures.

### Human gates

- **HUMAN DECISION:** Approve the primary optimization target—cold start, steady-state latency, view count, memory headroom, or visual quality—when improving one harms another.
- **STOP — target trade-off:** Present a Pareto-style comparison and ask which experience matters most for the intended audience. Do not select a product trade-off solely from synthetic metrics.
- **HUMAN VALIDATION:** Blind or side-by-side review is required for any quantization/resolution change whose numeric regression may be perceptually important.
- **STOP — quality regression:** If an optimization crosses the recorded numeric tolerance or visibly damages camera/depth quality, keep it behind an experimental option and wait for acceptance before making it the default.
- **HUMAN DECISION:** Required before investing in custom WGSL. Show that profiling identifies the operation as a material bottleneck and estimate maintenance cost.
- **STOP — feasibility boundary:** If the full path repeatedly crashes, cannot load within the accepted budget, or requires unsupported features, write a go/no-go decision with evidence and ask whether to reposition the project, partition further, or adopt a smaller model.

### Deliverables

- Versioned benchmark protocol and machine-readable results.
- Device/browser/precision compatibility matrix.
- Before/after report for each accepted optimization.
- Accuracy and visual-quality comparison.
- Memory/failure observations with limitations clearly stated.
- Go/no-go or scope decision record when relevant.

### Exit criteria

- At least one material improvement is reproduced under the frozen protocol.
- Accuracy and visual-quality effects are measured and accepted.
- Benchmark claims name the tested hardware and configuration.
- Failed experiments and negative results remain documented.
- The default configuration reflects an explicit human-approved trade-off when no option dominates.

---

## M5 — Portfolio release

### Objective

Ship an honest, reproducible static demo and technical narrative that clearly separates verified capabilities from goals and known limitations.

### Work plan

1. **AGENT:** Harden capability detection, error messages, progress, cancellation, and recovery.
2. **AGENT:** Add a benchmark/results page generated from checked-in summaries rather than hand-copied headline numbers.
3. **AGENT:** Complete architecture, model rewrite, compatibility, privacy, license, and troubleshooting documentation.
4. **AGENT:** Update the README with:
   - what works today;
   - tested devices/browsers;
   - exact local setup;
   - model/fixture acquisition;
   - measured performance;
   - known failures;
   - privacy behavior;
   - roadmap clearly labeled as future work.
5. **AGENT:** Create a production build with pinned dependencies and verify asset paths, cache invalidation, and fresh-load behavior.
6. **AGENT:** Run release checks on a clean environment and at least one target machine.
7. **AGENT:** Audit browser network traffic, repository secrets, source maps, third-party licenses, artifact licenses, and fixture permissions.
8. **AGENT:** Prepare a short demo script/video plan that uses reproducible fixtures and does not imply unsupported generality.
9. **AGENT:** Tag benchmark data and model manifests with the release commit.

### Human gates

- **HUMAN VALIDATION:** Perform final UX acceptance on the public production build, including first load, cached load, upload, inference, interaction, export, unsupported-device messaging, and failure recovery.
- **HUMAN VALIDATION:** Review all public claims against the evidence table. Pay special attention to “local,” “private,” “WebGPU,” view count, model identity, and performance language.
- **HUMAN APPROVAL:** Required before deployment, public weight/fixture publication, repository visibility changes, release/tag creation, posting a demo video, or publishing a technical article.
- **STOP — release approval:** The agent must provide the candidate URL/build, release commit, test matrix, known issues, license/privacy audit, and proposed public claims. It must wait for an explicit “approve release” response before publishing.
- **STOP — material release defect:** If the production build differs from the validated local build, leaks user data, has unclear licensing, or fails the primary target, block release and present remediation options.

### Deliverables

- Hosted static demo and reproducible production build.
- Public README and architecture documentation.
- Compatibility and benchmark pages.
- License and privacy audit.
- Known-issues list and troubleshooting guide.
- Human-approved release notes and, optionally, demo video/technical post.

### Exit criteria

- The hosted page performs inference without an application inference server.
- At least two overlapping images reconstruct into human-accepted recognizable geometry.
- Camera/depth outputs have traceable comparisons to the PyTorch reference.
- Target support, model size, load time, latency, memory limitations, and failure modes are documented.
- Every rewrite and unsupported operator is documented.
- Public claims are supported by committed evidence.
- The repository owner has explicitly approved publication.

---

## Suggested issue breakdown

Create issues only when the relevant milestone begins; avoid generating a large stale backlog.

### M0 issues

- Repository skeleton and evidence conventions
- Golden fixture provenance and manifest
- Reproducible PyTorch environment
- Reference runner and tensor-summary schema
- Reference visualization and human validation

### M1 issues

- Module/shape trace
- Minimal ONNX export
- Native ORT parity metrics
- Operator compatibility matrix
- Rewrite tests and decision record

### M2 issues

- ORT Web toy-model harness
- Worker and production asset loading
- WebGPU capability diagnostics
- Browser parity result export
- Target-device human validation

### M3 issues

- Image preprocessing parity
- Camera and point-map postprocessing
- Interactive viewer
- Network privacy verification
- Human visual acceptance

### M4 issues

- Frozen benchmark protocol
- Profiling baseline
- Quantization experiments
- Partitioning/tensor lifetime experiments
- Human-approved default configuration

### M5 issues

- Release documentation and evidence page
- Clean-build and target-device matrix
- License/privacy/security audit
- Production deployment candidate
- Final human release approval

## Definition of project done

The project is done when the release candidate satisfies M5—not when an isolated graph first executes. If full-model browser inference proves impractical, the project may still finish successfully only after a human approves a revised scope and the repository publishes a rigorous feasibility boundary, compatibility evidence, an optimized partial pipeline or smaller-model path, and public wording that accurately reflects those results.
