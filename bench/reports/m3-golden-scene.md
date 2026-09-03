# M3 golden-scene report

Status: automated implementation validation passed; human visual acceptance is
required.

## Measured run

- Date: 2026-09-02
- Machine: 16 GB M4 MacBook Pro
- Browser: Chromium 150 through the Codex in-app browser
- Adapter: Apple Metal 3
- Input: installed two-view VGGT kitchen fixture, pad-preprocessed to 518×518
- Execution provider: WebGPU only
- Cached/local session creation and model load: 8.0 s
- Inference: 51.1 s
- Default viewer sample: 26,876 points at the top 45% confidence
- Result values: finite
- Browser console errors: none

The upload-oriented inference message path, interactive viewer, confidence
filter, source-image point colors, camera centers/direction rays, orbit gesture,
reset control, and PLY-generation unit path are implemented. The installed
fixture button exercises the same worker result and viewer path as user images,
while bypassing browser image decoding so the exact reference tensor can remain
the deterministic regression input.

## Remaining human gate

Run the golden fixture and inspect the scene while orbiting and changing the
confidence filter. Explicitly confirm recognizability, plausible cameras,
absence of mirroring/inversion, and usable controls. Then supply or identify one
additional overlapping two-image scene with permission for local testing.
