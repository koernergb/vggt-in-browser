# M3 local-data privacy report

Status: implementation reviewed; production network capture pending final
browser validation.

## Data flow

Selected image `File` objects are decoded in the page with `createImageBitmap`.
Pixels are resized/padded in an `OffscreenCanvas`, converted to a float32 tensor,
and transferred directly to the local inference Web Worker. The worker sends
only model outputs back to the page. Preview URLs use `blob:` URLs and are
revoked when the selection changes.

There is no analytics, telemetry, crash reporting, remote logging, `FormData`,
image upload request, filename request payload, or server inference fallback.
The only application fetches are static application/model artifacts and the
explicit local parity fixture under `/local-*`. User reconstruction does not
invoke `fetch`.

Cancellation terminates the inference worker, discards its ONNX Runtime session,
and creates a clean worker so another run can start without reloading the page.

## Required verification

Before M3 completion, inspect production network traffic while selecting and
running non-fixture images. Confirm that no request contains image bytes,
filenames, thumbnails, blob URLs, or derived tensors. Record browser/version and
observed request URLs here. This is an objective human-verifiable release gate,
not permission to add telemetry.
