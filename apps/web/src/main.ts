import './style.css';
import {createPly, renderComparison, renderOrbit, type GeometryInputs, type ViewState} from './geometry';
import {preprocessImages, validateFiles} from './preprocess';

type WorkerResult =
  | {type: 'smoke-success'; output: number[]; sessionMs: number; inferenceMs: number}
  | {type: 'vggt-success'; sessionMs: number; inferenceMs: number; outputs: Record<string, {dims: readonly number[]; finite: boolean}>}
  | {type: 'parity-success'; inferenceMs: number; comparisons: Record<string, {maxAbs: number; meanAbs: number; finite: boolean}>; browserGeometry: GeometryInputs; referenceGeometry: GeometryInputs}
  | {type: 'user-success'; sessionMs: number; inferenceMs: number; geometry: GeometryInputs}
  | {type: 'error'; target: 'smoke' | 'vggt' | 'user'; message: string};

const diagnostics = document.querySelector<HTMLDListElement>('#diagnostics')!;
const gpuBadge = document.querySelector<HTMLSpanElement>('#gpu-badge')!;
const runBadge = document.querySelector<HTMLSpanElement>('#run-badge')!;
const runButton = document.querySelector<HTMLButtonElement>('#run')!;
const result = document.querySelector<HTMLPreElement>('#result')!;
const vggtBadge = document.querySelector<HTMLSpanElement>('#vggt-badge')!;
const vggtButton = document.querySelector<HTMLButtonElement>('#run-vggt')!;
const vggtResult = document.querySelector<HTMLPreElement>('#vggt-result')!;
const parityButton = document.querySelector<HTMLButtonElement>('#run-parity')!;
const geometrySection = document.querySelector<HTMLElement>('#geometry')!;
const geometryCanvas = document.querySelector<HTMLCanvasElement>('#geometry-canvas')!;
const confidence = document.querySelector<HTMLInputElement>('#confidence')!;
const confidenceValue = document.querySelector<HTMLOutputElement>('#confidence-value')!;
const projection = document.querySelector<HTMLSelectElement>('#projection')!;
const geometryStats = document.querySelector<HTMLParagraphElement>('#geometry-stats')!;
let lastGeometry: {reference: GeometryInputs; browser: GeometryInputs} | undefined;
const imageInput = document.querySelector<HTMLInputElement>('#images')!;
const previews = document.querySelector<HTMLDivElement>('#previews')!;
const userBadge = document.querySelector<HTMLSpanElement>('#user-badge')!;
const userButton = document.querySelector<HTMLButtonElement>('#run-user')!;
const exampleButton = document.querySelector<HTMLButtonElement>('#run-example')!;
const cancelButton = document.querySelector<HTMLButtonElement>('#cancel-user')!;
const userResult = document.querySelector<HTMLParagraphElement>('#user-result')!;
const userSection = document.querySelector<HTMLElement>('#user-geometry')!;
const userCanvas = document.querySelector<HTMLCanvasElement>('#user-canvas')!;
const userConfidence = document.querySelector<HTMLInputElement>('#user-confidence')!;
const userConfidenceValue = document.querySelector<HTMLOutputElement>('#user-confidence-value')!;
const pointSize = document.querySelector<HTMLInputElement>('#point-size')!;
const userStats = document.querySelector<HTMLParagraphElement>('#user-stats')!;
let userGeometry: GeometryInputs | undefined;
let viewState: ViewState = {yaw: -.7, pitch: -.45, zoom: 2};
let smokePassed = false;
let previewUrls: string[] = [];

function refreshUserGeometry() {
  if (!userGeometry) return;
  const keep = Number(userConfidence.value);
  userConfidenceValue.value = `${keep}%`;
  const stats = renderOrbit(userCanvas, userGeometry, keep, Number(pointSize.value), viewState);
  userStats.textContent = `${stats.pointCount.toLocaleString()} sampled points · confidence cutoff ${stats.threshold.toFixed(3)} · drag to orbit, wheel to zoom`;
}

function refreshGeometry() {
  if (!lastGeometry) return;
  const keepPercent = Number(confidence.value);
  confidenceValue.value = `${keepPercent}%`;
  const stats = renderComparison(
    geometryCanvas,
    lastGeometry.reference,
    lastGeometry.browser,
    keepPercent,
    projection.value as 'xy' | 'xz' | 'yz',
  );
  geometryStats.textContent = `Keeping the top ${keepPercent}% confidence per model · ${stats.pointCounts[0].toLocaleString()} native points · ${stats.pointCounts[1].toLocaleString()} browser points`;
}

function row(label: string, value: string) {
  diagnostics.insertAdjacentHTML('beforeend', `<dt>${label}</dt><dd>${value}</dd>`);
}

async function inspectGpu() {
  row('Secure context', String(window.isSecureContext));
  row('Cross-origin isolated', String(window.crossOriginIsolated));
  row('Browser', navigator.userAgent);
  if (!navigator.gpu) {
    gpuBadge.textContent = 'Unavailable';
    gpuBadge.className = 'badge fail';
    result.textContent = 'navigator.gpu is unavailable. Use a current WebGPU-capable browser.';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  if (!adapter) {
    gpuBadge.textContent = 'No adapter';
    gpuBadge.className = 'badge fail';
    result.textContent = 'WebGPU exists, but no adapter could be acquired.';
    return;
  }
  const info = adapter.info;
  row('Adapter', [info.vendor, info.architecture, info.device].filter(Boolean).join(' · ') || 'available');
  row('Max storage buffer', `${Math.round(adapter.limits.maxStorageBufferBindingSize / 1024 / 1024)} MiB`);
  row('Max buffer', `${Math.round(adapter.limits.maxBufferSize / 1024 / 1024)} MiB`);
  gpuBadge.textContent = 'Available';
  gpuBadge.className = 'badge pass';
  runButton.disabled = false;
  result.textContent = 'Ready to test ONNX Runtime Web.';
  runButton.click();
}

function createWorker() { return new Worker(new URL('./inference.worker.ts', import.meta.url), {type: 'module'}); }
let worker = createWorker();
function handleWorkerMessage({data}: MessageEvent<WorkerResult>) {
  if (data.type === 'user-success') {
    userGeometry = data.geometry;
    userButton.disabled = imageInput.files?.length !== 2; exampleButton.disabled = false; cancelButton.disabled = true;
    userBadge.textContent = 'Complete'; userBadge.className = 'badge pass';
    userResult.textContent = `Finished locally: model/session ${(data.sessionMs / 1000).toFixed(1)} s, inference ${(data.inferenceMs / 1000).toFixed(1)} s.`;
    userSection.hidden = false; refreshUserGeometry();
    return;
  }
  if (data.type === 'parity-success') {
    parityButton.disabled = false;
    vggtBadge.textContent = 'Parity measured';
    vggtBadge.className = 'badge pass';
    vggtResult.textContent = JSON.stringify(
      data,
      (key, value) => key.endsWith('Geometry') ? '[transferred tensor data]' : value,
      2,
    );
    lastGeometry = {reference: data.referenceGeometry, browser: data.browserGeometry};
    geometrySection.hidden = false;
    refreshGeometry();
    return;
  }
  if (data.type === 'vggt-success') {
    vggtButton.disabled = false;
    vggtBadge.textContent = 'Passed';
    vggtBadge.className = 'badge pass';
    vggtResult.textContent = JSON.stringify(data, null, 2);
    parityButton.disabled = false;
    return;
  }
  if (data.type === 'error') {
    const badge = data.target === 'smoke' ? runBadge : data.target === 'user' ? userBadge : vggtBadge;
    const output = data.target === 'smoke' ? result : data.target === 'user' ? userResult : vggtResult;
    badge.textContent = 'Failed';
    badge.className = 'badge fail';
    output.textContent = data.message;
    if (data.target === 'smoke') runButton.disabled = false;
    else if (data.target === 'user') { userButton.disabled = imageInput.files?.length !== 2; exampleButton.disabled = false; cancelButton.disabled = true; }
    else vggtButton.disabled = false;
    return;
  }
  runButton.disabled = false;
  const correct = JSON.stringify(data.output) === JSON.stringify([3, 5, 7, 9]);
  runBadge.textContent = correct ? 'Passed' : 'Wrong output';
  runBadge.className = correct ? 'badge pass' : 'badge fail';
  result.textContent = JSON.stringify({...data, expected: [3, 5, 7, 9], correct}, null, 2);
  if (correct) vggtButton.disabled = false;
  smokePassed = correct;
  if (correct) {
    exampleButton.disabled = false;
    if (imageInput.files?.length === 2) userButton.disabled = false;
  }
}
worker.onmessage = handleWorkerMessage;

function startUserRun(input: Float32Array, detail: string) {
  userButton.disabled = true; exampleButton.disabled = true; cancelButton.disabled = false;
  userBadge.textContent = 'Running…'; userBadge.className = 'badge neutral';
  userResult.textContent = detail;
  worker.postMessage({type: 'run-user', input}, [input.buffer]);
}

imageInput.addEventListener('change', () => {
  previewUrls.forEach(URL.revokeObjectURL); previewUrls = [];
  previews.replaceChildren(); userButton.disabled = true;
  try {
    const files = Array.from(imageInput.files ?? []); validateFiles(files);
    files.forEach((file, index) => {
      const url = URL.createObjectURL(file); previewUrls.push(url);
      const figure = document.createElement('figure');
      const image = document.createElement('img'); image.src = url; image.alt = `Selected view ${index + 1}`;
      const caption = document.createElement('figcaption'); caption.textContent = `View ${index + 1} · ${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MiB`;
      figure.append(image, caption); previews.append(figure);
    });
    userBadge.textContent = 'Ready'; userBadge.className = 'badge neutral';
    userResult.textContent = smokePassed ? 'Ready for local reconstruction.' : 'Images are valid. Run the WebGPU smoke test first.';
    userButton.disabled = !smokePassed;
  } catch (error) {
    userBadge.textContent = 'Invalid'; userBadge.className = 'badge fail';
    userResult.textContent = error instanceof Error ? error.message : String(error);
  }
});

userButton.addEventListener('click', async () => {
  try {
    userButton.disabled = true; cancelButton.disabled = false;
    userBadge.textContent = 'Preprocessing…'; userBadge.className = 'badge neutral';
    const files = Array.from(imageInput.files ?? []); const prepared = await preprocessImages(files);
    userResult.textContent = prepared.summaries.map((item, index) => `View ${index + 1}: ${item.width}×${item.height} → ${item.resizedWidth}×${item.resizedHeight}, centered in 518×518`).join(' · ');
    startUserRun(prepared.tensor, userResult.textContent ?? 'Running local reconstruction.');
  } catch (error) {
    userBadge.textContent = 'Failed'; userBadge.className = 'badge fail';
    userResult.textContent = error instanceof Error ? error.message : String(error); userButton.disabled = false; cancelButton.disabled = true;
  }
});

exampleButton.addEventListener('click', async () => {
  try {
    userBadge.textContent = 'Loading fixture…'; userBadge.className = 'badge neutral';
    const response = await fetch('/local-parity/images.f32');
    if (!response.ok) throw new Error(`Golden fixture is not installed (HTTP ${response.status}). Generate the ignored browser parity bundle first.`);
    const input = new Float32Array(await response.arrayBuffer());
    startUserRun(input, 'Running the installed two-view kitchen fixture locally.');
  } catch (error) {
    userBadge.textContent = 'Unavailable'; userBadge.className = 'badge fail';
    userResult.textContent = error instanceof Error ? error.message : String(error); exampleButton.disabled = false;
  }
});

cancelButton.addEventListener('click', () => {
  worker.terminate();
  worker = createWorker(); worker.onmessage = handleWorkerMessage;
  userBadge.textContent = 'Cancelled'; userBadge.className = 'badge neutral';
  userResult.textContent = 'Inference was cancelled and its worker state discarded. You may start another run.';
  userButton.disabled = imageInput.files?.length !== 2; exampleButton.disabled = !smokePassed; cancelButton.disabled = true;
});

userConfidence.addEventListener('input', refreshUserGeometry); pointSize.addEventListener('input', refreshUserGeometry);
document.querySelector('#reset-view')!.addEventListener('click', () => { viewState = {yaw: -.7, pitch: -.45, zoom: 2}; refreshUserGeometry(); });
document.querySelector('#export-ply')!.addEventListener('click', () => {
  if (!userGeometry) return;
  const blob = new Blob([createPly(userGeometry, Number(userConfidence.value))], {type: 'application/octet-stream'});
  const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'vggt-reconstruction.ply'; anchor.click(); URL.revokeObjectURL(url);
});
let dragging = false, lastX = 0, lastY = 0;
userCanvas.addEventListener('pointerdown', event => { dragging = true; lastX = event.clientX; lastY = event.clientY; userCanvas.setPointerCapture(event.pointerId); });
userCanvas.addEventListener('pointermove', event => { if (!dragging) return; viewState.yaw += (event.clientX - lastX) * .008; viewState.pitch = Math.max(-1.5, Math.min(1.5, viewState.pitch + (event.clientY - lastY) * .008)); lastX = event.clientX; lastY = event.clientY; refreshUserGeometry(); });
userCanvas.addEventListener('pointerup', () => { dragging = false; });
userCanvas.addEventListener('wheel', event => { event.preventDefault(); viewState.zoom = Math.max(.25, Math.min(5, viewState.zoom * Math.exp(-event.deltaY * .001))); refreshUserGeometry(); }, {passive: false});

runButton.addEventListener('click', () => {
  runButton.disabled = true;
  runBadge.textContent = 'Running…';
  runBadge.className = 'badge neutral';
  result.textContent = 'Creating a WebGPU-only inference session…';
  worker.postMessage({type: 'run-smoke'});
});

vggtButton.addEventListener('click', () => {
  vggtButton.disabled = true;
  vggtBadge.textContent = 'Loading…';
  vggtBadge.className = 'badge neutral';
  vggtResult.textContent = 'Loading 1.8 GiB of local external weights. This can take several minutes.';
  worker.postMessage({type: 'run-vggt'});
});

parityButton.addEventListener('click', () => {
  parityButton.disabled = true;
  vggtBadge.textContent = 'Running fixture…';
  vggtResult.textContent = 'Running the two-view kitchen fixture and comparing against native INT8.';
  worker.postMessage({type: 'run-parity'});
});

confidence.addEventListener('input', refreshGeometry);
projection.addEventListener('change', refreshGeometry);

inspectGpu().catch((error: unknown) => {
  gpuBadge.textContent = 'Error';
  gpuBadge.className = 'badge fail';
  result.textContent = error instanceof Error ? error.message : String(error);
});
