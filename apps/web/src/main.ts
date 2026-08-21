import './style.css';

type WorkerResult =
  | {type: 'smoke-success'; output: number[]; sessionMs: number; inferenceMs: number}
  | {type: 'vggt-success'; sessionMs: number; inferenceMs: number; outputs: Record<string, {dims: readonly number[]; finite: boolean}>}
  | {type: 'parity-success'; inferenceMs: number; comparisons: Record<string, {maxAbs: number; meanAbs: number; finite: boolean}>}
  | {type: 'error'; target: 'smoke' | 'vggt'; message: string};

const diagnostics = document.querySelector<HTMLDListElement>('#diagnostics')!;
const gpuBadge = document.querySelector<HTMLSpanElement>('#gpu-badge')!;
const runBadge = document.querySelector<HTMLSpanElement>('#run-badge')!;
const runButton = document.querySelector<HTMLButtonElement>('#run')!;
const result = document.querySelector<HTMLPreElement>('#result')!;
const vggtBadge = document.querySelector<HTMLSpanElement>('#vggt-badge')!;
const vggtButton = document.querySelector<HTMLButtonElement>('#run-vggt')!;
const vggtResult = document.querySelector<HTMLPreElement>('#vggt-result')!;
const parityButton = document.querySelector<HTMLButtonElement>('#run-parity')!;

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
}

const worker = new Worker(new URL('./inference.worker.ts', import.meta.url), {type: 'module'});
worker.onmessage = ({data}: MessageEvent<WorkerResult>) => {
  if (data.type === 'parity-success') {
    parityButton.disabled = false;
    vggtBadge.textContent = 'Parity measured';
    vggtBadge.className = 'badge pass';
    vggtResult.textContent = JSON.stringify(data, null, 2);
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
    const badge = data.target === 'smoke' ? runBadge : vggtBadge;
    const output = data.target === 'smoke' ? result : vggtResult;
    badge.textContent = 'Failed';
    badge.className = 'badge fail';
    output.textContent = data.message;
    if (data.target === 'smoke') runButton.disabled = false;
    else vggtButton.disabled = false;
    return;
  }
  runButton.disabled = false;
  const correct = JSON.stringify(data.output) === JSON.stringify([3, 5, 7, 9]);
  runBadge.textContent = correct ? 'Passed' : 'Wrong output';
  runBadge.className = correct ? 'badge pass' : 'badge fail';
  result.textContent = JSON.stringify({...data, expected: [3, 5, 7, 9], correct}, null, 2);
  if (correct) vggtButton.disabled = false;
};

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

inspectGpu().catch((error: unknown) => {
  gpuBadge.textContent = 'Error';
  gpuBadge.className = 'badge fail';
  result.textContent = error instanceof Error ? error.message : String(error);
});
