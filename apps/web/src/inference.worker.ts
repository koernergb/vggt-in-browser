import * as ort from 'onnxruntime-web/webgpu';

let vggtSession: ort.InferenceSession | undefined;

async function requireAsset(url: string, label: string) {
  try {
    const response = await fetch(url, {method: 'HEAD'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(
      `${label} is unavailable (${detail}). The page is still open, but its local model server may have stopped. Run "npm run dev -- --port 5174 --strictPort" from the repository and retry.`,
    );
  }
}

async function getVggtSession() {
  if (!vggtSession) {
    await requireAsset('/local-model/vggt-camera-depth-aggregator-int8.onnx', 'VGGT graph');
    await requireAsset('/local-model/vggt-camera-depth-aggregator-int8.onnx.data', 'VGGT external weights');
    vggtSession = await ort.InferenceSession.create('/local-model/vggt-camera-depth-aggregator-int8.onnx', {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      externalData: [{
        path: 'vggt-camera-depth-aggregator-int8.onnx.data',
        data: '/local-model/vggt-camera-depth-aggregator-int8.onnx.data',
      }],
    });
  }
  return vggtSession;
}

self.onmessage = async ({data}: MessageEvent<{type: string}>) => {
  if (data.type === 'run-parity') {
    try {
      await requireAsset('/local-parity/manifest.json', 'Browser parity manifest');
      const manifestResponse = await fetch('/local-parity/manifest.json');
      if (!manifestResponse.ok) throw new Error(`Parity manifest returned HTTP ${manifestResponse.status}`);
      const manifest = await manifestResponse.json() as {
        input: {shape: number[]; file: string}; outputs: Record<string, {shape: number[]; file: string}>;
      };
      const inputResponse = await fetch(`/local-parity/${manifest.input.file}`);
      if (!inputResponse.ok) throw new Error(`Parity input returned HTTP ${inputResponse.status}`);
      const inputBuffer: ArrayBuffer = await inputResponse.arrayBuffer();
      const inputData = new Float32Array(inputBuffer);
      const session = await getVggtSession();
      const started = performance.now();
      const actual = await session.run({images: new ort.Tensor('float32', inputData, manifest.input.shape)});
      const inferenceMs = performance.now() - started;
      const comparisons: Record<string, {maxAbs: number; meanAbs: number; finite: boolean}> = {};
      const browserGeometry: Record<string, Float32Array> = {};
      const referenceGeometry: Record<string, Float32Array> = {};
      const transfers: ArrayBuffer[] = [];
      for (const [name, metadata] of Object.entries(manifest.outputs)) {
        const expectedResponse = await fetch(`/local-parity/${metadata.file}`);
        if (!expectedResponse.ok) throw new Error(`${name} reference returned HTTP ${expectedResponse.status}`);
        const expectedBuffer: ArrayBuffer = await expectedResponse.arrayBuffer();
        const expected = new Float32Array(expectedBuffer);
        const output = await actual[name].getData();
        if (!(output instanceof Float32Array)) throw new Error(`${name} was not float32`);
        let maxAbs = 0;
        let sumAbs = 0;
        let finite = true;
        for (let index = 0; index < output.length; index++) {
          const difference = Math.abs(output[index] - expected[index]);
          maxAbs = Math.max(maxAbs, difference);
          sumAbs += difference;
          finite &&= Number.isFinite(output[index]);
        }
        comparisons[name] = {maxAbs, meanAbs: sumAbs / output.length, finite};
        browserGeometry[name] = output;
        referenceGeometry[name] = expected;
        transfers.push(output.buffer as ArrayBuffer, expected.buffer as ArrayBuffer);
      }
      self.postMessage(
        {type: 'parity-success', inferenceMs, comparisons, browserGeometry, referenceGeometry},
        {transfer: transfers},
      );
    } catch (error) {
      self.postMessage({type: 'error', target: 'vggt', message: error instanceof Error ? error.stack ?? error.message : String(error)});
    }
    return;
  }
  if (data.type === 'run-vggt') {
    try {
      const sessionStarted = performance.now();
      const session = await getVggtSession();
      const sessionMs = performance.now() - sessionStarted;
      const input = new ort.Tensor('float32', new Float32Array(1 * 2 * 3 * 518 * 518), [1, 2, 3, 518, 518]);
      const inferenceStarted = performance.now();
      const values = await session.run({images: input});
      const inferenceMs = performance.now() - inferenceStarted;
      const outputs: Record<string, {dims: readonly number[]; finite: boolean}> = {};
      for (const [name, tensor] of Object.entries(values)) {
        const tensorData = await tensor.getData();
        outputs[name] = {
          dims: tensor.dims,
          finite: tensorData instanceof Float32Array && tensorData.every(Number.isFinite),
        };
      }
      self.postMessage({type: 'vggt-success', sessionMs, inferenceMs, outputs});
    } catch (error) {
      self.postMessage({type: 'error', target: 'vggt', message: error instanceof Error ? error.stack ?? error.message : String(error)});
    }
    return;
  }
  if (data.type !== 'run-smoke') return;
  try {
    const sessionStarted = performance.now();
    const session = await ort.InferenceSession.create('/models/smoke-add.onnx', {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
    });
    const sessionMs = performance.now() - sessionStarted;
    const input = new ort.Tensor('float32', new Float32Array([1, 2, 3, 4]), [1, 4]);
    const inferenceStarted = performance.now();
    const outputs = await session.run({input});
    const inferenceMs = performance.now() - inferenceStarted;
    const outputData = await outputs.output.getData();
    if (!(outputData instanceof Float32Array)) throw new Error('Smoke output was not float32');
    const output = Array.from(outputData);
    self.postMessage({type: 'smoke-success', output, sessionMs, inferenceMs});
    await session.release();
  } catch (error) {
    self.postMessage({type: 'error', target: 'smoke', message: error instanceof Error ? error.stack ?? error.message : String(error)});
  }
};
