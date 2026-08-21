import * as ort from 'onnxruntime-web/webgpu';

let vggtSession: ort.InferenceSession | undefined;

async function getVggtSession() {
  if (!vggtSession) {
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
      const manifest = await fetch('/local-parity/manifest.json').then(response => response.json()) as {
        input: {shape: number[]; file: string}; outputs: Record<string, {shape: number[]; file: string}>;
      };
      const inputResponse = await fetch(`/local-parity/${manifest.input.file}`);
      const inputBuffer: ArrayBuffer = await inputResponse.arrayBuffer();
      const inputData = new Float32Array(inputBuffer);
      const session = await getVggtSession();
      const started = performance.now();
      const actual = await session.run({images: new ort.Tensor('float32', inputData, manifest.input.shape)});
      const inferenceMs = performance.now() - started;
      const comparisons: Record<string, {maxAbs: number; meanAbs: number; finite: boolean}> = {};
      for (const [name, metadata] of Object.entries(manifest.outputs)) {
        const expectedResponse = await fetch(`/local-parity/${metadata.file}`);
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
      }
      self.postMessage({type: 'parity-success', inferenceMs, comparisons});
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
