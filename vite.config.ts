import {defineConfig} from 'vite';
import {createReadStream, statSync} from 'node:fs';
import {resolve} from 'node:path';

const modelDirectory = resolve('model/artifacts/quantized-aggregator');
const parityDirectory = resolve('model/artifacts/browser-parity');

export default defineConfig({
  root: 'apps/web',
  build: {outDir: '../../dist', emptyOutDir: true},
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
  plugins: [{
    name: 'local-vggt-model',
    configureServer(server) {
      server.middlewares.use('/local-model/', (request, response, next) => {
        const name = request.url?.split('?')[0].replace(/^\//, '');
        if (!name || !/^vggt-camera-depth-aggregator-int8\.onnx(?:\.data)?$/.test(name)) return next();
        const path = resolve(modelDirectory, name);
        const stats = statSync(path);
        response.setHeader('Content-Type', 'application/octet-stream');
        response.setHeader('Content-Length', stats.size);
        response.setHeader('Cache-Control', 'no-store');
        if (request.method === 'HEAD') return response.end();
        createReadStream(path).pipe(response);
      });
      server.middlewares.use('/local-parity/', (request, response, next) => {
        const name = request.url?.split('?')[0].replace(/^\//, '');
        if (!name || !/^(?:manifest\.json|images\.f32|pose_enc\.f32|depth\.f32|depth_conf\.f32)$/.test(name)) return next();
        const path = resolve(parityDirectory, name);
        const stats = statSync(path);
        response.setHeader('Content-Type', name.endsWith('.json') ? 'application/json' : 'application/octet-stream');
        response.setHeader('Content-Length', stats.size);
        response.setHeader('Cache-Control', 'no-store');
        if (request.method === 'HEAD') return response.end();
        createReadStream(path).pipe(response);
      });
    },
  }],
});
