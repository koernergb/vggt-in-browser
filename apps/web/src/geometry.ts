export type Camera = {rotation: number[]; translation: number[]; center: number[]; fx: number; fy: number};
export type GeometryInputs = {pose_enc: Float32Array; depth: Float32Array; depth_conf: Float32Array; images?: Float32Array};
export type Point = {x: number; y: number; z: number; view: number; color?: [number, number, number]};

const SIZE = 518;
const unprojectionCache = new WeakMap<GeometryInputs, Map<string, ReturnType<typeof computeUnprojection>>>();

function quaternionMatrix(x: number, y: number, z: number, w: number) {
  const twoS = 2 / (x * x + y * y + z * z + w * w);
  return [
    1 - twoS * (y * y + z * z), twoS * (x * y - z * w), twoS * (x * z + y * w),
    twoS * (x * y + z * w), 1 - twoS * (x * x + z * z), twoS * (y * z - x * w),
    twoS * (x * z - y * w), twoS * (y * z + x * w), 1 - twoS * (x * x + y * y),
  ];
}

export function decodeCameras(pose: Float32Array): Camera[] {
  const cameras: Camera[] = [];
  for (let view = 0; view < 2; view++) {
    const offset = view * 9;
    const translation = Array.from(pose.slice(offset, offset + 3));
    const rotation = quaternionMatrix(pose[offset + 3], pose[offset + 4], pose[offset + 5], pose[offset + 6]);
    const center = [0, 1, 2].map(column => -(
      rotation[column] * translation[0] + rotation[3 + column] * translation[1] + rotation[6 + column] * translation[2]
    ));
    cameras.push({
      rotation, translation, center,
      fy: (SIZE / 2) / Math.tan(pose[offset + 7] / 2),
      fx: (SIZE / 2) / Math.tan(pose[offset + 8] / 2),
    });
  }
  return cameras;
}

function percentile(values: Float32Array, quantile: number) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(quantile * (sorted.length - 1))))];
}

function computeUnprojection(inputs: GeometryInputs, keepPercent: number, stride: number) {
  const cameras = decodeCameras(inputs.pose_enc);
  const threshold = percentile(inputs.depth_conf, 1 - keepPercent / 100);
  const points: Point[] = [];
  for (let view = 0; view < 2; view++) {
    const camera = cameras[view];
    const frameOffset = view * SIZE * SIZE;
    for (let v = 0; v < SIZE; v += stride) {
      for (let u = 0; u < SIZE; u += stride) {
        const index = frameOffset + v * SIZE + u;
        const depth = inputs.depth[index];
        if (!Number.isFinite(depth) || depth <= 0 || inputs.depth_conf[index] < threshold) continue;
        const cam = [(u - SIZE / 2) * depth / camera.fx, (v - SIZE / 2) * depth / camera.fy, depth];
        points.push({
          x: camera.center[0] + camera.rotation[0] * cam[0] + camera.rotation[3] * cam[1] + camera.rotation[6] * cam[2],
          y: camera.center[1] + camera.rotation[1] * cam[0] + camera.rotation[4] * cam[1] + camera.rotation[7] * cam[2],
          z: camera.center[2] + camera.rotation[2] * cam[0] + camera.rotation[5] * cam[1] + camera.rotation[8] * cam[2],
          view,
          color: inputs.images ? [0, 1, 2].map(channel => Math.round(inputs.images![view * 3 * SIZE * SIZE + channel * SIZE * SIZE + v * SIZE + u] * 255)) as [number, number, number] : undefined,
        });
      }
    }
  }
  return {points, cameras, threshold};
}

export function unproject(inputs: GeometryInputs, keepPercent: number, stride = 4) {
  let entries = unprojectionCache.get(inputs);
  if (!entries) { entries = new Map(); unprojectionCache.set(inputs, entries); }
  const key = `${keepPercent}:${stride}`;
  let result = entries.get(key);
  if (!result) { result = computeUnprojection(inputs, keepPercent, stride); entries.set(key, result); }
  return result;
}

export function renderComparison(canvas: HTMLCanvasElement, reference: GeometryInputs, browser: GeometryInputs, keepPercent: number, projection: 'xy' | 'xz' | 'yz') {
  const ref = unproject(reference, keepPercent);
  const candidate = unproject(browser, keepPercent);
  const axes = projection === 'xy' ? ['x', 'y'] : projection === 'xz' ? ['x', 'z'] : ['y', 'z'];
  const axis0 = axes[0] as 'x' | 'y' | 'z';
  const axis1 = axes[1] as 'x' | 'y' | 'z';
  const all = [...ref.points, ...candidate.points];
  const xs = all.map(point => point[axis0]).sort((a, b) => a - b);
  const ys = all.map(point => point[axis1]).sort((a, b) => a - b);
  const bounds = [xs[Math.floor(xs.length * .01)], xs[Math.floor(xs.length * .99)], ys[Math.floor(ys.length * .01)], ys[Math.floor(ys.length * .99)]];
  const context = canvas.getContext('2d')!;
  context.fillStyle = '#071019'; context.fillRect(0, 0, canvas.width, canvas.height);
  const panelWidth = canvas.width / 2;
  const colors = ['#35b9f1', '#ff8a3d'];
  const component = {x: 0, y: 1, z: 2};
  const draw = (geometry: typeof ref, panel: number, title: string) => {
    const left = panel * panelWidth;
    context.fillStyle = '#dbeaf2'; context.font = '700 15px system-ui'; context.fillText(title, left + 18, 25);
    for (const point of geometry.points) {
      const px = left + 15 + ((point[axis0] - bounds[0]) / Math.max(bounds[1] - bounds[0], 1e-8)) * (panelWidth - 30);
      const py = canvas.height - 18 - ((point[axis1] - bounds[2]) / Math.max(bounds[3] - bounds[2], 1e-8)) * (canvas.height - 55);
      if (px < left || px >= left + panelWidth || py < 35 || py >= canvas.height) continue;
      context.fillStyle = colors[point.view]; context.fillRect(px, py, 1.4, 1.4);
    }
    for (let index = 0; index < geometry.cameras.length; index++) {
      const center = geometry.cameras[index].center;
      const px = left + 15 + ((center[component[axis0]] - bounds[0]) / Math.max(bounds[1] - bounds[0], 1e-8)) * (panelWidth - 30);
      const py = canvas.height - 18 - ((center[component[axis1]] - bounds[2]) / Math.max(bounds[3] - bounds[2], 1e-8)) * (canvas.height - 55);
      context.fillStyle = '#fff'; context.beginPath(); context.arc(px, py, 5, 0, Math.PI * 2); context.fill(); context.fillText(`C${index}`, px + 7, py - 5);
    }
  };
  draw(ref, 0, 'Native INT8'); draw(candidate, 1, 'Browser WebGPU');
  context.strokeStyle = '#31505d'; context.beginPath(); context.moveTo(panelWidth, 0); context.lineTo(panelWidth, canvas.height); context.stroke();
  return {referenceThreshold: ref.threshold, browserThreshold: candidate.threshold, pointCounts: [ref.points.length, candidate.points.length]};
}

export type ViewState = {yaw: number; pitch: number; zoom: number};

export function renderOrbit(canvas: HTMLCanvasElement, inputs: GeometryInputs, keepPercent: number, pointSize: number, view: ViewState) {
  const geometry = unproject(inputs, keepPercent, 3);
  const context = canvas.getContext('2d')!;
  context.fillStyle = '#071019'; context.fillRect(0, 0, canvas.width, canvas.height);
  if (!geometry.points.length) return {pointCount: 0, threshold: geometry.threshold};
  const xs = geometry.points.map(point => point.x).sort((a, b) => a - b);
  const ys = geometry.points.map(point => point.y).sort((a, b) => a - b);
  const zs = geometry.points.map(point => point.z).sort((a, b) => a - b);
  const middle = (values: number[]) => (values[Math.floor(values.length * .01)] + values[Math.floor(values.length * .99)]) / 2;
  const center = [middle(xs), middle(ys), middle(zs)];
  const span = Math.max(xs[Math.floor(xs.length * .99)] - xs[Math.floor(xs.length * .01)], ys[Math.floor(ys.length * .99)] - ys[Math.floor(ys.length * .01)], zs[Math.floor(zs.length * .99)] - zs[Math.floor(zs.length * .01)], 1e-5);
  const cosineYaw = Math.cos(view.yaw), sineYaw = Math.sin(view.yaw), cosinePitch = Math.cos(view.pitch), sinePitch = Math.sin(view.pitch);
  const project = (x: number, y: number, z: number) => {
    const dx = x - center[0], dy = y - center[1], dz = z - center[2];
    const rotatedX = cosineYaw * dx - sineYaw * dz;
    const yawZ = sineYaw * dx + cosineYaw * dz;
    const rotatedY = cosinePitch * dy - sinePitch * yawZ;
    const scale = Math.min(canvas.width, canvas.height) * .72 * view.zoom / span;
    return [canvas.width / 2 + rotatedX * scale, canvas.height / 2 - rotatedY * scale];
  };
  const colors = ['#35b9f1', '#ff8a3d'];
  for (const point of geometry.points) {
    const [x, y] = project(point.x, point.y, point.z);
    if (x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) continue;
    context.fillStyle = point.color ? `rgb(${point.color.join(' ')})` : colors[point.view]; context.fillRect(x, y, pointSize, pointSize);
  }
  context.font = '700 15px system-ui';
  geometry.cameras.forEach((camera, index) => {
    const [x, y] = project(camera.center[0], camera.center[1], camera.center[2]);
    const length = span * .08;
    const forward = [camera.rotation[6], camera.rotation[7], camera.rotation[8]];
    const [tipX, tipY] = project(camera.center[0] + forward[0] * length, camera.center[1] + forward[1] * length, camera.center[2] + forward[2] * length);
    context.strokeStyle = '#fff'; context.lineWidth = 2; context.beginPath(); context.moveTo(x, y); context.lineTo(tipX, tipY); context.stroke();
    context.fillStyle = '#fff'; context.beginPath(); context.arc(x, y, 6, 0, Math.PI * 2); context.fill(); context.fillText(`C${index}`, x + 9, y - 7);
  });
  return {pointCount: geometry.points.length, threshold: geometry.threshold};
}

export function createPly(inputs: GeometryInputs, keepPercent: number) {
  const {points} = unproject(inputs, keepPercent, 2);
  const colors = [[53, 185, 241], [255, 138, 61]];
  const header = `ply\nformat ascii 1.0\ncomment VGGT browser local reconstruction\nelement vertex ${points.length}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`;
  return header + points.map(point => `${point.x} ${point.y} ${point.z} ${(point.color ?? colors[point.view]).join(' ')}`).join('\n') + '\n';
}
