export type Camera = {rotation: number[]; translation: number[]; center: number[]; fx: number; fy: number};
export type GeometryInputs = {pose_enc: Float32Array; depth: Float32Array; depth_conf: Float32Array};
export type Point = {x: number; y: number; z: number; view: number};

const SIZE = 518;

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

export function unproject(inputs: GeometryInputs, keepPercent: number, stride = 4) {
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
        });
      }
    }
  }
  return {points, cameras, threshold};
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
