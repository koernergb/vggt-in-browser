import assert from 'node:assert/strict';
import test from 'node:test';
import {createPly, decodeCameras, unproject, type GeometryInputs} from './geometry.ts';

function fixture(): GeometryInputs {
  const pixels = 2 * 518 * 518;
  const pose = new Float32Array(18);
  for (let view = 0; view < 2; view++) {
    pose[view * 9 + 6] = 1;
    pose[view * 9 + 7] = 1;
    pose[view * 9 + 8] = 1;
  }
  return {pose_enc: pose, depth: new Float32Array(pixels).fill(2), depth_conf: new Float32Array(pixels).fill(1)};
}

test('identity pose decodes camera center at origin', () => {
  assert.ok(decodeCameras(fixture().pose_enc)[0].center.every(value => value === 0));
});

test('unprojection returns finite points for both views', () => {
  const result = unproject(fixture(), 100, 259);
  assert.equal(result.points.length, 8);
  assert.deepEqual(new Set(result.points.map(point => point.view)), new Set([0, 1]));
  assert.ok(result.points.every(point => Number.isFinite(point.x + point.y + point.z)));
});

test('PLY export declares and writes the same vertex count', () => {
  const ply = createPly(fixture(), 100);
  const declared = Number(ply.match(/element vertex (\d+)/)?.[1]);
  assert.equal(ply.trim().split('\n').length - 11, declared);
});
