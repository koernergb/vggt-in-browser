import assert from 'node:assert/strict';
import test from 'node:test';
import {paddedDimensions} from './preprocess.ts';

test('landscape padding dimensions match VGGT patch rounding', () => {
  assert.deepEqual(paddedDimensions(779, 520), {width: 518, height: 350});
});

test('portrait padding dimensions remain patch divisible', () => {
  assert.deepEqual(paddedDimensions(520, 779), {width: 350, height: 518});
});

test('square images remain square', () => {
  assert.deepEqual(paddedDimensions(800, 800), {width: 518, height: 518});
});

test('invalid dimensions fail explicitly', () => {
  assert.throws(() => paddedDimensions(0, 20), /positive/);
});
