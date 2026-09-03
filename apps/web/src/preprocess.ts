export const MODEL_SIZE = 518;
export const PATCH_SIZE = 14;
export const MAX_IMAGE_BYTES = 25 * 1024 * 1024;
export const ACCEPTED_IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);

export function paddedDimensions(width: number, height: number) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error('Image dimensions must be positive.');
  }
  if (width >= height) return {width: MODEL_SIZE, height: Math.round(height * MODEL_SIZE / width / PATCH_SIZE) * PATCH_SIZE};
  return {width: Math.round(width * MODEL_SIZE / height / PATCH_SIZE) * PATCH_SIZE, height: MODEL_SIZE};
}

export function validateFiles(files: readonly File[]) {
  if (files.length !== 2) throw new Error('Select exactly two overlapping images. This fixed-shape candidate currently supports two views.');
  for (const file of files) {
    if (!ACCEPTED_IMAGE_TYPES.has(file.type)) throw new Error(`${file.name}: use JPEG, PNG, or WebP.`);
    if (file.size === 0 || file.size > MAX_IMAGE_BYTES) throw new Error(`${file.name}: file must be between 1 byte and 25 MiB.`);
  }
}

async function decode(file: File) {
  try {
    return await createImageBitmap(file, {imageOrientation: 'from-image', premultiplyAlpha: 'premultiply', colorSpaceConversion: 'default'});
  } catch {
    throw new Error(`${file.name}: the browser could not decode this image.`);
  }
}

export async function preprocessImages(files: readonly File[]) {
  validateFiles(files);
  const output = new Float32Array(1 * 2 * 3 * MODEL_SIZE * MODEL_SIZE);
  const plane = MODEL_SIZE * MODEL_SIZE;
  const summaries: Array<{width: number; height: number; resizedWidth: number; resizedHeight: number}> = [];
  for (let view = 0; view < files.length; view++) {
    const bitmap = await decode(files[view]);
    const resized = paddedDimensions(bitmap.width, bitmap.height);
    const canvas = new OffscreenCanvas(MODEL_SIZE, MODEL_SIZE);
    const context = canvas.getContext('2d', {willReadFrequently: true});
    if (!context) throw new Error('2D canvas preprocessing is unavailable.');
    context.fillStyle = '#fff';
    context.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    context.drawImage(bitmap, Math.floor((MODEL_SIZE - resized.width) / 2), Math.floor((MODEL_SIZE - resized.height) / 2), resized.width, resized.height);
    const rgba = context.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
    const viewOffset = view * 3 * plane;
    for (let pixel = 0; pixel < plane; pixel++) {
      output[viewOffset + pixel] = rgba[pixel * 4] / 255;
      output[viewOffset + plane + pixel] = rgba[pixel * 4 + 1] / 255;
      output[viewOffset + 2 * plane + pixel] = rgba[pixel * 4 + 2] / 255;
    }
    summaries.push({width: bitmap.width, height: bitmap.height, resizedWidth: resized.width, resizedHeight: resized.height});
    bitmap.close();
  }
  return {tensor: output, summaries};
}
