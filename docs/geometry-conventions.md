# Geometry conventions

The browser decodes VGGT's 9-value camera encoding as translation (3), an XYZW
quaternion (4), and vertical/horizontal field-of-view angles (2). The decoded
rotation and translation are world-to-camera extrinsics. Camera centers are
computed as `-Rᵀt`; depth pixels are first unprojected in camera coordinates and
then transformed to world coordinates with `Rᵀ(point - t)`.

Image coordinates use +X right and +Y down. Camera-space depth is +Z. No
unrecorded mirroring, axis swapping, scale normalization, or scene-specific
alignment is applied. VGGT depth supplies relative scene scale, not metric
units, so exported PLY coordinates must not be interpreted as meters.

The canvas viewer preserves VGGT's downward-positive Y sign when mapping to
canvas coordinates, which are also downward-positive. Orbiting changes only the
view transform; the underlying geometry and PLY coordinates are not rewritten.

Confidence filtering is percentile-based independently for each result. This is
intentional because WebGPU confidence values have measured numeric drift from
native INT8. The viewer samples every third pixel; PLY export samples every
second pixel. Each retained point uses its corresponding source-image pixel
color. Invalid and non-positive depths are discarded.

The current model artifact has a fixed `[1,2,3,518,518]` input. Consequently the
M3 UI deliberately requires exactly two views. Supporting three or four views
requires a separately exported model artifact and is a human product-scope gate.
