#!/usr/bin/env python3
"""Export the approved fixed-shape VGGT camera+depth slice to ONNX."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

PATCHES_DIR = Path(__file__).resolve().parents[1] / "patches"
if str(PATCHES_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PATCHES_DIR))

from position_getter import OnnxPositionGetter
from depth_position_embed import position_grid_to_embed


class CameraDepthExportWrapper(torch.nn.Module):
    """Flatten VGGT's nested/list output into three stable ONNX outputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.aggregator = model.aggregator
        self.camera_head = model.camera_head
        self.depth_head = model.depth_head

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aggregated_tokens, patch_start_idx = self.aggregator(images)
        pose_enc = self.camera_head(aggregated_tokens)[-1]
        depth, depth_conf = self.depth_head(
            aggregated_tokens, images=images, patch_start_idx=patch_start_idx
        )
        return pose_enc, depth, depth_conf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("model/artifacts/vggt-camera-depth.onnx"))
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument("--views", type=int, default=2, choices=(2,))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    from vggt.models.vggt import VGGT
    import vggt.heads.dpt_head as dpt_head

    # Upstream deliberately builds this frequency vector in float64 on CPU.
    # ONNX requires both Einsum inputs to share a type, so keep it in the
    # position grid's float32 dtype during export.
    dpt_head.position_grid_to_embed = position_grid_to_embed
    model = VGGT.from_pretrained(args.model_id).eval().cpu()
    # These heads are not part of the approved first slice. Release them before
    # tracing to reduce pressure on unified-memory development machines.
    model.point_head = None
    model.track_head = None
    model.aggregator.position_getter = OnnxPositionGetter()
    wrapper = CameraDepthExportWrapper(model).eval()
    del model
    gc.collect()
    example = torch.zeros((1, args.views, 3, 518, 518), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example,),
            str(args.output),
            input_names=["images"],
            output_names=["pose_enc", "depth", "depth_conf"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
            external_data=True,
        )

    manifest = {
        "schema_version": 1,
        "model_id": args.model_id,
        "opset": args.opset,
        "input_shape": list(example.shape),
        "outputs": ["pose_enc", "depth", "depth_conf"],
        "artifact": str(args.output),
        "status": "exported-not-yet-runtime-validated"
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Exported {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
