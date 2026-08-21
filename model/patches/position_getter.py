"""ONNX-compatible replacement for VGGT's RoPE PositionGetter."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


class OnnxPositionGetter:
    """Generate the same row-major `[y, x]` grid without `cartesian_prod`."""

    def __init__(self) -> None:
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
            self.position_cache[height, width] = torch.stack(
                (yy.reshape(-1), xx.reshape(-1)), dim=-1
            )
        positions = self.position_cache[height, width]
        return positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()
