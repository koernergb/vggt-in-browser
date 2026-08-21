"""ONNX-safe DPT sinusoidal position embedding helpers."""

from __future__ import annotations

import torch


def make_sincos_pos_embed(
    embed_dim: int, pos: torch.Tensor, omega_0: float = 100
) -> torch.Tensor:
    """Match VGGT's embedding while keeping both Einsum inputs the same dtype."""
    if embed_dim % 2:
        raise ValueError("embed_dim must be even")
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / torch.pow(pos.new_tensor(omega_0), omega)
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def position_grid_to_embed(
    pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100
) -> torch.Tensor:
    height, width, grid_dim = pos_grid.shape
    if grid_dim != 2:
        raise ValueError("position grid's final dimension must be 2")
    flat = pos_grid.reshape(-1, grid_dim)
    emb_x = make_sincos_pos_embed(embed_dim // 2, flat[:, 0], omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, flat[:, 1], omega_0)
    return torch.cat([emb_x, emb_y], dim=-1).view(height, width, embed_dim)
