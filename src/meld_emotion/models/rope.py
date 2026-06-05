"""Rotary positional embedding helpers for dialogue memory attention."""

from __future__ import annotations

import torch


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: int = 10000,
) -> torch.Tensor:
    """Apply RoPE to ``x`` using utterance positions.

    Args:
        x: Tensor with shape ``[B, N, D]``.
        positions: Tensor with shape ``[N]`` or ``[B, N]``.
        base: RoPE frequency base.
    """

    if x.ndim != 3:
        raise ValueError(f"x must have shape [B,N,D], got {tuple(x.shape)}")
    dim = int(x.shape[-1])
    if dim % 2 != 0:
        raise ValueError("RoPE requires an even last dimension")

    half = dim // 2
    pos = positions.to(device=x.device, dtype=x.dtype)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0).expand(x.shape[0], -1)
    if pos.shape != x.shape[:2]:
        raise ValueError(f"positions shape must be [N] or [B,N], got {tuple(positions.shape)}")

    idx = torch.arange(half, device=x.device, dtype=x.dtype)
    inv_freq = torch.pow(float(base), -idx / float(half))
    angles = pos.unsqueeze(-1) * inv_freq.view(1, 1, half)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = x_even * cos - x_odd * sin
    rotated[..., 1::2] = x_even * sin + x_odd * cos
    return rotated
