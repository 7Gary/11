"""Fractal attention fusion module for PatchCore."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class FractalAttentionConfig:
    """Configuration for the fractal attention fusion module."""

    fractal_order: int = 3
    attention_dropout: float = 0.0
    bias_temperature: float = 2.0
    residual_mix: float = 1.0


class _FractalMaskGenerator(torch.nn.Module):
    """Generate fractal-inspired masks via multi-scale box counting."""

    def __init__(self, fractal_order: int, eps: float = 1e-6) -> None:
        super().__init__()
        if fractal_order < 1:
            raise ValueError("fractal_order must be >= 1")
        self.fractal_order = fractal_order
        self.eps = eps

    def forward(
        self, features: torch.Tensor, patch_shape: Tuple[int, int]
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("Expected [B, N, D] tensor for fractal mask generation")
        batchsize, num_patches, _ = features.shape
        patch_h, patch_w = patch_shape
        if patch_h * patch_w != num_patches:
            raise ValueError("patch_shape does not match the number of patches")

        feature_map = features.view(batchsize, patch_h, patch_w, -1)
        feature_map = feature_map.permute(0, 3, 1, 2).contiguous()
        energy = feature_map.pow(2).mean(dim=1, keepdim=True)
        mean_energy = energy.mean(dim=(-1, -2), keepdim=True)
        std_energy = energy.std(dim=(-1, -2), keepdim=True).clamp_min(self.eps)

        fractal_maps = []
        max_kernel = max(1, min(patch_h, patch_w))
        for level in range(self.fractal_order):
            kernel = min(max(1, 2**level), max_kernel)
            padding = kernel // 2
            pooled = F.avg_pool2d(
                energy, kernel_size=kernel, stride=1, padding=padding, count_include_pad=False
            )
            if pooled.shape[-2:] != (patch_h, patch_w):
                pooled = self._match_spatial_size(pooled, patch_h, patch_w)
            occupancy = torch.sigmoid((pooled - mean_energy) / std_energy)
            fractal_maps.append(occupancy)

        stacked = torch.stack(fractal_maps, dim=0).mean(dim=0)
        stacked = stacked.squeeze(1)
        return stacked.view(batchsize, num_patches)

    @staticmethod
    def _match_spatial_size(
        tensor: torch.Tensor, target_h: int, target_w: int
    ) -> torch.Tensor:
        """Center-crop or pad to match the requested spatial resolution."""
        _, _, h, w = tensor.shape

        if h > target_h:
            start = (h - target_h) // 2
            tensor = tensor[..., start : start + target_h, :]
            if tensor.shape[-2] > target_h:
                tensor = tensor[..., :target_h, :]
        elif h < target_h:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            tensor = F.pad(tensor, (0, 0, pad_top, pad_bottom))

        _, _, h, w = tensor.shape
        if w > target_w:
            start = (w - target_w) // 2
            tensor = tensor[..., :, start : start + target_w]
            if tensor.shape[-1] > target_w:
                tensor = tensor[..., :, :target_w]
        elif w < target_w:
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            tensor = F.pad(tensor, (pad_left, pad_right, 0, 0))

        return tensor


class FractalAttentionFusion(torch.nn.Module):
    """Fractal Attention Fusion module that replaces yarn voxel mapping."""

    def __init__(self, feature_dim: int, config: FractalAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or FractalAttentionConfig()
        self.feature_dim = feature_dim
        self.mask_generator = _FractalMaskGenerator(self.config.fractal_order)

        self.query_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.key_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.value_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.output_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.dropout = torch.nn.Dropout(self.config.attention_dropout)

        self.register_buffer("last_fractal_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_fractal_coherence", torch.tensor(0.0), persistent=False)

        self.last_fractal_map: torch.Tensor | None = None
        self.last_normalized_map: torch.Tensor | None = None
        self.last_patch_shape: Tuple[int, int] | None = None

    def forward(
        self,
        flat_features: torch.Tensor,
        patch_shape: Tuple[int, int],
        batchsize: int,
    ) -> torch.Tensor:
        if flat_features.ndim != 2:
            raise ValueError("Expected flattened [B*N, D] patch embeddings")
        if batchsize <= 0:
            raise ValueError("batchsize must be positive")

        num_patches = patch_shape[0] * patch_shape[1]
        expected = batchsize * num_patches
        if flat_features.shape[0] != expected:
            raise ValueError(
                "Flat features do not match patch grid dimensions: "
                f"expected {expected}, got {flat_features.shape[0]}"
            )

        features = flat_features.view(batchsize, num_patches, self.feature_dim)
        fractal_scores = self.mask_generator(features, patch_shape)
        normalized_scores = fractal_scores - fractal_scores.mean(dim=-1, keepdim=True)
        normalized_scores = normalized_scores / (
            fractal_scores.std(dim=-1, keepdim=True).clamp_min(1e-6)
        )

        queries = self.query_proj(features)
        keys = self.key_proj(features)
        values = self.value_proj(features)

        attn_logits = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.feature_dim)
        bias = -torch.abs(
            normalized_scores.unsqueeze(2) - normalized_scores.unsqueeze(1)
        ) * self.config.bias_temperature
        attn_logits = attn_logits + bias

        attention = torch.softmax(attn_logits, dim=-1)
        attention = self.dropout(attention)
        fused = torch.matmul(attention, values)
        fused = self.output_proj(fused)
        fused = fused * self.config.residual_mix
        output = features + fused

        with torch.no_grad():
            coherence_matrix = 1.0 - torch.abs(
                normalized_scores.unsqueeze(2) - normalized_scores.unsqueeze(1)
            )
            coherence_score = (attention * coherence_matrix).sum(dim=(-2, -1)) / attention.sum(
                dim=(-2, -1)
            ).clamp_min(1e-6)
            self.last_fractal_mean.copy_(fractal_scores.mean().detach())
            self.last_fractal_coherence.copy_(coherence_score.mean().detach())

            self.last_patch_shape = patch_shape
            self.last_fractal_map = fractal_scores.detach().cpu().view(batchsize, *patch_shape)
            self.last_normalized_map = (
                normalized_scores.detach().cpu().view(batchsize, *patch_shape)
            )

        return output.view_as(flat_features)

    def get_last_fractal_map(self, normalized: bool = True) -> torch.Tensor | None:
        """Return the most recent fractal attention map."""

        if normalized:
            return None if self.last_normalized_map is None else self.last_normalized_map.clone()
        return None if self.last_fractal_map is None else self.last_fractal_map.clone()