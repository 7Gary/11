"""轻量级 LSNet 结构，复现 “See Large, Focus Small” 的大/小尺度协同。.

参考 https://github.com/THU-MIG/lsnet 的核心思路，模块设计与论文保持一致：
* **Large** 分支：使用多尺度空洞深度卷积 + 逐点卷积拓展感受野，模拟大核卷积捕获织物周期纹理；
* **Small** 分支：标准 3×3 深度卷积后接通道注意力（SE），强化局部突变；
* **Fusion**：将各分支特征与原始特征级联，经 BN + ReLU 的 1×1 融合得到补强后的 patch 嵌入，
  并导出归一化注意力热力图用于可视化。
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn


class _DepthwiseDilatedConv(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fc(self.pool(x))
        return x * weight, weight


class LSNetFusion(nn.Module):
    """基于 LSNet 的多尺度特征融合。

    Args:
        embed_dim: 输入/输出的特征维度。
        dilation_rates: Large 分支使用的空洞率配置。
        reduction: 融合时的通道压缩比例。
    """

    def __init__(
        self,
        embed_dim: int,
        dilation_rates: Iterable[int] = (3, 5),
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.large_branch = nn.ModuleList(
            [_DepthwiseDilatedConv(embed_dim, rate) for rate in dilation_rates if rate > 1]
        )
        self.small_depthwise = nn.Sequential(
            nn.Conv2d(
                embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.small_project = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.small_bn = nn.BatchNorm2d(embed_dim)
        self.small_activation = nn.ReLU(inplace=True)
        self.channel_attention = _ChannelAttention(embed_dim, reduction=max(reduction, 1))

        fusion_channels = embed_dim * (len(self.large_branch) + 2)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
        )

        self.attention_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1, bias=True),
        )

    def forward(
        self, features: torch.Tensor, patch_shape: Tuple[int, int], batchsize: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行 LSNet 融合。

        Args:
            features: 形状 ``(B * H * W, C)`` 的 patch 嵌入。
            patch_shape: patch 网格尺寸 ``(H, W)``。
            batchsize: 当前批大小 ``B``。

        Returns:
            Tuple，其中第一个元素为融合后的展平特征，第二个元素为注意力热力图
            （形状 ``(B, H, W)``）。
        """

        if features.numel() == 0:
            return features, torch.zeros((batchsize, *patch_shape), device=features.device)

        height, width = patch_shape
        spatial = features.view(batchsize, height, width, self.embed_dim).permute(0, 3, 1, 2)

        large_feats = [conv(spatial) for conv in self.large_branch]

        small = self.small_depthwise(spatial)
        small = self.small_project(small)
        small = self.small_bn(small)
        small = self.small_activation(small)
        small, ca_map = self.channel_attention(small)

        fused = torch.cat([*large_feats, small, spatial], dim=1)
        fused = self.fusion(fused)
        fused = fused + spatial  # 残差保持原始 patch 表征

        attn = torch.sigmoid(self.attention_head(fused))
        fused = fused * (1.0 + attn)

        combined_map = attn * ca_map.mean(dim=1, keepdim=True)
        attn_map = combined_map.squeeze(1)
        fused_flat = fused.permute(0, 2, 3, 1).reshape(-1, self.embed_dim)
        return fused_flat, attn_map