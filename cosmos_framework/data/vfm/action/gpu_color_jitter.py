# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GPU-accelerated ColorJitter using torchvision.transforms.v2.

Replaces the CPU-bound torchvision.transforms.ColorJitter in DataLoader workers.
v2.ColorJitter natively supports CUDA tensors, producing results identical to
the CPU version (same RGB↔HSV conversion logic) but running on GPU (~3ms vs ~1.7s).

Usage:
    export COSMOS_GPU_COLOR_JITTER=1  # enables GPU path
    # DataLoader only does RandomCrop+Resize (CPU, ~0.1s)
    # ColorJitter runs on GPU after h2d, before forward
"""

import torch
from torchvision.transforms import v2


class GPUColorJitter:
    """Applies ColorJitter on GPU tensors using torchvision v2.

    Input: video as list of [T, C, H, W] tensors (list-collated) or stacked [B, T, C, H, W].
    Each sample in the batch gets independent random jitter params,
    but all frames within one sample share the same params (temporal consistency).
    """

    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        self._jitter = v2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    @torch.no_grad()
    def __call__(self, video) -> list | torch.Tensor:
        if isinstance(video, list):
            return [self._apply_single(v) if isinstance(v, torch.Tensor) else v for v in video]
        if isinstance(video, torch.Tensor):
            if video.dim() == 5:
                return torch.stack([self._apply_single(video[i]) for i in range(video.shape[0])])
            if video.dim() == 4:
                return self._apply_single(video)
        return video

    def _apply_single(self, x: torch.Tensor) -> torch.Tensor:
        """Apply jitter to a single video [T, C, H, W]."""
        return self._jitter(x)
