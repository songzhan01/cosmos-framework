# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GPU-accelerated ColorJitter using torchvision.transforms.v2.

Replaces the CPU-bound torchvision.transforms.ColorJitter in DataLoader workers.
v2.ColorJitter natively supports CUDA tensors, producing results identical to
the CPU version (same RGB↔HSV conversion logic) but running on GPU (~3ms vs ~1.7s).

The random jitter parameters (brightness, contrast, saturation, hue) are sampled
on CPU in the DataLoader worker (preserving the same RNG sequence as the original
CPU-only path), stored in the sample dict, and applied deterministically on GPU.

Usage:
    export COSMOS_GPU_COLOR_JITTER=1  # enables GPU path
    # DataLoader only does RandomCrop+Resize + parameter sampling (CPU)
    # ColorJitter apply runs on GPU after h2d, before forward
"""

import torch
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue


class GPUColorJitterParams:
    """Samples ColorJitter parameters on CPU (in DataLoader worker).

    Call in __getitem__ to get a dict of params that travels with the sample.
    """

    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        self.brightness = (max(0, 1 - brightness), 1 + brightness) if brightness > 0 else None
        self.contrast = (max(0, 1 - contrast), 1 + contrast) if contrast > 0 else None
        self.saturation = (max(0, 1 - saturation), 1 + saturation) if saturation > 0 else None
        self.hue = (-hue, hue) if hue > 0 else None

    def sample(self) -> dict:
        """Sample random params using current CPU RNG (same as torchvision)."""
        # torchvision ColorJitter samples the random application order first,
        # then samples the per-op factors. Keep this order so the CPU RNG
        # sequence matches the original CPU-only path exactly.
        params = {"order": torch.randperm(4).tolist()}
        if self.brightness is not None:
            params["brightness"] = float(torch.empty(1).uniform_(*self.brightness).item())
        if self.contrast is not None:
            params["contrast"] = float(torch.empty(1).uniform_(*self.contrast).item())
        if self.saturation is not None:
            params["saturation"] = float(torch.empty(1).uniform_(*self.saturation).item())
        if self.hue is not None:
            params["hue"] = float(torch.empty(1).uniform_(*self.hue).item())
        return params


class GPUColorJitter:
    """Applies pre-sampled ColorJitter parameters on GPU tensors.

    Input: video as list of [T, C, H, W] tensors (list-collated) or stacked [B, T, C, H, W].
    Each sample must have corresponding params from GPUColorJitterParams.sample().
    """

    @torch.no_grad()
    def __call__(self, video, jitter_params: list[dict]) -> list | torch.Tensor:
        if isinstance(video, list):
            return [
                self._apply_single(v, p) if isinstance(v, torch.Tensor) else v
                for v, p in zip(video, jitter_params)
            ]
        if isinstance(video, torch.Tensor) and video.dim() == 5:
            return torch.stack([
                self._apply_single(video[i], jitter_params[i])
                for i in range(video.shape[0])
            ])
        return video

    def _apply_single(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply jitter to a single video [T, C, H, W] with pre-sampled params."""
        for idx in params.get("order", [0, 1, 2, 3]):
            if idx == 0 and "brightness" in params:
                x = adjust_brightness(x, params["brightness"])
            elif idx == 1 and "contrast" in params:
                x = adjust_contrast(x, params["contrast"])
            elif idx == 2 and "saturation" in params:
                x = adjust_saturation(x, params["saturation"])
            elif idx == 3 and "hue" in params:
                x = adjust_hue(x, params["hue"])
        return x
