from __future__ import annotations
import math
from typing import Literal
import torch
from torch import Tensor, nn
import torch.nn.functional as F
ScheduleMode = Literal['linear', 'cosine', 'constant']

class ScheduledDropBlock2d(nn.Module):

    def __init__(self, max_drop_prob: float=0.1, block_size: int=3, schedule: ScheduleMode='linear', eps: float=1e-06) -> None:
        super().__init__()
        if not 0.0 <= max_drop_prob < 1.0:
            raise ValueError(f'max_drop_prob must be in [0, 1), got {max_drop_prob}')
        if block_size < 1:
            raise ValueError(f'block_size must be >= 1, got {block_size}')
        if block_size % 2 == 0:
            raise ValueError(f'block_size should be odd so the masked block has a unique center; got {block_size}. Use 3 or 5.')
        if schedule not in {'linear', 'cosine', 'constant'}:
            raise ValueError(f'Unsupported schedule: {schedule}')
        self.max_drop_prob = float(max_drop_prob)
        self.block_size = int(block_size)
        self.schedule = schedule
        self.eps = float(eps)
        self.current_drop_prob = self.max_drop_prob if self.schedule == 'constant' else 0.0
        self.current_epoch = 0
        self.total_epochs = 0

    def extra_repr(self) -> str:
        return f"max_drop_prob={self.max_drop_prob}, block_size={self.block_size}, schedule='{self.schedule}', current_drop_prob={self.current_drop_prob:.4f}"

    def set_epoch(self, epoch: int, total_epochs: int) -> float:
        if total_epochs < 1:
            raise ValueError(f'total_epochs must be >= 1, got {total_epochs}')
        if epoch < 1:
            raise ValueError(f'epoch must be >= 1 (1-based), got {epoch}')
        self.current_epoch = int(epoch)
        self.total_epochs = int(total_epochs)
        if self.schedule == 'constant':
            p = self.max_drop_prob
        elif total_epochs == 1:
            p = self.max_drop_prob
        else:
            progress = (epoch - 1) / float(total_epochs - 1)
            progress = min(max(progress, 0.0), 1.0)
            if self.schedule == 'linear':
                factor = progress
            else:
                factor = 0.5 * (1.0 - math.cos(math.pi * progress))
            p = self.max_drop_prob * factor
        self.current_drop_prob = float(p)
        return self.current_drop_prob

    def _gamma(self, h: int, w: int, block_size: int, drop_prob: float) -> float:
        valid_h = h - block_size + 1
        valid_w = w - block_size + 1
        if valid_h <= 0 or valid_w <= 0:
            return 0.0
        gamma = drop_prob * (h * w) / float(block_size * block_size) / float(valid_h * valid_w)
        return float(min(max(gamma, 0.0), 1.0))

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.current_drop_prob <= 0.0:
            return x
        if x.ndim != 4:
            raise ValueError(f'ScheduledDropBlock2d expects NCHW 4-D input, got shape {tuple(x.shape)}')
        n, _, h, w = x.shape
        block_size = min(self.block_size, h, w)
        if block_size % 2 == 0:
            block_size -= 1
        if block_size < 1:
            return x
        gamma = self._gamma(h, w, block_size, self.current_drop_prob)
        if gamma <= 0.0:
            return x
        center_mask = torch.rand((n, 1, h, w), device=x.device, dtype=torch.float32) < gamma
        half = block_size // 2
        valid_centers = torch.zeros((1, 1, h, w), device=x.device, dtype=torch.bool)
        valid_centers[:, :, half:h - half, half:w - half] = True
        center_mask &= valid_centers
        block_mask = F.max_pool2d(center_mask.to(dtype=x.dtype), kernel_size=block_size, stride=1, padding=half)
        keep_mask = 1.0 - block_mask
        keep_count = keep_mask.sum()
        if keep_count <= 0:
            return x
        scale = keep_mask.numel() / (keep_count + self.eps)
        return x * keep_mask * scale
