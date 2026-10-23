import torch
import torch.nn as nn

class GCBlock(nn.Module):

    def __init__(self, channels, ratio=16):
        super().__init__()
        if channels // ratio < 1:
            raise ValueError(f'gc_ratio {ratio} too large for {channels} channels')
        mid = channels // ratio
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.channel_add_conv = nn.Sequential(nn.Conv2d(channels, mid, kernel_size=1), nn.LayerNorm([mid, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(mid, channels, kernel_size=1))

    def forward(self, x):
        b, c, h, w = x.shape
        mask = self.conv_mask(x).view(b, 1, h * w)
        mask = torch.softmax(mask, dim=2).view(b, 1, h, w)
        context = (x * mask).sum(dim=(2, 3), keepdim=True)
        return x + self.channel_add_conv(context)
