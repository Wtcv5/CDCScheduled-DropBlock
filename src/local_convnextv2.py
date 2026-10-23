import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]
        raise ValueError(self.data_format)

class GRN(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-06)
        return self.gamma * (x * nx) + self.beta + x

class Block(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x

class Stage(nn.Module):

    def __init__(self, dim, depth):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ConvNeXtV2TinyFeatures(nn.Module):

    def __init__(self, in_chans=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)):
        super().__init__()
        self.num_features = dims[-1]
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), LayerNorm(dims[0], eps=1e-06, data_format='channels_first')))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(LayerNorm(dims[i], eps=1e-06, data_format='channels_first'), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))
        self.stages = nn.ModuleList([Stage(dims[i], depths[i]) for i in range(4)])
        self.norm = nn.LayerNorm(dims[-1], eps=1e-06)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean(dim=(-2, -1)))

def make_local_convnextv2(variant='tiny'):
    cfg = {'atto': ((2, 2, 6, 2), (40, 80, 160, 320)), 'tiny': ((3, 3, 9, 3), (96, 192, 384, 768))}
    if variant not in cfg:
        raise ValueError(variant)
    depths, dims = cfg[variant]
    return ConvNeXtV2TinyFeatures(depths=depths, dims=dims)
