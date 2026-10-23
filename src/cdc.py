import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _logit(p: float) -> float:
    p = min(max(float(p), 1e-06), 1.0 - 1e-06)
    return math.log(p / (1.0 - p))

class CentralDifferenceDWConv2d(nn.Module):

    def __init__(self, conv_dw: nn.Conv2d, theta_mode: str='learnable', theta_init: float=0.05, shared_alpha=None):
        super().__init__()
        if not isinstance(conv_dw, nn.Conv2d):
            raise TypeError('conv_dw must be nn.Conv2d')
        if conv_dw.groups != conv_dw.in_channels:
            raise ValueError('CDC wrapper expects a depthwise convolution')
        if conv_dw.out_channels != conv_dw.in_channels:
            raise ValueError('Expected depthwise conv with in_ch == out_ch')
        if theta_mode not in {'fixed', 'learnable', 'shared'}:
            raise ValueError(theta_mode)
        self.conv_dw = conv_dw
        self.theta_mode = theta_mode
        if theta_mode == 'fixed':
            self.register_buffer('theta_fixed', torch.tensor(float(theta_init), dtype=torch.float32))
        elif theta_mode == 'learnable':
            self.alpha = nn.Parameter(torch.tensor(_logit(theta_init), dtype=torch.float32))
        else:
            if shared_alpha is None:
                raise ValueError('shared mode requires shared_alpha')
            self.__dict__['_shared_alpha'] = shared_alpha

    @property
    def theta(self):
        if self.theta_mode == 'fixed':
            return self.theta_fixed
        if self.theta_mode == 'learnable':
            return torch.sigmoid(self.alpha)
        return torch.sigmoid(self.__dict__['_shared_alpha'])

    def forward(self, x):
        vanilla = self.conv_dw(x)
        theta = self.theta
        kernel_sum = self.conv_dw.weight.sum(dim=(2, 3), keepdim=True)
        center = F.conv2d(x, kernel_sum, bias=None, stride=self.conv_dw.stride, padding=0, dilation=1, groups=self.conv_dw.groups)
        if center.shape[-2:] != vanilla.shape[-2:]:
            raise RuntimeError(f'CDC center/vanilla spatial mismatch: {center.shape} vs {vanilla.shape}')
        return vanilla - theta * center

def replace_convnext_dwconv_with_cdc(backbone, stage_indices=(0,), theta_mode='learnable', theta_init=0.05):
    shared_alpha = None
    if theta_mode == 'shared':
        shared_alpha = nn.Parameter(torch.tensor(_logit(theta_init), dtype=torch.float32))
        backbone.register_parameter('cdc_shared_alpha', shared_alpha)
    wrappers = []
    for sidx in stage_indices:
        stage = backbone.stages[sidx]
        if not hasattr(stage, 'blocks'):
            raise AttributeError(f'backbone.stages[{sidx}] has no .blocks')
        for block in stage.blocks:
            old = block.conv_dw
            wrapper = CentralDifferenceDWConv2d(old, theta_mode=theta_mode, theta_init=theta_init, shared_alpha=shared_alpha)
            block.conv_dw = wrapper
            wrappers.append(wrapper)
    return wrappers

def get_theta_values(backbone):
    vals = []
    for m in backbone.modules():
        if isinstance(m, CentralDifferenceDWConv2d):
            vals.append(float(m.theta.detach().cpu()))
    return vals
