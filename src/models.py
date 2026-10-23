import torch
import torch.nn as nn
from cdc import replace_convnext_dwconv_with_cdc
from ordinal import CoralLayer
from scheduled_dropblock import ScheduledDropBlock2d
from gcblock import GCBlock

class RockMassConvNeXtV2(nn.Module):

    def __init__(self, num_classes, use_cdc=False, cdc_stages=(0,), theta_mode='learnable', theta_init=0.05, head_type='ce', pretrained=True, model_name='convnextv2_tiny.fcmae_ft_in22k_in1k', local_variant='tiny', use_dropblock=False, dropblock_max_prob=0.1, dropblock_block_size=3, dropblock_schedule='linear', use_gc=False, gc_ratio=16):
        super().__init__()
        if head_type not in {'ce', 'coral'}:
            raise ValueError(head_type)
        self.num_classes = int(num_classes)
        self.head_type = head_type
        self.use_cdc = bool(use_cdc)
        if pretrained:
            try:
                import timm
                self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            except ImportError as e:
                raise ImportError('pretrained=True requires timm in this environment. Use --no-pretrained only for preliminary CPU experiments.') from e
        else:
            from local_convnextv2 import make_local_convnextv2
            self.backbone = make_local_convnextv2(local_variant)
        if not hasattr(self.backbone, 'num_features'):
            raise AttributeError('backbone has no num_features')
        feat_dim = int(self.backbone.num_features)
        self.cdc_wrappers = []
        if use_cdc:
            if not hasattr(self.backbone, 'stages'):
                raise ValueError(f'CDC requires a ConvNeXt-style backbone with .stages; got model_name={model_name}. Use --backbone convnextv2_tiny.fcmae_ft_in22k_in1k for CDC runs.')
            self.cdc_wrappers = replace_convnext_dwconv_with_cdc(self.backbone, stage_indices=tuple(cdc_stages), theta_mode=theta_mode, theta_init=theta_init)
        self.dropblock = None
        self._dropblock_handle = None
        if use_dropblock:
            if not hasattr(self.backbone, 'stages') or len(self.backbone.stages) < 4:
                raise ValueError('DropBlock requires a 4-stage ConvNeXt-style backbone with .stages; got model_name={model_name}')
            self.dropblock = ScheduledDropBlock2d(max_drop_prob=dropblock_max_prob, block_size=dropblock_block_size, schedule=dropblock_schedule)
            stage4 = self.backbone.stages[3]
            self._dropblock_handle = stage4.register_forward_pre_hook(self._dropblock_pre_hook)
        self.gc = None
        self._gc_handle = None
        if use_gc:
            if not hasattr(self.backbone, 'stages') or len(self.backbone.stages) < 4:
                raise ValueError(f'GC block requires a 4-stage ConvNeXt-style backbone with .stages; got model_name={model_name}')
            feat_dim = int(self.backbone.num_features)
            self.gc = GCBlock(feat_dim // 2, ratio=gc_ratio)
            if hasattr(self.backbone, 'downsample_layers'):
                target = self.backbone.downsample_layers[3]
            else:
                target = self.backbone.stages[3]
            self._gc_handle = target.register_forward_pre_hook(self._gc_pre_hook)
        if head_type == 'ce':
            self.head = nn.Linear(feat_dim, num_classes)
        else:
            self.head = CoralLayer(feat_dim, num_classes)

    def _dropblock_pre_hook(self, module, args):
        return (self.dropblock(args[0]),)

    def _gc_pre_hook(self, module, args):
        return (self.gc(args[0]),)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)

def create_experiment_model(experiment, num_classes, pretrained=True, cdc_stages=(0,), theta_mode='learnable', theta_init=0.05, local_variant='tiny', model_name='convnextv2_tiny.fcmae_ft_in22k_in1k', use_dropblock=False, dropblock_max_prob=0.1, dropblock_block_size=3, dropblock_schedule='linear', use_gc=False, gc_ratio=16):
    experiment = experiment.upper()
    table = {'M0': dict(use_cdc=False, head_type='ce'), 'M1': dict(use_cdc=True, head_type='ce'), 'M2': dict(use_cdc=False, head_type='coral'), 'M3': dict(use_cdc=True, head_type='coral')}
    if experiment not in table:
        raise ValueError(f'experiment must be one of {list(table)}')
    return RockMassConvNeXtV2(num_classes=num_classes, pretrained=pretrained, cdc_stages=cdc_stages, theta_mode=theta_mode, theta_init=theta_init, local_variant=local_variant, model_name=model_name, use_dropblock=use_dropblock, dropblock_max_prob=dropblock_max_prob, dropblock_block_size=dropblock_block_size, dropblock_schedule=dropblock_schedule, use_gc=use_gc, gc_ratio=gc_ratio, **table[experiment])
