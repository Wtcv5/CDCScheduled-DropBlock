import torch
import torch.nn as nn
import torch.nn.functional as F

class CoralLayer(nn.Module):

    def __init__(self, in_features, num_classes, preinit_bias=True):
        super().__init__()
        if num_classes < 2:
            raise ValueError('num_classes must be >=2')
        self.num_classes = int(num_classes)
        self.shared_weight = nn.Linear(in_features, 1, bias=False)
        if preinit_bias:
            bias = torch.arange(num_classes - 1, 0, -1, dtype=torch.float32) / (num_classes - 1)
        else:
            bias = torch.zeros(num_classes - 1, dtype=torch.float32)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return self.shared_weight(x) + self.bias

def levels_from_labels(labels, num_classes):
    if labels.dtype != torch.long:
        labels = labels.long()
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    return (labels.unsqueeze(1) > thresholds).float()

def coral_loss(logits, labels, num_classes):
    levels = levels_from_labels(labels, num_classes)
    return F.binary_cross_entropy_with_logits(logits, levels, reduction='mean')

def coral_probabilities(logits):
    return torch.sigmoid(logits)

def coral_predict(logits, threshold=0.5):
    probs = coral_probabilities(logits)
    return (probs > threshold).sum(dim=1).long()

def class_probs_from_cumulative(logits):
    q = torch.sigmoid(logits)
    p0 = 1.0 - q[:, :1]
    if q.shape[1] == 1:
        p_last = q[:, -1:]
        p = torch.cat([p0, p_last], dim=1)
    else:
        middle = q[:, :-1] - q[:, 1:]
        p_last = q[:, -1:]
        p = torch.cat([p0, middle, p_last], dim=1)
    p = p.clamp_min(0.0)
    return p / p.sum(dim=1, keepdim=True).clamp_min(1e-08)
