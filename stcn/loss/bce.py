import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES

@LOSSES.register_module()
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p
        self.this_p = 0

    def forward(self, input, target, it = 0):
        if it < self.start_warm:
            return F.cross_entropy(input, target)

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        self.this_p = this_p
        return loss.mean()
    