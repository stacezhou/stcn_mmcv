import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES

class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p
        self.this_p = 0

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        self.this_p = this_p
        return loss.mean()
    
@LOSSES.register_module()
class StcnBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BootstrappedCE()

    def forward(self, pred_logits,cls_gt,a_map, it=0):
        loss = 0
        for i,per_frame in enumerate(a_map):
            c_gt = cls_gt[i].unsqueeze(0)
            logits = pred_logits[per_frame].unsqueeze(0)
            loss += self.bce(logits,c_gt,it)[0]

        return loss
    
    @property
    def p(self):
        return self.bce.this_p