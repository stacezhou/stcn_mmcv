from mmcv.runner import BaseModule
import torch
from .stcn import VOSMODEL
import torch.nn as nn
import torch.nn.functional as F
from .component import BasicBlock

class UpsampleBlock(BaseModule):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = BasicBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x

@VOSMODEL.register_module()
class MaskDecoder(BaseModule):
    def __init__(self, indim):
        super().__init__()
        self.compress = BasicBlock(indim, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, V, feats, fii):
        f16_thin = feats['f16_thin'][fii]
        f8 = feats['f8'][fii]
        f4 = feats['f4'][fii]
        x = torch.cat([V,f16_thin], dim=1)
        x = self.compress(V)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        logits = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        prob = torch.sigmoid(logits)

        return logits, prob