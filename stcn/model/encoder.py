import torch.nn as nn
import torch
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES
from .stcn import VOSMODEL
from .cbam import CBAM
from mmdet.models.backbones.resnet import BasicBlock

@VOSMODEL.register_module()
class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = BasicBlock(indim, outdim)
        self.attention = CBAM(outdim)
        self.block2 = BasicBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x

@VOSMODEL.register_module()
class ValueEncoder(BaseModule):
    def __init__(self,
            backbone, 
            feature_fusion, 
            init_cfg=None):
        super().__init__(init_cfg)
        self.backbone = BACKBONES.build(backbone)
        self.feature_fusion = VOSMODEL.build(feature_fusion)

    def forward(self, mask, feats):
        img = feats['img']
        f16 = feats['f16']
        f = torch.cat([img,mask],1)
        x = self.backbone(f)
        x = self.feature_fusion(x, f16)
        return x

@VOSMODEL.register_module()
class KeyEncoder(BaseModule):
    def __init__(self,
            backbone, 
            key_proj,
            key_comp,
            init_cfg=None):
        super().__init__(init_cfg)
        self.backbone = BACKBONES.build(backbone)
        self.key_proj = VOSMODEL.build(key_proj)
        self.key_comp = VOSMODEL.build(key_comp)

    def forward(self, img, feats=None):
        b = img.shape[0]
        f4,f8,f16 = self.backbone(img)

        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*H*W
        k16 = k16.view(b, *k16.shape[-3:]).contiguous()

        # B*C*H*W
        f16_thin = f16_thin.view(b, *f16_thin.shape[-3:])
        f16 = f16.view(b, *f16.shape[-3:])
        f8 = f8.view(b, *f8.shape[-3:])
        f4 = f4.view(b, *f4.shape[-3:])
        feats = {
            'f16' : f16,
            'f8' : f8,
            'f4' : f4,
            'img' : img,
            'f16_thin' : f16_thin,
            'K' : k16
        }

        return k16, feats


@VOSMODEL.register_module()
class KeyProjection(BaseModule):
    def __init__(self, indim, keydim, ortho_init = False):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        if ortho_init:
            nn.init.orthogonal_(self.key_proj.weight.data)
            nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x):
        return self.key_proj(x)
