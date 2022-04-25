from mmcv.runner import BaseModule
from mmdet.models import BACKBONES


class UpsampleBlock(BaseModule):
    pass

class Decoder(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
    
    def forward(self,x):
        return x
