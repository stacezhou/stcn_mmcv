from mmcv.runner import BaseModule
from mmdet.models import BACKBONES

class Encoder(BaseModule):
    def __init__(self,backbone, feature_fusion, init_cfg=None):
        super().__init__(init_cfg)
        self.backbone = BACKBONES.build(backbone)
        self.feature_fusion = feature_fusion
        if self.feature_fusion:
            self.feature_fusion_block = None

    def forward(self, img, feats=None):
        x = self.backbone(img)
        if self.feature_fusion:
            x = self.feature_fusion_block(x, feats)
        return x
