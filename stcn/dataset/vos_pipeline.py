from torchvision import transforms
from mmdet.datasets import PIPELINES
from PIL import Image

@PIPELINES.register_module()
class LoadMaskFromFile:
    def __call__(self, results):
        mask_file = results['mask_info']['filename']
        gt_mask = Image.open(mask_file).convert('L')
        return {
            **results,
            'gt_mask': gt_mask,
            'mask_field':['gt_mask']
        }