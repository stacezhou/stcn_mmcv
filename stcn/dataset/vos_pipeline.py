from torchvision import transforms
from mmdet.datasets import PIPELINES
from mmcv.parallel import DataContainer as DC
from PIL import Image
import numpy as np

@PIPELINES.register_module()
class LoadMaskFromFile:
    def __call__(self, results):
        mask_file = results['ann_info']['masks']
        gt_mask = Image.open(mask_file).convert('P').__array__()
        return {
            **results,
            'gt_mask': gt_mask,
            'mask_fields':['gt_mask']
        }

@PIPELINES.register_module()
class MergeImgMask:
    def __call__(self, results):
        img = results['img']
        mask = results['gt_mask']
        img_w_mask = np.concatenate([img,np.expand_dims(mask,2)], axis=2)
        results['img'] = img_w_mask
        results['mask_fields'].remove('gt_mask')
        # results['img_fields'] = ['img', 'gt_mask']
        return results

@PIPELINES.register_module()
class SplitImgMask:
    def __call__(self, results):
        img_w_mask = results['img']
        img = img_w_mask[:,:,:3]
        mask = img_w_mask[:,:,3]
        results['img'] = img
        results['gt_mask'] = mask
        results['mask_fields'].append('gt_mask')
        return results


from collections import namedtuple
Mask = namedtuple('albu_mask','masks')
@PIPELINES.register_module()
class EnAlbu:
    def __call__(self, results):
        for key in results['mask_fields']:
            results[key] = Mask([results[key]])
        return results



@PIPELINES.register_module()
class OutAlbu:
    def __call__(self, results):
        for key in results['mask_fields']:
            results[key] = results[key][0]
        return results


@PIPELINES.register_module()
class SafeCollect:
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:

            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key not in results:
                continue
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
