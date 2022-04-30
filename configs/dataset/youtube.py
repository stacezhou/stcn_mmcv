img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pipeline= [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile'),
    # dict(type='RandomCrop', ),
    dict(type='MergeImgMask'),
    dict(type='Resize', img_scale=(896, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='SplitImgMask'),
    dict(type='ImageToTensor',keys=['gt_mask']),
    dict(type='ToDataContainer',fields=(dict(key='gt_mask',stack=True),)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='SafeCollect', keys=['img', 'gt_mask'], ),
]
data = dict(
    type='VOSDataset',
    pipeline = pipeline,
    image_root = '/data/YouTube/train_480p/JPEGImages',
    mask_root = '/data/YouTube/train_480p/Annotations',
)