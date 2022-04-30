pipeline= [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile'),
    dict(type='Resize', img_scale=(384, 384), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomCrop', ),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_mask']),
]
data = dict(
    type = 'StaticDataset',
    pipeline = pipeline,
    num_frames = 3,
    image_root = '/data/static/BIG_small',
)