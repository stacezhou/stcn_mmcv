img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
meta_keys = ('flag', 'filename', 'ori_filename','labels',
     'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
      'flip', 'flip_direction', 'img_norm_cfg')

vos_train_pipeline= [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile'),
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


vos_test_pipeline= [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile'),
    dict(type='MergeImgMask'), 
    dict(type='Pad', size_divisor=32),
    dict(type='SplitImgMask'),
    dict(type='ImageToTensor',keys=['gt_mask']),
    dict(type='ToDataContainer',fields=(dict(key='gt_mask',stack=True),)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='SafeCollect',
        keys=['img', 'gt_mask'],
        meta_keys=meta_keys,
    ),
]
vos_test_pipeline_wo_mask = [
    dict(type='LoadImageFromFile'),
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='SafeCollect',
        keys=['img'],
        meta_keys=meta_keys,
    ),

]

youtube_train = dict(
        type='VOSDataset',
        pipeline = vos_train_pipeline,
        frame_limit = 20,
        image_root = '/data/YouTube/train_480p/JPEGImages',
        mask_root = '/data/YouTube/train_480p/Annotations',
)
youtube_valid = dict(
        type='VOSDataset',
        test_mode = True,
        pipeline = vos_test_pipeline,
        wo_mask_pipeline =vos_test_pipeline_wo_mask,
        image_root = '/data/YouTube/valid/JPEGImages',
        mask_root = '/data/YouTube/valid/Annotations',
        palette = '/data/YouTube/valid/Annotations/0a49f5265b/00000.png',
)

youtube_debug_valid = dict(
        type='VOSDataset',
        test_mode = True,
        pipeline = vos_test_pipeline,
        wo_mask_pipeline = vos_test_pipeline_wo_mask,
        image_root = '/data/YouTube/debug/JPEGImages',
        mask_root = '/data/YouTube/debug/valid_Annotations',
        palette = '/data/YouTube/valid/Annotations/0a49f5265b/00000.png',
)