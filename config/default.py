_base_ = [ '_base/default_runtime.py','model/stcn.py']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
meta_keys = ('flag', 'filename', 'ori_filename','labels',
     'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
      'flip', 'flip_direction', 'img_norm_cfg')
train_vos_pipeline= [
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
    dict(type='SafeCollect',
        keys=['img', 'gt_mask'],
        meta_keys=meta_keys,
    ),
]
test_vos_pipeline= [
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
wo_mask_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='SafeCollect',
        keys=['img'],
        meta_keys=meta_keys,
    ),

]
data = dict(
    workers_per_gpu = 0,
    samples_per_gpu = 2,
    max_objs_per_gpu = 8,
    nums_frame = 3,
    train = dict(
        type='VOSDataset',
        max_skip=10,
        min_skip=1,
        pipeline = train_vos_pipeline,
        image_root = '/data/YouTube/train_480p/JPEGImages',
        mask_root = '/data/YouTube/train_480p/Annotations',
    ),
    test = dict(
        type='VOSDataset',
        pipeline = test_vos_pipeline,
        wo_mask_pipeline = wo_mask_pipeline,
        test_mode = True,
        image_root = '/data/YouTube/valid/JPEGImages',
        mask_root = '/data/YouTube/valid/Annotations',
    ),
)

optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[190])
runner = dict(type='EpochBasedRunner', max_epochs=30)
find_unused_parameters = True
fp16 = dict(loss_scale=512.)