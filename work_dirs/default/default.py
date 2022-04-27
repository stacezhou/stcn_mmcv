checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
key_dim = 64
value_dim = 512
custom_imports = dict(
    imports=['stcn.loss.bce', 'stcn.model'], allow_failed_imports=False)
model = dict(
    type='STCN',
    init_cfg=None,
    key_encoder=dict(
        type='KeyEncoder',
        backbone=dict(
            type='ResNet',
            depth=50,
            out_indices=(0, 1, 2),
            frozen_stages=1,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        key_proj=dict(
            type='KeyProjection', indim=1024, keydim=64, ortho_init=True),
        key_comp=dict(type='KeyProjection', indim=1024, keydim=512)),
    value_encoder=dict(
        type='ValueEncoder',
        backbone=dict(
            type='ResNet',
            depth=18,
            in_channels=4,
            out_indices=(2, ),
            frozen_stages=1),
        feature_fusion=dict(type='FeatureFusionBlock', indim=1280,
                            outdim=512)),
    mask_decoder=dict(type='MaskDecoder', indim=512),
    memory=dict(type='AffinityMemoryBank'),
    loss_fn=dict(type='BootstrappedCE'))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_vos_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile'),
    dict(type='MergeImgMask'),
    dict(type='Resize', img_scale=(896, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='SplitImgMask'),
    dict(type='ImageToTensor', keys=['gt_mask']),
    dict(type='ToDataContainer', fields=({
        'key': 'gt_mask',
        'stack': True
    }, )),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='SafeCollect', keys=['img', 'gt_mask'])
]
data = dict(
    workers_per_gpu=2,
    samples_per_gpu=2,
    train=dict(
        type='VOSTrainDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadMaskFromFile'),
            dict(type='MergeImgMask'),
            dict(type='Resize', img_scale=(896, 480), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Pad', size_divisor=32),
            dict(type='SplitImgMask'),
            dict(type='ImageToTensor', keys=['gt_mask']),
            dict(
                type='ToDataContainer',
                fields=({
                    'key': 'gt_mask',
                    'stack': True
                }, )),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='SafeCollect', keys=['img', 'gt_mask'])
        ],
        image_root='/data/DAVIS/2017/trainval/JPEGImages/480p',
        mask_root='/data/DAVIS/2017/trainval/Annotations/480p'),
    val=dict(
        type='VOSTrainDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadMaskFromFile'),
            dict(type='MergeImgMask'),
            dict(type='Resize', img_scale=(896, 480), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Pad', size_divisor=32),
            dict(type='SplitImgMask'),
            dict(type='ImageToTensor', keys=['gt_mask']),
            dict(
                type='ToDataContainer',
                fields=({
                    'key': 'gt_mask',
                    'stack': True
                }, )),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='SafeCollect', keys=['img', 'gt_mask'])
        ],
        image_root='/data/DAVIS/2017/trainval/JPEGImages/480p',
        mask_root='/data/DAVIS/2017/trainval/Annotations/480p'),
    test=dict(
        type='VOSTrainDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadMaskFromFile'),
            dict(type='MergeImgMask'),
            dict(type='Resize', img_scale=(896, 480), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Pad', size_divisor=32),
            dict(type='SplitImgMask'),
            dict(type='ImageToTensor', keys=['gt_mask']),
            dict(
                type='ToDataContainer',
                fields=({
                    'key': 'gt_mask',
                    'stack': True
                }, )),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='SafeCollect', keys=['img', 'gt_mask'])
        ],
        image_root='/data/DAVIS/2017/trainval/JPEGImages/480p',
        mask_root='/data/DAVIS/2017/trainval/Annotations/480p'))
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[190])
runner = dict(type='EpochBasedRunner', max_epochs=210)
work_dir = './work_dirs/default'
auto_resume = False
gpu_ids = [0]
