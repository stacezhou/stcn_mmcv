davis_train = dict(
    type='VOSDataset',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadMaskFromFile'),
        dict(type='EnAlbu'),
        dict(
            type='Albu',
            transforms=[
                dict(
                    type='RandomResizedCrop',
                    height=480,
                    width=896,
                    scale=(0.8, 1.0),
                    ratio=(0.7, 1.3),
                    p=1),
                dict(type='ShiftScaleRotate', p=0.9),
                dict(
                    type='RandomBrightnessContrast',
                    brightness_limit=[0.1, 0.3],
                    contrast_limit=[0.1, 0.3],
                    p=0.2),
                dict(type='ChannelShuffle', p=0.1),
                dict(
                    type='OneOf',
                    transforms=[
                        dict(
                            type='RGBShift',
                            p=1.0,
                            r_shift_limit=(-20, 20),
                            g_shift_limit=(-20, 20),
                            b_shift_limit=(-20, 20))
                    ],
                    p=0.1)
            ],
            keymap=dict(gt_mask='masks', img='image')),
        dict(type='OutAlbu'),
        dict(type='MergeImgMask'),
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
    frame_limit=36,
    shuffle_videos=True,
    random_skip=False,
    max_skip=5,
    min_skip=1,
    max_objs_per_gpu=10,
    max_per_frame=3,
    image_root='/data/DAVIS/2017/trainval/JPEGImages/480p',
    mask_root='/data/DAVIS/2017/trainval/Annotations/480p')
ovis_train = dict(
    type='VOSDataset',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadMaskFromFile'),
        dict(type='EnAlbu'),
        dict(
            type='Albu',
            transforms=[
                dict(
                    type='RandomResizedCrop',
                    height=480,
                    width=896,
                    scale=(0.8, 1.0),
                    ratio=(0.7, 1.3),
                    p=1),
                dict(type='ShiftScaleRotate', p=0.9),
                dict(
                    type='RandomBrightnessContrast',
                    brightness_limit=[0.1, 0.3],
                    contrast_limit=[0.1, 0.3],
                    p=0.2),
                dict(type='ChannelShuffle', p=0.1),
                dict(
                    type='OneOf',
                    transforms=[
                        dict(
                            type='RGBShift',
                            p=1.0,
                            r_shift_limit=(-20, 20),
                            g_shift_limit=(-20, 20),
                            b_shift_limit=(-20, 20))
                    ],
                    p=0.1)
            ],
            keymap=dict(gt_mask='masks', img='image')),
        dict(type='OutAlbu'),
        dict(type='MergeImgMask'),
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
    frame_limit=36,
    shuffle_videos=True,
    random_skip=False,
    max_skip=5,
    min_skip=1,
    max_objs_per_gpu=10,
    max_per_frame=3,
    image_root='/data/OVIS_img/train',
    mask_root='/data/OVIS_anno/OVIS_anno')
model = dict(
    type='STCN',
    init_cfg=dict(type='Kaiming', layer='Conv2d'),
    seg_background=False,
    max_per_frame=3,
    key_encoder=dict(
        type='KeyEncoder',
        backbone=dict(
            type='ResNet',
            depth=50,
            out_indices=(0, 1, 2),
            frozen_stages=2,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        key_proj=dict(
            type='KeyProjection', indim=1024, keydim=64, ortho_init=True),
        key_comp=dict(type='KeyProjection', indim=1024, keydim=512)),
    value_encoder=dict(
        type='ValueEncoder',
        backbone=dict(
            type='ResNet', depth=18, in_channels=4, out_indices=(2, )),
        feature_fusion=dict(type='FeatureFusionBlock', indim=1280,
                            outdim=512)),
    mask_decoder=dict(type='MaskDecoder', indim=512),
    memory=dict(
        type='AffinityMemoryBank',
        top_k=-1,
        mem_every=5,
        include_last=False,
        thin_reading_scale=8),
    loss_fn=dict(
        type='BootstrappedCE', start_warm=10000, end_warm=40000, top_p=0.15))
custom_imports = dict(
    imports=['stcn.loss.bce', 'stcn.model'], allow_failed_imports=False)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=25,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
find_unused_parameters = True
train_data_config = dict(
    max_per_frame=3,
    max_objs_per_gpu=10,
    frame_limit=36,
    shuffle_videos=True,
    random_skip=False,
    max_skip=5,
    min_skip=1)
data = dict(
    workers_per_gpu=0,
    samples_per_gpu=4,
    nums_frame=4,
    train=dict(
        type='VOSDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadMaskFromFile'),
            dict(type='EnAlbu'),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='RandomResizedCrop',
                        height=480,
                        width=896,
                        scale=(0.8, 1.0),
                        ratio=(0.7, 1.3),
                        p=1),
                    dict(type='ShiftScaleRotate', p=0.9),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[0.1, 0.3],
                        contrast_limit=[0.1, 0.3],
                        p=0.2),
                    dict(type='ChannelShuffle', p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RGBShift',
                                p=1.0,
                                r_shift_limit=(-20, 20),
                                g_shift_limit=(-20, 20),
                                b_shift_limit=(-20, 20))
                        ],
                        p=0.1)
                ],
                keymap=dict(gt_mask='masks', img='image')),
            dict(type='OutAlbu'),
            dict(type='MergeImgMask'),
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
        frame_limit=36,
        shuffle_videos=True,
        random_skip=False,
        max_skip=5,
        min_skip=1,
        max_objs_per_gpu=10,
        max_per_frame=3,
        image_root='/data/YouTube/train_480p/JPEGImages',
        mask_root='/data/YouTube/train_480p/Annotations'),
    val=dict(
        type='VOSDataset',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadMaskFromFile'),
            dict(type='MergeImgMask'),
            dict(type='Pad', size_divisor=16),
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
            dict(
                type='SafeCollect',
                keys=['img', 'gt_mask'],
                meta_keys=('flag', 'filename', 'ori_filename', 'labels',
                           'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg'))
        ],
        wo_mask_pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Pad', size_divisor=16),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='SafeCollect',
                keys=['img'],
                meta_keys=('flag', 'filename', 'ori_filename', 'labels',
                           'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg'))
        ],
        image_root='/data/YouTube/debug/JPEGImages',
        mask_root='/data/YouTube/debug/valid_Annotations',
        palette='/data/YouTube/valid/Annotations/0a49f5265b/00000.png'),
    test=dict(
        type='VOSDataset',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadMaskFromFile'),
            dict(type='MergeImgMask'),
            dict(type='Pad', size_divisor=16),
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
            dict(
                type='SafeCollect',
                keys=['img', 'gt_mask'],
                meta_keys=('flag', 'filename', 'ori_filename', 'labels',
                           'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg'))
        ],
        wo_mask_pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Pad', size_divisor=16),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='SafeCollect',
                keys=['img'],
                meta_keys=('flag', 'filename', 'ori_filename', 'labels',
                           'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg'))
        ],
        image_root='/data/YouTube/valid/JPEGImages',
        mask_root='/data/YouTube/valid/Annotations',
        palette='/data/YouTube/valid/Annotations/0a49f5265b/00000.png'))
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.3333333333333333,
    step=[10000])
runner = dict(type='EpochBasedRunner', max_epochs=20)
fp16 = dict(loss_scale=512.0)
evaluation = dict(start=200, save_best='mIoU', interval=200, by_epoch=False)
work_dir = 'work_dirs/yv_full_train_noskip'
validate = True
auto_resume = False
gpu_ids = range(0, 5)
