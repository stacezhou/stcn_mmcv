img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)
meta_keys = ('flag', 'filename', 'ori_filename','labels',
     'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
      'flip', 'flip_direction', 'img_norm_cfg')

vos_train_pipeline= [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile'),
    dict(type='EnAlbu'),
    dict(type='Albu', 
        transforms = [
            dict(type='Resize',
                height=896, 
                width=480, 
                p=1, 
                interpolation=0),
            dict(type='RandomResizedCrop',
                height=384,
                width=384,
                scale=(0.36, 1.0), 
                ratio=(0.7, 1.3),
                p = 1,
                ),
            dict(type='ShiftScaleRotate', p=0.9),
            dict(type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.2),
            dict(type='OneOf',
                transforms=[
                    # dict(type='Blur', blur_limit=3, p=1.0),
                    # dict(type='MedianBlur', blur_limit=3, p=1.0),
                    # dict(type='MotionBlur',blur_limit=(3, 7),p=1.0),
                    dict(type='RGBShift',p=1.0, r_shift_limit=(-20, 20),
                     g_shift_limit=(-20, 20), b_shift_limit=(-20, 20))
                ],
                p=0.1),
        ],
        keymap = {'gt_mask':'masks','img':'image'},
    ),
    dict(type='OutAlbu'),
    dict(type='MergeImgMask'),
    # dict(type='Resize', img_scale=(896, 480), keep_ratio=True),
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
    dict(type='Pad', size_divisor=16),
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
    dict(type='Pad', size_divisor=16),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='SafeCollect',
        keys=['img'],
        meta_keys=meta_keys,
    ),

]

train_data = dict(
    type='VOSDataset',
    pipeline = vos_train_pipeline,
    palette = None,
    test_mode = False,
    nums_frame = 4,
    max_objs_per_gpu= 8,
    max_objs_per_frame = 3,
    max_skip = 10,
    min_skip = 1,
    # image_root = 
    # mask_root = 
)
test_data = dict(
    type='VOSDataset',
    test_mode = True,
    pipeline = vos_test_pipeline,
    wo_mask_pipeline = vos_test_pipeline_wo_mask,
    # image_root = 
    # mask_root = 
    # palette = 
)

#============================================================
from configs.base.model_stcn import model,custom_imports
data = dict(
    workers_per_gpu = 0,
    samples_per_gpu = 4,
    train = train_data,
    val = test_data,
    test = test_data,
)
log_config = dict(
    interval=51,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='NNIHook'),
        dict(type='TensorboardLoggerHook'),
    ])

optimizer = dict(type='Adam', lr=0.0005)

# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    by_epoch = False,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[10000])

runner = dict(type='IterBasedRunner', max_iters=15000)
checkpoint_config = dict(interval=5000)

fp16 = dict(loss_scale=512.)

evaluation = dict(
    start=100,
    save_best='mIoU',
    interval=501,
    by_epoch=False)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
auto_resume = True
workflow = [('train', 1)]

opencv_num_threads = 0
mp_start_method = 'fork'
find_unused_parameters = True
#========================================================
del img_norm_cfg,meta_keys,train_data,test_data,vos_test_pipeline,vos_test_pipeline_wo_mask,vos_train_pipeline
