from configs.dataset.youtube import youtube_train,youtube_valid,youtube_debug_valid
from configs.model.stcn import model, custom_imports
from configs._base.default_runtime import *

data = dict(
    workers_per_gpu = 0,
    samples_per_gpu = 2,
    nums_frame = 4,
    sampler = dict(
        shuffle_videos = True,
        random_skip = True,
        max_skip = 5,
        min_skip = 1,
        max_objs_per_gpu=-1,
    ),
    train = youtube_train,
    val = youtube_debug_valid,
    test = youtube_valid,
)

optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[10])
runner = dict(type='EpochBasedRunner', max_epochs=20)
fp16 = dict(loss_scale=512.)

evaluation = dict(
    start=10,
    save_best='mIoU',
    interval=1000,
    by_epoch=False)

del youtube_debug_valid,youtube_train,youtube_valid