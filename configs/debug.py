from configs.dataset.youtube import youtube_train,youtube_valid,youtube_debug_valid
from configs.model.stcn import model, custom_imports
from configs._base.default_runtime import *

data = dict(
    workers_per_gpu = 0,
    samples_per_gpu = 2,
    nums_frame = 4,
    train = youtube_train,
    val = youtube_debug_valid,
    test = youtube_valid,
)

optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch = False,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[10000])
runner = dict(type='EpochBasedRunner', max_epochs=10)
fp16 = dict(loss_scale=512.)

evaluation = dict(
    start=100,
    save_best='mIoU',
    interval=200,
    by_epoch=False)

del youtube_debug_valid,youtube_train,youtube_valid