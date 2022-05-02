from configs.dataset.youtube import youtube_train,youtube_valid,youtube_debug_valid
from configs.model.stcn import model, custom_imports
from configs._base.default_runtime import *

youtube_train.update(dict(
    max_per_frame = 2,
    max_objs_per_gpu= 4,
    frame_limit = 10,
    shuffle_videos = True,
    random_skip = False,
    max_skip = 5,
    min_skip = 1,
))

data = dict(
    workers_per_gpu = 0,
    samples_per_gpu = 2,
    nums_frame = 2,
    train = [
            youtube_train,
            ],
    val = youtube_debug_valid,
    test = youtube_debug_valid,
)

model['memory'].update(dict(
    top_k = 10,
    mem_every = 5,
    include_last = True,
    thin_reading_scale = 8,
))
model.update(dict(
    max_per_frame = 2
))
data['val'].update(dict(
    frame_limit = 10
))

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
    start=10,
    save_best='mIoU',
    interval=100,
    by_epoch=False)

del youtube_debug_valid,youtube_train,youtube_valid