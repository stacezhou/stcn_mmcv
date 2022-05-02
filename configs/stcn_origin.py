from configs.dataset.youtube import youtube_train,youtube_valid,youtube_debug_valid
from configs.dataset.davis import davis_train
from configs.dataset.ovis import ovis_train
from configs.model.stcn import model, custom_imports
from configs._base.default_runtime import *

train_data_config = dict(
    max_per_frame = 3,
    max_objs_per_gpu= 10,
    frame_limit = 36,
    shuffle_videos = True,
    random_skip = True,
    max_skip = 5,
    min_skip = 1,
)
youtube_train.update(train_data_config)
davis_train.update(train_data_config)
ovis_train.update(train_data_config)

data = dict(
    workers_per_gpu = 0,
    samples_per_gpu = 4,
    nums_frame = 4,
    train = [youtube_train, davis_train],
    val = youtube_debug_valid,
    test = youtube_valid,
)

model.update(dict(
    max_per_frame = 3
))
model['key_encoder']['backbone']['frozen_stages'] = 3
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    by_epoch = False,
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 3,
    step=[10000])
runner = dict(type='EpochBasedRunner', max_epochs=20)
fp16 = dict(loss_scale=512.)

evaluation = dict(
    start=200,
    save_best='mIoU',
    interval=200,
    by_epoch=False)

del youtube_debug_valid,youtube_train,youtube_valid
