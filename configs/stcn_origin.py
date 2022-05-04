from configs.dataset.youtube import youtube_train,youtube_valid,youtube_debug_valid
from configs.dataset.davis import davis_train
from configs.dataset.ovis import ovis_train
from configs.model.stcn import model, custom_imports
from configs._base.default_runtime import *

train_data_config = dict(
    max_per_frame = 3,
    max_objs_per_gpu= 10,
    shuffle_videos = True,
    random_skip = False,
    nums_frame = 4,
    max_skip = 5,
    min_skip = 1,
)
youtube_train.update(train_data_config)
davis_train.update(train_data_config)
ovis_train.update(train_data_config)

data = dict(
    workers_per_gpu = 0,
    samples_per_gpu = 4,
    train = youtube_train,
    val = youtube_debug_valid,
    test = youtube_valid,
)

model.update(dict(
    max_per_frame = 3,
    multi_scale_train = False,
))
model['key_encoder']['backbone']['frozen_stages'] = 3
model['loss_fn'].update(dict(
    start_warm=5000, 
    end_warm=10000, 
    top_p=0.15,
))

model['memory'].update(dict(
    top_k = 10,
    mem_every = 5,
    include_last = True,
    thin_reading_scale = 8,
))

optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    by_epoch = False,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[8000])
runner = dict(type='IterBasedRunner', max_iters=15000)
checkpoint_config = dict(interval=2500)
fp16 = dict(loss_scale=512.)

evaluation = dict(
    start=100,
    save_best='mIoU',
    interval=100,
    by_epoch=False)

del youtube_debug_valid,youtube_train,youtube_valid,davis_train,ovis_train,train_data_config
