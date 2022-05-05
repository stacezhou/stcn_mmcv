from configs.base.default import *
from configs.base.default import data,model
from configs.base.path import youtube_path,davis_path
youtube_train = data['train'].copy()
davis_train = data['train'].copy()
youtube_valid = data['test'].copy()

youtube_train.update(youtube_path['train'])
youtube_valid.update(youtube_path['val'])
davis_train.update(davis_path)

youtube_train.update(dict(
    max_skip = 5,
    min_skip = 1,
    max_objs_per_frame = 2,
    max_objs_per_gpu= 8,
    nums_frame = 3,
))

data.update(dict(
    samples_per_gpu = 4,
    train = [youtube_train] + [davis_train] * 5,
    val = youtube_valid,
    test = youtube_valid,
))

model.update(dict(
    max_objs_per_frame = 2,
    seg_background = False,
    multi_scale_train = False,
    train_scales = [1],
    multi_scale_test = False,
    test_scales = [1, 1.3, 2],

))

model['memory'].update(dict(
    top_k = 20,
    mem_every = 5,
    include_last = False,
    train_memory_strategy = False,
    thin_reading_scale = 8,
))

model['key_encoder']['backbone']['frozen_stages'] = 3
model['loss_fn'].update(dict(
    start_warm=10000, 
    end_warm=40000, 
    top_p=0.15
))

optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    by_epoch = False,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[10000])
