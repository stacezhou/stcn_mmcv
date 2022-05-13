from configs.base.default import *
from configs.base.path import youtube_path,davis_path

youtube_train = data['train'].copy()
youtube_valid = data['train'].copy()

youtube_train.update(youtube_path['train'])
youtube_valid.update(youtube_path['val'])

youtube_mini_valid = data['test'].copy()
youtube_mini_valid.update(youtube_path['mini_val'])

for train in [youtube_train,youtube_valid]:
    train.update(dict(
        max_skip = 1,
        min_skip = 1,
        max_objs_per_frame = 3,
        max_objs_per_gpu= 8,
        nums_frame = 5,
    ))
del train


total_iters = 150000
increase_skip = [int(total_iters*f) for f in [0.1,0.2,0.3,0.4,0.8]]
max_skips = [10,15,20,25,5]

#=======================================================
inject = {
    it : f'for ds in data.datasets: ds.max_skip = {skip}'
    for it,skip in zip(increase_skip,max_skips)
}

data.update(dict(
    samples_per_gpu = 4,
    train = [youtube_valid]*20,
    val = youtube_mini_valid,
    test = youtube_valid,
))

model.update(dict(
    max_objs_per_frame = 3,
    seg_background = False,
    multi_scale_train = False,
    train_scales = [1],
    multi_scale_test = False,
    test_scales = [1, 1.3, 2],
))

model['memory'].update(dict(
    top_k = 20,
    mem_every = 1,
    include_last = True,
    train_memory_strategy = False,
    thin_reading_scale = 8,
))

model['key_encoder']['backbone']['frozen_stages'] = 3
model['loss_fn'].update(dict(
    start_warm=20000, 
    end_warm=70000, 
    top_p=0.15
))

optimizer = dict(type='Adam', lr=1e-3, weight_decay=1e-7)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
runner = dict(type='IterBasedRunner', max_iters=total_iters)

lr_config = dict(
    policy='step',
    by_epoch = False,
    # warmup='linear',
    # warmup_iters=1000,
    # warmup_ratio=1.0 / 3,
    step=[125000])

annotation = """ 尽量和最初的 stcn 保持一致的配置版本 """
#==================================================
del (youtube_path,youtube_train,youtube_valid,youtube_mini_valid,
    davis_path,total_iters,increase_skip,max_skips
    )