from configs.base.default import *
from configs.base.default import data,model
from configs.base.path import static_path,youtube_path


static_train = data['train'].copy()
static_train.update(static_path)
youtube_mini_valid = data['test'].copy()
youtube_mini_valid.update(youtube_path['mini_val'])
youtube_valid = data['test'].copy()
youtube_valid.update(youtube_path['val'])


#=======================================================
data.update(dict(
    samples_per_gpu = 8,
    train = static_train,
    val = youtube_mini_valid,
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

model['key_encoder']['backbone']['frozen_stages'] = -1
model['loss_fn'].update(dict(
    start_warm=20000, 
    end_warm=70000, 
    top_p=0.15
))

optimizer = dict(type='Adam', lr=1e-5, weight_decay=1e-7)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
runner = dict(type='IterBasedRunner', max_iters=300000)

lr_config = dict(
    policy='step',
    by_epoch = False,
    # warmup='linear',
    # warmup_iters=1000,
    # warmup_ratio=1.0 / 3,
    step=[150000])

annotation = """ 尽量和最初的 stcn 保持一致的配置版本 """
#==================================================
del youtube_path,youtube_valid,static_path,static_train,youtube_mini_valid