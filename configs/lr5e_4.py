from configs.stcn_origin import *
model['key_encoder']['backbone']['frozen_stages'] = 3
optimizer = dict(type='Adam', lr=5e-4, weight_decay=1e-7)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    by_epoch = False,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[125000])

annotation = '增加学习率，以加快训练速度，同时设置 warmup 和梯度剪裁，增加稳定性'