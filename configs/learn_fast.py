from configs.stcn_origin import *
annotation = '复现之前训练比较快的配置'
model['key_encoder']['backbone']['frozen_stages'] = 3

for train in data['train']:
    train['pipeline'][3]['transforms'][0] = dict(type='RandomResizedCrop',
                    # height=384,
                    # width=384,
                    height = 480,
                    width = 896,
                    # scale=(0.36, 1.0), 
                    scale=(0.8, 1.0), 
                    ratio=(0.7, 1.3),
                    p = 1,
                    )
    train['pipeline'][-3] = dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
    train.update(dict(
        max_skip = 2
    ))
data.update(dict(
    samples_per_gpu = 2,
))
del train