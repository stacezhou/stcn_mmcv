from configs.lr5e_4slowSkip import *
annotation = '训练时 crop 的尺寸更大一些，可能收敛更快？更改了 Albu RandomResizedCrop的尺寸, 显存不够了，batch—size 调成2'
for train in data['train']:
    train['pipeline'][3]['transforms'][0] = dict(type='RandomResizedCrop',
                    # height=384,
                    # width=384,
                    height = 480,
                    width = 896,
                    scale=(0.36, 1.0), 
                    ratio=(0.7, 1.3),
                    p = 1,
                    )

data.update(dict(
    samples_per_gpu = 2,
))
del train