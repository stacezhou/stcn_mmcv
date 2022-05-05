from configs.lr5e_4 import *
for train in data['train']:
    train.update(dict(
        max_skip = 2
    ))
del train
annotation = "开始的时候，skip 小一些，便于收敛"