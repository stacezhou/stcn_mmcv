from configs.work_well_fast import *
runner = dict(type='IterBasedRunner', max_iters=300000)
# model['key_encoder']['backbone']['frozen_stages'] = 0
model.update(dict(
    max_objs_per_frame = 2
))
model['memory'].update(dict(
    top_k = 20,
    include_last= False,
))
for train in data['train']:
    train.update(dict(
        max_objs_per_frame = 2
    ))
    train['pipeline'][3]['transforms'][0].update(dict(
        height=384,
        width=384,
    ))
del train