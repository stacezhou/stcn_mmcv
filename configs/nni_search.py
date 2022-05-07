from configs.work_well_fast import *
annotation = '使用 nni 进行快速超参数搜索'
'''@nni.variable(nni.choice(True,False),name=origin_pipeline)'''
origin_pipeline = False
if origin_pipeline:
    from configs.stcn_origin import data as ori_data
    for train in data['train']:
        train['pipeline'] = ori_data['train'][0]['pipeline']
    data['val']['pipeline'] = ori_data['val']['pipeline']
    data['val']['wo_mask_pipeline'] = ori_data['val']['wo_mask_pipeline']
    data['test']['pipeline'] = ori_data['test']['pipeline']
    data['test']['wo_mask_pipeline'] = ori_data['test']['wo_mask_pipeline']
    del train,ori_data


'''@nni.variable(nni.choice(5e-4,5e-3,5e-5,5e-2),name=learning_rate)'''
learning_rate = 5e-4
optimizer = dict(type='Adam', lr=learning_rate, weight_decay=1e-7)

log_config = dict(
    interval=51,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
        dict(type='NNIHook',
            metric_full_name = 'mIoU',
            final_iter = 3000,
            # interval=101,
            ),
    ])

'''@nni.variable(nni.choice(-1,1,2,3),name=frozen_stages) '''
frozen_stages = 1
model['key_encoder']['backbone']['frozen_stages'] = frozen_stages

'''@nni.variable(nni.choice(2,3,4),name=max_objs) '''
max_objs = 3
model['max_objs_per_frame'] = max_objs
evaluation = dict(
    start=101,
    save_best='mIoU',
    interval=101,
    by_epoch=False)
