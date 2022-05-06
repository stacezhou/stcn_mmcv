from configs.work_well_fast import *
annotation = '使用 nni 进行快速超参数搜索'

'''@nni.variable(nni.loguniform(1e-5, 0.1),name=learning_rate)'''
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

'''@nni.variable(nni.choice(2,3,1),name=max_objs) '''
max_objs = 3
model['max_objs_per_frame'] = max_objs
evaluation = dict(
    start=101,
    save_best='mIoU',
    interval=101,
    by_epoch=False)
