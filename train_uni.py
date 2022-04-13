from model_uni import STCNModel
import datetime
from mmcv.runner import EpochBasedRunner
from mmcv.runner import DistSamplerSeedHook
from mmcv.utils import get_logger
import torch.distributed as dist
from mmcv.parallel import MMDistributedDataParallel as DDP
from hooks_uni import EvalHook,DistEvalHook
import torch.optim as optim
import torch
import numpy as np
import random
from pathlib import Path
from os import environ
from dataset_uni import get_dataset,para,increase_skip_fraction,get_dataloader

MAX_EPOCH = 3000
####### parameters
# random seed
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

if para['benchmark']:
    torch.backends.cudnn.benchmark = True

####### Init distributed environment
if 'LOCAL_RANK' not in environ:
    world_size = 1
    local_rank = 0
else:
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    print('I am rank %d in this world of size %d!' % (local_rank, world_size))

####### dataloader
def renew_dataloader(**kw):
    dataset = get_dataset(**kw)
    return get_dataloader(dataset,world_size,local_rank)

eval_dataloader = renew_dataloader(stage=3, max_skip=10, valset=True)

####### model
stcn_model = STCNModel()
if world_size > 1:
    stcn_model = DDP(stcn_model.cuda(),device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False)

####### runner
optimizer = optim.Adam(filter(
    lambda p: p.requires_grad, stcn_model.parameters()), lr=para['lr'], weight_decay=1e-7)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, para['steps'], para['gamma'])
logger = get_logger('stcn')
if para['id'] != 'NULL':
    work_dir = f'work_dir/{para["id"]}'
else:
    work_dir = f'work_dir/exp_{datetime.datetime.now().strftime("%b%d_%H.%M.%S")}'

runner = EpochBasedRunner(
    model = stcn_model.cuda(),
    optimizer=optimizer,
    work_dir=work_dir,
    logger=logger,
    meta={},
    max_epochs=MAX_EPOCH
)
# learning rate scheduler config
lr_config = dict(policy='step', step=[1000,2000])
if para['amp']:
    from mmcv.runner.hooks import Fp16OptimizerHook
    optimizer_config = Fp16OptimizerHook(grad_clip=None,loss_scale='dynamic')
else:
    optimizer_config = dict(grad_clip=None)
# configuration of saving checkpoints periodically
checkpoint_config = dict(interval=10)
# save log periodically and multiple hooks can be used simultaneously
log_config = dict(interval=10, 
    hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook')
    ])
runner.register_training_hooks(
    lr_config=lr_config,
    optimizer_config=optimizer_config,
    checkpoint_config=checkpoint_config,
    log_config=log_config
)
runner.register_hook(DistSamplerSeedHook())
def evaluate(results):
    iou = np.array(results)
    mean = iou.mean()
    max = iou.max()
    min = iou.min()
    return {'iou_mean': mean, 'iou_max': max,'iou_min':min}
if world_size > 1:
    runner.register_hook(DistEvalHook(eval_dataloader, interval=5,start=1, evaluate_fn=evaluate,gpu_collect=True))
else:
    runner.register_hook(EvalHook(eval_dataloader, interval=5,start=1, evaluate_fn=evaluate))

np.random.seed(np.random.randint(2**30-1) + local_rank*100)
if para['load_model']:
    runner.resume(para['load_model'])

runner.run(
    [   
        renew_dataloader(stage=0),
        renew_dataloader(stage=3,max_skip=10),
        renew_dataloader(stage=3,max_skip=15),
        renew_dataloader(stage=3,max_skip=20),
        renew_dataloader(stage=3,max_skip=25),
        renew_dataloader(stage=3,max_skip=5),
    ],
    [
        ('train',2),
        ('train',50),
        ('train',50),
        ('train',50),
        ('train',248),
        ('train',100)
    ]
)

if world_size > 1:
    dist.destroy_process_group()