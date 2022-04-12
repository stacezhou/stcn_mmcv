from model_uni import STCNModel
from mmcv.runner import EpochBasedRunner
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

eval_dataloader = renew_dataloader(max_skip=10, valset=False)

####### model
stcn_model = STCNModel()
if world_size > 1:
    stcn_model = DDP(stcn_model.cuda(),device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False)

####### runner
optimizer = optim.Adam(filter(
    lambda p: p.requires_grad, stcn_model.parameters()), lr=para['lr'], weight_decay=1e-7)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, para['steps'], para['gamma'])
logger = get_logger('stcn')
runner = EpochBasedRunner(
    model = stcn_model.cuda(),
    optimizer=optimizer,
    work_dir='/tmp/debug',
    logger=logger,
    meta={},
    max_epochs=1000
)
# learning rate scheduler config
lr_config = dict(policy='step', step=[2, 3])
optimizer_config = dict(grad_clip=None)
# configuration of saving checkpoints periodically
checkpoint_config = dict(interval=100)
# save log periodically and multiple hooks can be used simultaneously
log_config = dict(interval=5, 
    hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook')
    ])
runner.register_training_hooks(
    lr_config=lr_config,
    optimizer_config=optimizer_config,
    checkpoint_config=checkpoint_config,
    log_config=log_config
)

def evaluate(results):
    iou = np.array(results)
    mean = iou.mean()
    max = iou.max()
    min = iou.min()
    return {'iou_mean': mean, 'iou_max': max,'iou_min':min}
if world_size > 1:
    runner.register_hook(DistEvalHook(eval_dataloader, interval=10, evaluate_fn=evaluate))
else:
    runner.register_hook(EvalHook(eval_dataloader, interval=10, evaluate_fn=evaluate))

runner.run(
    [   
        # renew_dataloader(stage=0),
        renew_dataloader(stage=3,max_skip=5),
        renew_dataloader(stage=3,max_skip=10),
    ],
    [
        # ('train',1),
        ('train',1),
        ('train',1)
    ]
)

if world_size > 1:
    dist.destroy_process_group()