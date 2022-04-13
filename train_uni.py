from model_uni import STCNModel
from runner_uni import EpochBasedRunner
import torch.distributed as dist
from mmcv.parallel import MMDistributedDataParallel as DDP
from hooks_uni import EvalHook,DistEvalHook
import torch.optim as optim
import torch
import numpy as np
import random
from os import environ
from dataset_uni import get_dataset,para,get_dataloader

####### parameters
# random seed

if para['benchmark']:
    torch.backends.cudnn.benchmark = True

####### Init distributed environment
if 'LOCAL_RANK' not in environ:
    world_size = 1
    local_rank = 0
    torch.manual_seed(14159265)
    np.random.seed(14159265)
    random.seed(14159265)
else:
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    print('I am rank %d in this world of size %d!' % (local_rank, world_size))
    np.random.seed(np.random.randint(2**30-1) + local_rank*100)
    torch.manual_seed(np.random.randint(2**30-1) + local_rank*100)
    random.seed(np.random.randint(2**30-1) + local_rank*100)

####### dataloader
def renew_dataloader(**kw):
    dataset = get_dataset(**kw)
    return get_dataloader(dataset,world_size,local_rank)

####### model
stcn_model = STCNModel().cuda()
if world_size > 1:
    stcn_model = DDP(stcn_model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False)

####### runner
optimizer = optim.Adam(filter(
    lambda p: p.requires_grad, stcn_model.parameters()), lr=para['lr'], weight_decay=1e-7)

runner = EpochBasedRunner(
    model = stcn_model,
    lr_config={'policy':'step','step':[1000,2000],'gamma':0.1},
    optimizer=optimizer,
    exp_id=para['id'],
    log_interval=10,
    checkpoint_interval=2
)

##### eval config
def evaluate(results,label):
    iou = np.array(results).mean()
    return {f'iou_{label}': iou}

eval_dataloader = renew_dataloader(stage=3, max_skip=10, valset=True)
eval_static_dataloader = renew_dataloader(stage=0,valset=True)
if world_size > 1:
    runner.register_hook(DistEvalHook(eval_static_dataloader, interval=1,start=2, 
        evaluate_fn=lambda x:evaluate(x,'duts_te'),gpu_collect=True))
    runner.register_hook(DistEvalHook(eval_dataloader, interval=1,start=2, 
        evaluate_fn=lambda x:evaluate(x,'davis_val'),gpu_collect=True))
else:
    runner.register_hook(EvalHook(eval_static_dataloader, interval=1,start=2, 
        evaluate_fn=lambda x:evaluate(x,'duts_te')))
    runner.register_hook(EvalHook(eval_dataloader, interval=1,start=2,
        evaluate_fn=lambda x:evaluate(x,'davis_val')))

##### runner run
if para['load_model'] is not None:
    runner.resume(para['load_model'])
runner.run(
    [   
        renew_dataloader(stage=0), # 6000 iter per epch for stage 0
        renew_dataloader(stage=0,sec=True), # 6000 iter per epch for stage 0
    ],
    [
        ('train',1),
        ('train',1),
    ],
    max_epochs = 10
)
runner.run(
    [   
        renew_dataloader(stage=3,max_skip=10), # 6000 iter per epoch for stage 3
        renew_dataloader(stage=3,max_skip=15),
        renew_dataloader(stage=3,max_skip=20),
        renew_dataloader(stage=3,max_skip=25),
        renew_dataloader(stage=3,max_skip=5),
    ],
    [
        ('train',3),
        ('train',3),
        ('train',2),
        ('train',8),
        ('train',4)
    ],
    max_epochs=runner.epoch + 20
)

if world_size > 1:
    dist.destroy_process_group()