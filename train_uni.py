from model_uni import STCNModel
from runner_uni import EpochBasedRunner,IterBasedRunner
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

def init_seeds(seed=0):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)

if 'LOCAL_RANK' not in environ:
    world_size = 1
    local_rank = 0
    init_seeds(14159265)
else:
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    init_seeds(14159265 + local_rank*100)

if local_rank == 0:
    print(para)

####### dataloader
def renew_dataloader(batch_size=4,**kw):
    dataset = get_dataset(**kw)
    return get_dataloader(dataset,world_size,local_rank,batch_size=batch_size)

####### model
stcn_model = STCNModel().cuda()
if world_size > 1:
    stcn_model = DDP(stcn_model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False)

####### runner
optimizer = optim.Adam(filter(lambda p: p.requires_grad, stcn_model.parameters()), lr=para['lr'], weight_decay=1e-7)
runner = IterBasedRunner(
    model = stcn_model,
    optimizer=optimizer,
    lr_config={'policy':'step','step':para['steps'],'gamma':para['gamma']},
    exp_id=para['id'],
    log_interval=50,
    checkpoint_interval=5000,
    load_network=para['load_network'],
    resume_model=para['load_model'],
    amp=para['amp']

)

batch_size = para['batch_size']
if para['stage'] == 0:
    runner.run(
        [   # data_loaders
            renew_dataloader(stage=0,batch_size=batch_size,val=True),
            renew_dataloader(stage=0,batch_size=batch_size),
        ],
        [ 
            ('val',50),  # model.val_step(**data_batch) for data_batch in data_loader
            ('train',500),  # model.train_step(**data_batch) for data_batch in data_loader
        ],
        iters=para['iterations']
    )

elif para['stage'] == 3:
    all_iters = para['iterations']
    skip = [10,15,20,25,5]
    skip_fraction = [0.1, 0.1, 0.1, 0.4, 0.2]
    for skip,skip_fraction in zip(skip,skip_fraction):
        runner.run(
            [   
                renew_dataloader(stage=3,max_skip=skip,batch_size=batch_size,val=True),
                renew_dataloader(stage=3,max_skip=skip,batch_size=batch_size),
            ],
            [ 
                ('val',50) ,
                ('train',500), 
            ],
            iters=all_iters * skip_fraction
        )

if world_size > 1:
    dist.destroy_process_group()