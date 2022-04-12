from model_uni import STCNModel
from mmcv.runner import EpochBasedRunner
from mmcv.utils import get_logger
import torch.distributed as dist
from mmcv.parallel import MMDistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataset.static_dataset import StaticTransformDataset
from mmcv.runner.hooks import EvalHook,DistEvalHook
from dataset.vos_dataset import VOSDataset
from util.hyper_para import HyperParameters
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch
import numpy as np
import random
from pathlib import Path
from util.load_subset import load_sub_davis, load_sub_yv
from os import environ

####### parameters
para = HyperParameters()
para.parse()

# random seed
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

if para['benchmark']:
    torch.backends.cudnn.benchmark = True

####### Init distributed environment
if 'LOCAL_RANK' not in environ:
    world_size = 1
else:
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    print('I am rank %d in this world of size %d!' % (local_rank, world_size))


####### model
stcn_model = STCNModel()
if world_size > 1:
    stcn_model = DDP(stcn_model.cuda(),device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False)

####### dataset
davis_im_path = Path(para['davis_root']) / '2017' / 'trainval' / 'JPEGImages' / '480p'
davis_mask_path = Path(para['davis_root']) / '2017' / 'trainval' / 'Annotations' / '480p'
yv_im_path = Path(para['yv_root']) / 'train_480p' / 'JPEGImages' 
yv_mask_path = Path(para['yv_root']) / 'train' / 'Annotations' 
max_skip = 5

yv_dataset = VOSDataset(yv_im_path,yv_mask_path,max_skip//5, is_bl=False, subset=load_sub_yv())
train_subset = load_sub_davis()
davis_dataset = VOSDataset(davis_im_path,davis_mask_path,max_skip, is_bl=False, subset=train_subset)
train_dataset = ConcatDataset([davis_dataset]*5 + [yv_dataset])
if world_size > 1:
    train_sampler = DistributedSampler(train_dataset, rank=local_rank,shuffle=True)
    train_loader = DataLoader(train_dataset, 4, sampler= train_sampler,num_workers=para['num_workers'],drop_last=True, pin_memory=True)
else:
    train_loader = DataLoader(train_dataset, 4, num_workers=para['num_workers'],drop_last=True, pin_memory=True)

val_dataset = VOSDataset(davis_im_path,davis_mask_path,max_skip,is_bl=False,subset=load_sub_davis(),val=True)
if world_size > 1:
    val_sampler = DistributedSampler(val_dataset, rank=local_rank,shuffle=True)
    val_dataloader = DataLoader(val_dataset, 4, sampler= val_sampler,num_workers=para['num_workers'],drop_last=True, pin_memory=True)
else:
    val_dataloader = DataLoader(val_dataset, 4, num_workers=para['num_workers'],drop_last=True, pin_memory=True)
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
# configuration of optimizer
optimizer_config = dict(grad_clip=None)
# configuration of saving checkpoints periodically
checkpoint_config = dict(interval=100)
# save log periodically and multiple hooks can be used simultaneously
log_config = dict(interval=5, 
    hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook')
    ])
# register hooks to runner and those hooks will be invoked automatically
runner.register_training_hooks(
    lr_config=lr_config,
    optimizer_config=optimizer_config,
    checkpoint_config=checkpoint_config,
    log_config=log_config
)
if world_size > 1:
    runner.register_hook(DistEvalHook(val_dataloader, interval=10))
else:
    runner.register_hook(EvalHook(val_dataloader, interval=10))

runner.run(
    [train_loader],
    [('train',1)]
)
if world_size > 1:
    dist.destroy_process_group()