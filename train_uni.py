from model_uni import STCNModel
from mmcv.runner import EpochBasedRunner
from mmcv.utils import get_logger

from dataset.static_dataset import StaticTransformDataset
from dataset.vos_dataset import VOSDataset
from util.hyper_para import HyperParameters
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch
import numpy as np
import random
from pathlib import Path
from util.load_subset import load_sub_davis, load_sub_yv

para = HyperParameters()
para.parse()
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

if para['benchmark']:
    torch.backends.cudnn.benchmark = True

stcn_model = STCNModel()

davis_im_path = Path(para['davis_root']) / '2017' / 'trainval' / 'JPEGImages' / '480p'
davis_mask_path = Path(para['davis_root']) / '2017' / 'trainval' / 'Annotations' / '480p'
max_skip = 20

davis_dataset = VOSDataset(davis_im_path,davis_mask_path,max_skip, is_bl=False, subset=load_sub_davis())
train_loader = DataLoader(davis_dataset, para['batch_size'], num_workers=para['num_workers'],drop_last=True, pin_memory=True)
optimizer = optim.Adam(filter(
    lambda p: p.requires_grad, stcn_model.parameters()), lr=para['lr'], weight_decay=1e-7)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, para['steps'], para['gamma'])
logger = get_logger('stcn')
runner = EpochBasedRunner(
    model = stcn_model,
    optimizer=optimizer,
    work_dir='/tmp/debug',
    logger=logger,
    max_epochs=4
)
# learning rate scheduler config
lr_config = dict(policy='step', step=[2, 3])
# configuration of optimizer
optimizer_config = dict(grad_clip=None)
# configuration of saving checkpoints periodically
checkpoint_config = dict(interval=1)
# save log periodically and multiple hooks can be used simultaneously
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
# register hooks to runner and those hooks will be invoked automatically
runner.register_training_hooks(
    lr_config=lr_config,
    optimizer_config=optimizer_config,
    checkpoint_config=checkpoint_config,
    log_config=log_config)

runner.run(
    [train_loader],
    [('train',1)]
)