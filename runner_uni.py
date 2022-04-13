from datetime import datetime
from mmcv.runner import EpochBasedRunner as EpochBasedRunner_
from mmcv.runner.hooks import Fp16OptimizerHook
from mmcv.utils import get_logger
from mmcv.parallel import MMDistributedDataParallel as DDP
from mmcv.runner import DistSamplerSeedHook

class EpochBasedRunner(EpochBasedRunner_):
    def __init__(self, model, lr_config,
            batch_processor=None, optimizer=None, work_dir=None, 
            logger=None, meta={}, max_iters=None, max_epochs=None,
            # distributed_meta = None, 
            exp_id = None,
            amp = False,
            optimizer_config = None,
            checkpoint_interval = None,
            checkpoint_config = None,
            log_config = None,
            log_interval = None,
            ):
        if exp_id is not None:
            work_dir = f'work_dir/{exp_id}'
            logger = get_logger(exp_id)
        else:
            work_dir = f'work_dir/exp_{datetime.now().strftime("%b%d_%H.%M.%S")}'
            logger = get_logger('exp')
            
        super().__init__(model, batch_processor, optimizer,  
            max_iters, max_epochs,
            meta = meta,
            logger=logger,
            work_dir=work_dir,)
        
        if checkpoint_config is None and isinstance(checkpoint_interval, int):
            checkpoint_config = dict(interval = checkpoint_interval)

        if log_config is None and isinstance(log_interval, int):
            log_config = dict(interval= log_interval,
                hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook')
                ])
        
        if optimizer_config is None:
            if amp:
                optimizer_config = Fp16OptimizerHook(grad_clip=None,loss_scale='dynamic')
            else:
                optimizer_config = dict(grad_clip=None)

        self.register_training_hooks(
            lr_config=lr_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            log_config=log_config,
            timer_config=None
        )

        self.register_hook(DistSamplerSeedHook())
