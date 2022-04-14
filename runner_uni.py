from datetime import datetime
from mmcv.runner import EpochBasedRunner as _EpochBasedRunner
from mmcv.runner import IterBasedRunner as _IterBasedRunner
from mmcv.runner.hooks import Fp16OptimizerHook
from mmcv.utils import get_logger
from mmcv.parallel import MMDistributedDataParallel as DDP
from mmcv.runner import DistSamplerSeedHook
from mmcv.runner.hooks import TensorboardLoggerHook as _TensorboardLoggerHook
from mmcv.runner.hooks import TextLoggerHook as _TextLoggerHook
class TensorboardLoggerHook(_TensorboardLoggerHook):
    def after_val_iter(self, runner):
        if not self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.by_epoch and self.every_n_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

class TextLoggerHook(_TextLoggerHook):
    def after_val_iter(self, runner):
        if not self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.by_epoch and self.every_n_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

class EpochBasedRunner(_EpochBasedRunner):
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
            load_network = None,
            resume_model = None
            ):
        if exp_id is not None:
            work_dir = f'work_dir/{exp_id}'
            logger = get_logger(exp_id)
        else:
            work_dir = f'work_dir/exp_{datetime.now().strftime("%b%d_%H.%M.%S")}'
            logger = get_logger('exp')
            
        super().__init__(model=model, batch_processor=batch_processor,
            optimizer=optimizer,  
            max_iters=max_iters, 
            max_epochs=max_epochs,
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
        )

        self.register_hook(DistSamplerSeedHook())

        if load_network is not None:
            self.load_checkpoint(load_network)
        if resume_model is not None:
            self.resume(resume_model)

    def run(self, data_loaders, workflow, max_epochs=None,epochs=None, **kwargs):
        if epochs is not None:
            max_epochs = self.epoch + epochs

        super().run(data_loaders, workflow, max_epochs, **kwargs)

class IterBasedRunner(_IterBasedRunner):

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
            load_network=None,
            resume_model=None
            ):
        if exp_id is not None:
            work_dir = f'work_dir/{exp_id}'
            logger = get_logger(exp_id)
        else:
            work_dir = f'work_dir/exp_{datetime.now().strftime("%b%d_%H.%M.%S")}'
            logger = get_logger('exp')
            
        super().__init__(model=model, batch_processor=batch_processor,
            optimizer=optimizer,  
            max_iters=max_iters, 
            max_epochs=max_epochs,
            meta = meta,
            logger=logger,
            work_dir=work_dir)
        
        if checkpoint_config is None and isinstance(checkpoint_interval, int):
            checkpoint_config = dict(interval = checkpoint_interval)

        if log_config is None and isinstance(log_interval, int):
            self.register_hook(
                TensorboardLoggerHook(interval=log_interval,by_epoch=False),
                priority='VERY_LOW'
            )
            self.register_hook(
                TextLoggerHook(interval=log_interval,by_epoch=False),
                priority='VERY_LOW'
            )
        
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
        )
        if load_network is not None:
            self.load_checkpoint(load_network)

        if resume_model is not None:
            self.resume(resume_model)

    def run(self, data_loaders, workflow, max_iters=None,iters=None, **kwargs):
        if iters is not None:
            max_iters = self.iter + iters

        _iter = self.iter
        super().run(data_loaders, workflow, max_iters,_iter=_iter, **kwargs)