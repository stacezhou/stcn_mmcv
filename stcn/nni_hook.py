from mmcv.runner import HOOKS,LoggerHook
from mmcv.runner.dist_utils import master_only

@HOOKS.register_module()
class NNIHook(LoggerHook):

    def __init__(self,
                 metric_full_name = 'mIoU',
                 final_iter = 30000,
                 interval= 100,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=False):
        super(NNIHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.metric_full_name = metric_full_name
        self.final_iter = final_iter
    
    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if tag == 'val/' + self.metric_full_name:
                metric = val
                """@nni.report_intermediate_result(metric)"""
                if runner._epoch == self.final_epoch:
                    metric = val
                    """@nni.report_final_result(metric)"""