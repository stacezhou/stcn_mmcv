from mmcv.runner.hooks import EvalHook as EvalHook_
from mmcv.runner.hooks import DistEvalHook as DistEvalHook_
import warnings

class EvalHook(EvalHook_):

    def __init__(self, dataloader, start=None, interval=1, by_epoch=True, save_best=None, rule=None, test_fn=None, greater_keys=None, less_keys=None, out_dir=None, file_client_args=None,evaluate_fn=None, **eval_kwargs):
        super().__init__(dataloader, start, interval, by_epoch, save_best, rule, test_fn, greater_keys, less_keys, out_dir, file_client_args, **eval_kwargs)
        self.evaluate_fn = evaluate_fn

    
    def evaluate(self, runner, results):

        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.evaluate_fn(results)

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            # If the performance of model is pool, the `eval_res` may be an
            # empty dict and it will raise exception when `self.save_best` is
            # not None. More details at
            # https://github.com/open-mmlab/mmdetection/issues/6265.
            if not eval_res:
                warnings.warn(
                    'Since `eval_res` is an empty dict, the behavior to save '
                    'the best checkpoint will be skipped in this evaluation.')
                return None

            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None


class DistEvalHook(DistEvalHook_):
    def __init__(self, dataloader, start=None, interval=1, by_epoch=True, save_best=None, rule=None, test_fn=None, greater_keys=None, less_keys=None, broadcast_bn_buffer=True, tmpdir=None, evaluate_fn=None, gpu_collect=False, out_dir=None, file_client_args=None, **eval_kwargs):
        super().__init__(dataloader, start, interval, by_epoch, save_best, rule, test_fn, greater_keys, less_keys, broadcast_bn_buffer, tmpdir, gpu_collect, out_dir, file_client_args, **eval_kwargs)

        self.evaluate_fn = evaluate_fn

    
    def evaluate(self, runner, results):

        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.evaluate_fn(results)

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            # If the performance of model is pool, the `eval_res` may be an
            # empty dict and it will raise exception when `self.save_best` is
            # not None. More details at
            # https://github.com/open-mmlab/mmdetection/issues/6265.
            if not eval_res:
                warnings.warn(
                    'Since `eval_res` is an empty dict, the behavior to save '
                    'the best checkpoint will be skipped in this evaluation.')
                return None

            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None