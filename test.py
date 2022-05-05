import torch
from stcn.para import get_config
from stcn import VOSMODEL
from stcn.dataset.dataloader import build_dataloader
from mmdet.datasets import build_dataset
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from stcn.testor import multi_gpu_test
from pathlib import Path
import mmcv
def main():
    cfg, meta, timestamp, distributed = get_config()
    model = VOSMODEL.build(cfg.model)
    wrap_fp16_model(model)
    load_checkpoint(model, cfg.load_from)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=len(cfg.gpu_ids),
            dist=distributed)

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    if hasattr(cfg, 'out_dir'):
        out_dir = cfg.out_dir
    else:
        out_dir = None
    results = multi_gpu_test(model, data_loader, out_dir=out_dir)
    
    if results is not None:
        mmcv.dump(results,Path(cfg.work_dir) / 'test_results_details.pkl')
        eval_res = dataset.evaluate(results)
        print(eval_res)


if __name__ == '__main__':
    with torch.no_grad():
        main()