import mmcv
import torch
from stcn.para import get_config
from stcn import VOSMODEL
from stcn.dataset.dataloader import build_dataloader
from mmdet.datasets import DATASETS
from mmcv.runner.checkpoint import load_checkpoint
from mmcv.runner import (get_dist_info, load_checkpoint, wrap_fp16_model)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel


def test_dataset(model, data_loader):
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader.dataset))

    model.eval()
    for data in data_loader:
        img = data['img'].data[0]
        gt_mask = data['gt_mask'].data[0]
        img_metas = data['img_metas'].data[0]
        output = model(img=img,gt_mask=gt_mask,img_metas=img_metas,return_loss = False)

        if rank == 0:
            batch_size = len(data)
            for _ in range(batch_size * world_size):
                prog_bar.update()

def main():
    cfg, meta, timestamp, distributed = get_config()
    model = VOSMODEL.build(cfg.model)
    wrap_fp16_model(model)
    # load_checkpoint(model, cfg.load_from)

    dataset = DATASETS.build(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            max_objs_per_gpu=999,
            nums_frame=1,
            num_gpus=len(cfg.gpu_ids),
            dist=distributed)
    # put model on gpus
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        test_dataset(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        test_dataset(model, data_loader)



if __name__ == '__main__':
    with torch.no_grad():
        main()