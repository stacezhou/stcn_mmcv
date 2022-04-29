import mmcv
import torch
from stcn.para import get_config
from stcn import VOSMODEL
from stcn.dataset.dataloader import build_dataloader
from mmdet.datasets import DATASETS
from mmcv.runner import (get_dist_info, load_checkpoint, wrap_fp16_model)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from PIL import Image
from pathlib import Path

def test_dataset(model, data_loader, output_dir):
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader.dataset))


    palette = Image.open(data_loader.dataset.palette).getpalette()
    model.eval()
    for data in data_loader:
        if len(data) == 0:
            continue
        img = data['img'].data[0]
        if 'gt_mask' in data:
            gt_mask = data['gt_mask'].data[0]
        else:
            gt_mask = None
        img_metas = data['img_metas'].data[0]
        output = model(img=img,gt_mask=gt_mask,img_metas=img_metas,return_loss = False)

        mask = output['mask'][0]
        filename = output['img_metas'][0]['ori_filename']
        out_path = Path(output_dir) / 'Annotations' / (filename[:-4] + '.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(mask)
        im.putpalette(palette)
        im.save(out_path)

        if rank == 0:
            for _ in range(world_size):
                prog_bar.update()

def main():
    cfg, meta, timestamp, distributed = get_config()
    model = VOSMODEL.build(cfg.model)
    wrap_fp16_model(model)
    load_checkpoint(model, cfg.load_from)

    dataset = DATASETS.build(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            nums_frame=1,
            num_gpus=len(cfg.gpu_ids),
            dist=distributed)
    # put model on gpus
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        test_dataset(model, data_loader, output_dir=cfg.work_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        test_dataset(model, data_loader, output_dir=cfg.work_dir)



if __name__ == '__main__':
    with torch.no_grad():
        main()