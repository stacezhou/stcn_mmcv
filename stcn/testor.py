# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time
from .dataset.metric import split_object_masks, metric_frame_JF as metric_JF
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from pathlib import Path
from PIL import Image

def multi_gpu_test(model, data_loader, tmpdir='/tmp/stcn', out_dir = None,gpu_collect=False, do_evaluate = False):
    """Test model with multiple gpus.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    palette = Image.open(data_loader.dataset.palette).getpalette()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            mask = model(return_loss=False, **data)[0]
            if mask is None:
                continue

            img_metas = data['img_metas'].data[0][0]
            H_,W_ ,_ = img_metas['pad_shape']
            h,w,c =  img_metas['ori_shape']
            H,W = mask.shape
            assert H == H_ and W == W_
            mask = mask[:h,:w]

            if out_dir is not None:
                filename = img_metas['ori_filename']
                out_path = Path(out_dir) / 'Annotations' / (filename[:-4] + '.png')
                out_path.parent.mkdir(parents=True, exist_ok=True)
                im = Image.fromarray(mask)
                im.putpalette(palette)
                im.save(out_path)

            if do_evaluate is not None:
                'compute score'
                gt_mask = data['gt_mask'].data[0][0].squeeze(0)[:h,:w]
                labels = img_metas['labels'][1:]
                pred, gt = split_object_masks(mask, gt_mask, labels)
                JF = metric_JF(pred, gt)
                result = [JF]
            else:
                result = [None]


        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            while not Path(part_file).exists():
                print('waitting')
                time.sleep(1)
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results
