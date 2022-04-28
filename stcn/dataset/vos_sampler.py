# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from typing import  List, TypeVar

T_co = TypeVar('T_co', covariant=True)
# class DistributedGroupSampler(Sampler):

def compact_to(target, avi_nums, times, top=True):
    if times == 1:
        for n in avi_nums:
            if n == target:
                return [n]
        return [None]

    for n in avi_nums:
        output = compact_to(target - n, avi_nums, times - 1, False)
        if None not in output:
            return [n, *output]

    return [None]


class DistributedGroupSampler(Sampler):
    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 max_objs_per_gpu=16,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0
        self.max_objs_per_gpu = max_objs_per_gpu

        assert hasattr(self.dataset, 'nums_objs')
        assert hasattr(self.dataset, 'max_nums_frame')

        self.M = self.dataset.max_nums_frame
        self._collate()
    
    def _collate(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        n_vi_dict = defaultdict(list)
        for vi,n in enumerate(self.dataset.nums_objs):
            n_vi_dict[n].append(vi)
        # todo random shuffle

        ns = sorted(list(n_vi_dict.keys()))
        target = self.max_objs_per_gpu
        I_groups = []

        print(ns)
        while True:
            ns = [n for n in ns if len(n_vi_dict[n])]
            n_group = compact_to(target, ns, self.samples_per_gpu)
            while None in n_group and target > 0:
                target -= 1
                n_group = compact_to(target, ns, self.samples_per_gpu)
            if target == 0:
                break

            while True:
                try:
                    i_group = [n_vi_dict[n].pop() for n in n_group]
                except:
                    break
                I_groups.append(i_group)


        # todo shuffle
        max_size = len(I_groups)  // self.num_replicas * self.num_replicas
        self.I_groups = I_groups[:max_size]

        indices_group = []
        for group in I_groups:
            f_id_group = [list(range(i*self.M, (i+1)*self.M)) for i in group]
            transposed = list(zip(*f_id_group))
            indices_group.append(transposed)
        
        self.indices = [x for group in indices_group for x in group]
        self.num_samples = len(self.indices) // self.num_replicas

    def __iter__(self):
        # subsample
        offset = self.num_samples * self.rank
        indices = self.indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class BatchSampler(Sampler[List[int]]):

    def __init__(self, sampler: Sampler[int], T_batch_size: int, drop_last: bool) -> None:
        self.sampler = sampler
        self.batch_size = T_batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for ids in self.sampler:
            batch.append(ids)
            if len(batch) == self.batch_size:
                yield [i for ids in batch for i in ids]
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield [i for ids in batch for i in ids]
    def __len__(self):
        return len(self.sampler)
