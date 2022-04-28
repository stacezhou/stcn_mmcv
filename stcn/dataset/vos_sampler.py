# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from typing import  List, TypeVar

T_co = TypeVar('T_co', covariant=True)
# class DistributedGroupSampler(Sampler):

class DistributedGroupSampler(Sampler):
    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 max_objs_per_gpu=4,
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
        for vi,n in self.dataset.nums_objs:
            self.flag[n].append(vi)
        # todo random shuffle
        n_vi = [(n,vi) for n,vis in n_vi_dict.items() for vi in vis]
        N = [n for n,vi in n_vi]
        I = [vi for n,vi in n_vi]

        r = len(N) - 1
        S = N[r] + N[0] * (self.samples_per_gpu)
        while S > self.max_objs_per_gpu and r:
            r -= 1
            S = N[r] + N[0] * (self.samples_per_gpu)
        assert r, 'max_objs_per_gpu is too small'

        I_groups = []
        step = self.samples_per_gpu - 1
        l = step
        while l < r:
            S = N[r] + sum([N[max(l-i,0)] for i in range(step)])
            while S > self.max_objs_per_gpu:
                l -= 1
                S = N[r] + sum([N[max(l-i,0)] for i in range(step)])
            group = [I[max(l-i,0)] for i in range(step)]
            group.append(I[r])
            I_groups.append(group)
            l += step
            r -= 1

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

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        self.sampler = sampler

    def __iter__(self):
        for batch in self.sampler:
            yield batch

    def __len__(self):
        return len(self.sampler)