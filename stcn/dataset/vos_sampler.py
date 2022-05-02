# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from typing import  List, TypeVar
import random

T_co = TypeVar('T_co', covariant=True)

class DistributedGroupSampler(Sampler):
    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 *k,**kw):
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
        self.indices = self.dataset.get_indices(self.samples_per_gpu)
        if self.dataset.test_mode:
            M = self.dataset.M
            self.num_samples = (len(self.indices) // M + self.num_replicas) // self.num_replicas * M
        else:
            self.num_samples = len(self.indices) // self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        random.seed(self.epoch + self.seed)

        # subsample
        self.indices = self.dataset.get_indices(self.samples_per_gpu)
        self.indices += self.indices[:self.num_samples]
        offset = self.num_samples * self.rank
        indices = self.indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class BatchSampler(Sampler[List[int]]):

    def __init__(self, sampler: Sampler[int], nums_frame: int, drop_last: bool) -> None:
        self.sampler = sampler
        self.batch_size = nums_frame
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for ids in self.sampler:
            batch.append(ids)
            if len(batch) == self.batch_size:
                yield self.sampler.dataset.flat_fn(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield self.sampler.dataset.flat_fn(batch)
    def __len__(self):
        return len(self.sampler)
