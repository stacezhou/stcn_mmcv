# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from typing import  List, TypeVar
import random

T_co = TypeVar('T_co', covariant=True)

def compact_to(target, options, nums, top=True):
    'target: 10, options: 5,4,3,2, nums: 3 --> [5,3,2]'
    if nums == 1:
        for n in options:
            if n == target:
                return [n]
        return [None]

    for n in options:
        output = compact_to(target - n, options, nums - 1, False)
        if None not in output:
            return [n, *output]

    return [None]


class DistributedGroupSampler(Sampler):
    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 shuffle_videos = False,
                 random_skip = False,
                 max_objs_per_gpu=-1,
                 max_skip = 10,
                 max_per_frame = 3,
                 min_skip = 1,
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
        self.shuffle_videos = shuffle_videos
        self.random_skip = random_skip
        self.min_skip = min_skip
        self.nums_objs = [min(x, max_per_frame) for x in self.dataset.nums_objs]
        self.M = self.dataset.max_nums_frame
        self.max_skip = max_skip

        if self.dataset.test_mode:
            self.indices = [[x] for x in list(range(len(self.dataset)))]
            self.num_samples =  (len(self.nums_objs) // self.num_replicas + 1 ) * self.dataset.max_nums_frame
        else:
            m = len(self.nums_objs) // 2
            median = sorted(self.nums_objs)[m]
            if max_objs_per_gpu == -1:
                max_objs_per_gpu = median * self.samples_per_gpu
            self.max_objs_per_gpu = max_objs_per_gpu
            print(f'max objs per gpu is {max_objs_per_gpu}')
            self._collate()
    
    def _collate(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        random.seed(self.epoch + self.seed)

        n_vi_dict = defaultdict(list) # n vi : the video index is 'vi' and it has 'n' targets
        for vi,n in enumerate(self.nums_objs):
            n_vi_dict[n].append(vi)

        if self.shuffle_videos:
            for n,vis in n_vi_dict.items():
                random.shuffle(vis)

        ns = sorted(list(n_vi_dict.keys()))
        target = self.max_objs_per_gpu
        I_groups = []

        print({n:len(vi) for n,vi in n_vi_dict.items()})
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

        if self.shuffle_videos:
            random.shuffle(I_groups)

        max_size = len(I_groups)  // self.num_replicas * self.num_replicas
        self.I_groups = I_groups[:max_size]

        def clamp(f_id, v_l):
            if f_id >= v_l: # 0,1,2,3, 2,1, 2,3, 2,1, 2,3
                x = (f_id - v_l) // (v_l - 2)
                f_id = (f_id - v_l) % (v_l - 2)
                if x % 2 == 0:
                    f_id -= 2
                else:
                    f_id += 2
            return f_id

        indices_group = []
        for group in I_groups:
            if self.random_skip:
                f_id_group = []
                for vi in group:
                    f_ids = []
                    f_id = 0
                    for j in range(self.M):
                        f_id = f_id + random.randint(self.min_skip, self.max_skip)
                        f_ids.append(f_id)
                    f_ids = [self.M + clamp(f,self.M) for f in f_ids]
                    f_id_group.append(f_ids)
            else:
                f_id_group = [list(range(i*self.M, (i+1)*self.M)) for i in group]
            transposed = list(zip(*f_id_group))
            indices_group.append(transposed)
        
        self.indices = [x for group in indices_group for x in group]
        self.num_samples = len(self.indices) // self.num_replicas

    def __iter__(self):
        # subsample
        offset = self.num_samples * self.rank
        indices = self.indices[offset:offset + self.num_samples]

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
                yield [i for ids in batch for i in ids]
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield [i for ids in batch for i in ids]
    def __len__(self):
        return len(self.sampler)
