from torch.utils.data import Dataset
from pathlib import Path
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import DATASETS
import mmcv
from .utils import generate_meta
import random
from collections import defaultdict

def sub_path(a_path, b_path):
    b_parts = b_path.parts
    a_parts = a_path.parts
    assert b_parts == a_parts[:len(b_parts)]
    return Path().joinpath(*a_parts[len(b_parts):])

Path.__sub__ = sub_path

@DATASETS.register_module()
class VOSStaticDataset(Dataset):
    def __init__(self, 
                    image_root, 
                    mask_root, 
                    pipeline=[], 
                    test_mode = False,
                    palette = None,
                    *k,**kw):

        self.palette = palette
        self.test_mode = test_mode
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.images = [str(x - self.image_root) for x in self.image_root.rglob('*.jpg')]
        self.masks = [str(x - self.mask_root) for x in self.mask_root.rglob('*.png')]
        assert len(self.masks) == len(self.images)

        self.pipeline = Compose(pipeline)
        
        
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        image = self.images[index]
        mask = image[:-4] + '.png'
        data = {
            'img_prefix' : str(self.image_root),
            'img_info':{'filename': str(image)},
            'ann_info': {
                'masks' : str(self.mask_root / mask)
                },
        }
        data = self.pipeline(data)
        return data
    
    def get_indices(self, samplers_per_gpu):
        indices = [list(range(i,i+samplers_per_gpu)) 
            for i in range(0,len(self),samplers_per_gpu)]
        random.shuffle(indices)
        return indices

    def flat_fn(self, batch_index):
        batch_index = batch_index[0:1] * len(batch_index)
        return [i for ids in batch_index for i in ids]

@DATASETS.register_module()
class VOSDataset(Dataset):
    def __init__(self, 
            image_root, 
            mask_root, 
            pipeline=[], 
            frame_limit = 20, 
            palette = None,
            wo_mask_pipeline = [], 
            test_mode=False,
            shuffle_videos = False,
            random_skip = False,
            max_objs_per_gpu=8,
            max_skip = 10,
            max_per_frame = 3,
            min_skip = 1,
            **kw):

        # pipeline
        self.pipeline = Compose(pipeline)
        self.wo_mask_pipeline = Compose(wo_mask_pipeline)
        
        # image & mask
        self.image_root = image_root
        self.mask_root = mask_root

        meta_stcn = Path(image_root) / 'meta_stcn.json'
        if meta_stcn.exists():
            self.data_infos = mmcv.load(str(meta_stcn))
        else:
            self.data_infos = generate_meta(image_root, mask_root)
            mmcv.dump(self.data_infos, meta_stcn)
        self.videos = sorted(list(self.data_infos.keys()))

        # 
        all_nums_frames = [v['nums_frame'] for k,v in self.data_infos.items()]
        video_M = max(all_nums_frames) 
        self.M = min(video_M, frame_limit) 

        self.nums_objs = [self.data_infos[v]['nums_obj'] for v in self.videos]

        # random skip
        self.shuffle_videos = shuffle_videos
        self.random_skip = random_skip
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.nums_objs = [min(x, max_per_frame) for x in self.nums_objs]

        self.max_objs_per_gpu = max(max_objs_per_gpu,2)

        # for output mask
        self.palette = palette
        self.test_mode = test_mode
        
        
    def __len__(self):
        return self.M * len(self.videos)


    def __getitem__(self, index):
        if not self.test_mode:
            return self.prepare_train_data(index)
        else:
            return self.prepare_test_data(index)
    
    def prepare_train_data(self, index):
        v_id = index // self.M
        v = self.videos[v_id]
        f_id = index % self.M
        flag = 'new_video' if f_id == 0 else ''
        v_l = self.data_infos[v]['nums_frame']
        if f_id >= v_l: # 0,1,2,3, 2,1, 2,3, 2,1, 2,3
            x = (f_id - v_l) // (v_l - 2)
            f_id = (f_id - v_l) % (v_l - 2)
            if x % 2 == 0:
                f_id -= 2
            else:
                f_id += 2
        image, mask = self.data_infos[v]['frame_and_mask'][f_id]
        data = {
            'flag'  : flag,
            'labels' : self.data_infos[v]['labels'],
            'img_prefix' : self.image_root,
            'img_info':{'filename': image},
            'ann_info': {
                'masks' : str(Path(self.mask_root) / mask) ,
                },
        }
        
        data = self.pipeline(data)
        return data
    
    def prepare_test_data(self, index):
        v_id = index // self.M
        v = self.videos[v_id]
        f_id = index % self.M
        flag = 'new_video' if f_id == 0 else ''
        v_l = self.data_infos[v]['nums_frame']
        if f_id >= v_l: # 0,1,2,3, 2,1,0, 1,2,3, 2,1,0, 1,2,3
            return {}
        
        
        image, mask = self.data_infos[v]['frame_and_mask'][f_id]
        mask = str(Path(self.mask_root) / mask) if mask is not None else None
        data = {
            'flag'  : flag,
            'labels' : self.data_infos[v]['labels'],
            'img_prefix' : self.image_root,
            'img_info':{'filename': image},
            'ann_info': {
                'masks' : mask,
                },
        }
        if mask is None:
            data = self.wo_mask_pipeline(data)
        else:
            data = self.pipeline(data)
        return data

    def evaluate(self, results, logger, **kwargs):
        J = [x['J'].mean() for x in results if x is not None]
        F = [x['F'].mean() for x in results if x is not None]
        import numpy as np
        J = np.array(J).mean()
        F = np.array(F).mean()
        return {
            'mIoU':J,
            'F':F
        }

    
    def flat_fn(self, batch_index):
        return [i for ids in batch_index for i in ids]
    def get_indices(self, samples_per_gpu):
        if self.test_mode:
            return [[x] for x in range(len(self))]
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
            n_group = compact_to(target, ns, samples_per_gpu)
            while None in n_group and target > 0:
                target -= 1
                n_group = compact_to(target, ns, samples_per_gpu)
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
        
        indices = [x for group in indices_group for x in group]
        return indices


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