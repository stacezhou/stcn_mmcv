from torch.utils.data import Dataset,ConcatDataset as _ConcatDataset
from pathlib import Path
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import DATASETS
import mmcv
import numpy as np
from .utils import generate_meta
import random
from collections import defaultdict
import pandas as pd

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
                    nums_frame = 4,
                    palette = None,
                    *k,**kw):

        self.palette = palette
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.images = [str(x - self.image_root) for x in self.image_root.rglob('*.jpg')]
        self.masks = [str(x - self.mask_root) for x in self.mask_root.rglob('*.png')]
        assert len(self.masks) == len(self.images)

        self.test_mode = False
        self.pipeline = Compose(pipeline)
        self.nums_frame = nums_frame
        
        
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        if index >= len(self):
            index = index % len(self)
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
        indices = [[x]*self.nums_frame for x in indices]
        random.shuffle(indices)
        return indices

    def flat_fn(self, batch_index):
        return [i for ids in batch_index for i in ids]

    def evaluate(self, results, logger, **kwargs):
        return {'no_support_static_validate':1}

@DATASETS.register_module()
class VOSDataset(Dataset):
    def __init__(self, 
            image_root, 
            mask_root, 
            pipeline=[], 
            palette = None,
            wo_mask_pipeline = [], 
            test_mode=False,
            nums_frame = 4,
            max_objs_per_gpu=8,
            max_skip = 10,
            max_objs_per_frame = 3,
            min_skip = 1,
            **kw):

        # pipeline
        self.pipeline = Compose(pipeline)
        self.wo_mask_pipeline = Compose(wo_mask_pipeline)
        
        # image & mask
        self.image_root = image_root
        self.mask_root = mask_root

        self.test_mode = test_mode
        if test_mode:
            min_skip = 1
            max_skip = 1
            nums_frame = 1
        meta_stcn = Path(image_root) / 'meta_stcn.json'
        if meta_stcn.exists():
            self.data_infos = mmcv.load(str(meta_stcn))
        else:
            self.data_infos = generate_meta(image_root, mask_root)
            mmcv.dump(self.data_infos, meta_stcn)
        self.data_infos = {k:v for k,v in self.data_infos.items() 
                            if v['nums_frame'] > min_skip * nums_frame}
        self.videos = sorted(list(self.data_infos.keys()))

        # 
        all_nums_frames = [v['nums_frame'] for k,v in self.data_infos.items()]
        self.M = max(all_nums_frames) 
        if not test_mode:
            self.M = (self.M+nums_frame) // nums_frame * nums_frame

        self.nums_frame = nums_frame

        self.nums_objs = [self.data_infos[v]['nums_obj'] for v in self.videos]

        # random skip
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.nums_objs = [min(x, max_objs_per_frame) for x in self.nums_objs]

        self.max_objs_per_gpu = max(max_objs_per_gpu,2)

        # for output mask
        self.palette = palette
        
        
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
        if index < 0:
            return {'drop_flag':-1}
        v_id = index // self.M
        v = self.videos[v_id]
        f_id = index % self.M
        flag = 'new_video' if f_id == 0 else ''
        v_l = self.data_infos[v]['nums_frame']
        if f_id >= v_l: # 0,1,2,3, 2,1,0, 1,2,3, 2,1,0, 1,2,3
            return {'drop_flag':0}
        
        
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

    def evaluate(self, results, logger=None, **kwargs):
        results = [results[i*self.M:(i+1)*self.M] for i in range(len(self.videos))]
        results_by_video = dict()
        all_JF = 0
        for v,result in zip(self.videos, results):
            J = [x['J'].mean() for x in result if not isinstance(x,int)]
            F = [x['F'].mean() for x in result if not isinstance(x,int)]
            J = np.array(J).mean().tolist()
            F = np.array(F).mean().tolist()
            JF = (J+F) / 2
            results_by_video[f'{v}']=JF
            all_JF += JF
        all_JF /= len(self.videos)

        results_by_frame = dict()
        for i in range(self.M):
            J = [result[i]['J'].mean() for result in results if not isinstance(result[i],int)]
            F = [result[i]['F'].mean() for result in results if not isinstance(result[i],int)]
            J = np.array(J).mean().tolist()
            F = np.array(F).mean().tolist()
            JF = (J+F) / 2
            results_by_frame[f'{i:02d}']=JF

        pd.set_option('display.max_rows', 200)       
        video_result =     pd.DataFrame.from_dict(results_by_video, orient='index', columns=['JF']) .sort_values('JF')
        frame_result =     pd.DataFrame.from_dict(results_by_frame, orient='index', columns=['JF']) .sort_index()
        if logger is not None:
            logger.info(video_result)
            logger.info(frame_result)
            logger.info(all_JF)
        else:
            print(video_result)
            print(frame_result)
            print(all_JF)
            video_result.to_csv('latest_result_video.json')
        res_eval = dict()
        if Path('evaluate_meta.json').exists():
            meta = mmcv.load('evaluate_meta.json')
            index_set = set(video_result.index)
            for tag,v_list in meta.items():
                index = list(set(v_list) & index_set)
                res_eval[tag] = video_result.loc[index].mean().item()
        return {
            'mIoU':all_JF,
            **res_eval
        }

    
    def flat_fn(self, batch_index):
        return [i for ids in batch_index for i in ids]
    def get_indices(self, samples_per_gpu):
        if self.test_mode:
            return [[[x]] for x in range(len(self))]
        n_vi_dict = defaultdict(list) # n vi : the video index is 'vi' and it has 'n' targets
        for vi,n in enumerate(self.nums_objs):
            n_vi_dict[n].append(vi)

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

        random.shuffle(I_groups)

        indices = []
        for group in I_groups:
            f_id_group = []
            for vi in group:
                f_ids = self.random_pick_frames(vi)
                f_id_group.append(f_ids)
            transposed = list(zip(*f_id_group))
            indices.append(transposed)
        
        return indices
    
    def random_pick_frames(self, vi):
        v = self.videos[vi]
        max_fid = min(self.data_infos[v]['nums_frame'] - 1, self.M)

        skips = [random.randint(self.min_skip,self.max_skip) 
                for i in range(self.nums_frame - 1)]
        trial_times = 0
        while sum(skips) > max_fid and trial_times < 10:
            skips = [random.randint(self.min_skip,self.max_skip) 
                    for i in range(self.nums_frame - 1)]
            trial_times += 1
        if trial_times == 10:
            skips = [self.min_skip for i in range(self.nums_frame - 2)]
            skips += [random.randint(0,min(self.max_skip, max_fid-sum(skips)))]
        random.shuffle(skips)
        start = random.randint(0, max_fid - sum(skips))
        offsets = [0] 
        for skip in skips:
            offsets.append(offsets[-1] + skip)
        fids = [x+start+self.M * vi for x in offsets]
        assert fids[0] >= vi*self.M and fids[-1] < (vi+1)*self.M
        return fids



        

DATASETS._module_dict.pop('ConcatDataset')
@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    def __init__(self, datasets) -> None:
        super().__init__(datasets)

    @property
    def test_mode(self):
        for ds in self.datasets:
            assert not ds.test_mode, 'concat dataset only support train mode'
        return False
    @property
    def nums_frame(self):
        nums_f = [ds.nums_frame for ds in self.datasets]
        assert len(set(nums_f)) == 1,'nums_frame of multi dataset should be same'
        return nums_f[0]

    def flat_fn(self, batch_index):
        return [i for ids in batch_index for i in ids]
    def get_indices(self, samples_per_gpu):
        indices = []
        cums = [0] + self.cummulative_sizes
        for ds,cum in zip(self.datasets, cums[:-1]):
            indice = ds.get_indices(samples_per_gpu)
            cum_indice = [[[cum + idx for idx in ids ] for ids in idx_group] for idx_group in indice]
            indices.extend(cum_indice)
        return indices
    
    def evaluate(self, results, logger=None, **kwargs):
        return {'not_support_concat_test':1}

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