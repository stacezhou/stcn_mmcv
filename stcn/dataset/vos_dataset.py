from torch.utils.data import Dataset
from pathlib import Path
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import DATASETS
import mmcv
from .utils import generate_meta


@DATASETS.register_module()
class VOSDataset(Dataset):
    def __init__(self, image_root, mask_root, pipeline=[],wo_mask_pipeline = [], max_skip=10, num_frames=3, max_objs_per_frame = 2, min_skip=1, test_mode=False, **kw):

        self.pipeline = Compose(pipeline)
        self.wo_mask_pipeline = Compose(wo_mask_pipeline)

        self.image_root = image_root
        self.mask_root = mask_root

        meta_stcn = Path(image_root) / 'meta_stcn.json'
        if meta_stcn.exists():
            self.data_infos = mmcv.load(str(meta_stcn))
        else:
            self.data_infos = generate_meta(image_root, mask_root)
            mmcv.dump(self.data_infos, meta_stcn)
        self.videos = sorted(list(self.data_infos.keys()))

        self.min_skip = min_skip
        self.max_skip = max_skip
        self.num_frames = num_frames
        self.min_length = min_skip * (num_frames -1) + 1

        all_nums_frames = [v['nums_frame'] for k,v in self.data_infos.items()]
        self.max_nums_frame = max(all_nums_frames) 
        assert min(all_nums_frames)  > self.min_length, 'too big min_skip'
        self.test_mode = test_mode
        self.max_objs_per_frame = max_objs_per_frame

        self.nums_objs = [self.data_infos[v]['nums_obj'] for v in self.videos]

        self.seed = 0
        
        
    def __len__(self):
        return self.max_nums_frame * len(self.videos)

    # def _random_choose_frames(self,frame_list):
    #     assert self.num_frames < len(frame_list)
    #     offset = [0]
    #     flex_quota = len(frame_list) - self.min_length 
    #     start_idx = randint(0, flex_quota)
    #     flex_quota -= (start_idx - self.min_skip)
    #     for i in range(self.num_frames - 1):
    #         plus = min(self.max_skip, flex_quota)
    #         fq = randint(self.min_skip,plus) 
    #         flex_quota -= (fq - self.min_skip)
    #         offset.append(offset[-1] + fq)
 
    #     frames = [frame_list[start_idx + i] for i in offset]
    #     return frames

    def __getitem__(self, index):
        if not self.test_mode:
            return self.prepare_train_data(index)
        else:
            return self.prepare_test_data(index)
    
    def prepare_train_data(self, index):
        v_id = index // self.max_nums_frame
        v = self.videos[v_id]
        f_id = index % self.max_nums_frame
        flag = 'new_video' if f_id == 0 else ''
        v_l = self.data_infos[v]['nums_frame']
        if f_id >= v_l: # 0,1,2,3, 2,1,0, 1,2,3, 2,1,0, 1,2,3
            x = (f_id - v_l) // (v_l - 1)
            f_id = (f_id - v_l) % (v_l - 1)
            if x % 2 == 0:
                f_id -= 2
            else:
                f_id += 1
        
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
        v_id = index // self.max_nums_frame
        v = self.videos[v_id]
        f_id = index % self.max_nums_frame
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