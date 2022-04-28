from torch.utils.data import Dataset
from pathlib import Path
from random import randint
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import DATASETS
import mmcv
import numpy as np
from .utils import generate_meta


def listdir(path, complete_path = True):
    if complete_path:
        return sorted([str(d) for d in Path(path).iterdir()])
    else:
        return sorted([str(d.name) for d in Path(path).iterdir()])

def listfile(path, pattern, complete_path = True):
    if complete_path:
        return sorted([str(f) for f in Path(path).glob(pattern=pattern)])
    else:
        return sorted([str(f.name) for f in Path(path).glob(pattern=pattern)])

@DATASETS.register_module()
class StaticDataset(Dataset):
    def __init__(self,  pipeline=[], num_frames=3, image_root=None,video_root=None):
        assert image_root is not None or video_root is not None
        if image_root is not None:
            self.images = listfile(image_root,'*.jpg')
            self.masks = listfile(image_root, '*.png')
            assert len(self.images) == len(self.masks)
        else:
            self.images = []
            self.masks = []
            for image_root in listdir(video_root):
                images = listfile(image_root,'*.jpg')
                masks = listfile(image_root, '*.png')
                assert len(images) == len(masks)
                self.images += images
                self.masks += masks

        self.num_frames = num_frames
        self.pipeline = Compose(pipeline)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        data = {
            'img_prefix' : None,
            'img_info':{'filename': image},
            'ann_info': {'masks' : mask}
        }
        data = self.pipeline(data)
            
        return data
        

    def __iter__(self):
        idx = randint(0,len(self)-1)
        output = []
        for i in range(self.num_frames):
            output.append(self[idx])
        return output

@DATASETS.register_module()
class VOSTrainDataset(Dataset):
    def __init__(self, image_root, mask_root, pipeline=[],repeat_dataset = 1, max_skip=10, num_frames=3, min_skip=1, test_mode=False, **kw):

        self.pipeline = Compose(pipeline)
        self.repeat = repeat_dataset

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
        # self._set_group_flag()

        self.nums_objs = [self.data_infos[v]['nums_obj'] for v in self.videos]
        
        

    # def _set_group_flag(self):
    #     """Set flag according to image aspect ratio.

    #     Images with aspect ratio greater than 1 will be set as group 1,
    #     otherwise group 0.
    #     """
    #     self.flag = np.zeros(len(self), dtype=np.uint8)
    #     for i,v in enumerate(self.videos):
    #         H = self.data_infos[v]['img_height']
    #         W = self.data_infos[v]['img_width']
    #         if H / W > 1:
    #             self.flag[i] = 1
        
    def __len__(self):
        self.max_nums_frame * len(self.videos)

    
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
        # chosen_frames = self._random_choose_frames(frames)
        # # chosen_frames = frames[:self.num_frames]
        # data_batch = []
        # for i,(image,mask) in enumerate(chosen_frames):
        #     data = {
        #         'img_prefix' : self.image_root,
        #         'img_info':{'filename': image},
        #         'ann_info': {
        #             'masks' : str(Path(self.mask_root) / mask) ,
        #             'flag'  : 'new_video' if i == 0 else ''
        #             },
        #     }
        #     data = self.pipeline(data)
        #     data_batch.append(data)

        # return data_batch
    
    def prepare_train_data(self, index):
        v_id = index // self.max_nums_frame
        v = self.videos[v_id]
        f_id = index % self.max_nums_frame
        flag = 'new_video' if f_id == 0 else ''
        v_l = self.data_infos[v]['nums_frame']
        if f_id >= v_l: # 0,1,2,3, 2,1,0, 1,2,3, 2,1,0, 1,2,3
            x = (f_id - v_l) // (self.max_nums_frame - 1)
            f_id = (f_id - v_l) % (self.max_nums_frame - 1)
            if x % 2 == 0:
                f_id -= 2
            else:
                f_id += 1
                
        image, mask = self.data_infos[v]['frame_and_mask'][f_id]
        data = {
            'flag'  : flag,
            'img_prefix' : self.image_root,
            'img_info':{'filename': image},
            'ann_info': {
                'masks' : str(Path(self.mask_root) / mask) ,
                },
        }
        data = self.pipeline(data)
        return data
    
    def prepare_test_data(self, index):
        pass



@DATASETS.register_module()
class VOSTestDataset(Dataset):

    def __init__(self, image_root, ref_mask_root, gt_mask_root = None, pipeline = None):
        self.videos = listdir(image_root)
        self.mask_root = ref_mask_root
        self.gt_mask_root = gt_mask_root
        self.pipeline = pipeline
        self.has_read_frames = False

    def read_frames(self):
        if self.has_read_frames:
            return
        self.frames = None
        pass 

    def __len__(self):
        self.read_frames()
        return len(self.videos)

    def __getitem__(self, index):
        self.read_frames()
        return self.frames[index]
