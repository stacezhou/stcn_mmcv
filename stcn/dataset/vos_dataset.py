from torch.utils.data import Dataset
from pathlib import Path
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import DATASETS
import mmcv
from .utils import generate_meta


@DATASETS.register_module()
class VOSDataset(Dataset):
    def __init__(self, image_root, mask_root, pipeline=[],
        frame_limit = 20, palette = None,
        wo_mask_pipeline = [], max_objs_per_frame = 2, test_mode=False, **kw):

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

        all_nums_frames = [v['nums_frame'] for k,v in self.data_infos.items()]
        video_max_nums_frame = max(all_nums_frames) 
        self.max_nums_frame = min(video_max_nums_frame, frame_limit) 
        self.test_mode = test_mode
        self.max_objs_per_frame = max_objs_per_frame

        self.nums_objs = [self.data_infos[v]['nums_obj'] for v in self.videos]

        self.seed = 0
        self.palette = palette
        
        
    def __len__(self):
        return self.max_nums_frame * len(self.videos)


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

    def evaluate(self, results, logger, **kwargs):
        # for name, val in eval_res.items():
        #     runner.log_buffer.output[name] = val
        pass