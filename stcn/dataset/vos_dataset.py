from torch.utils.data import Dataset
from pathlib import Path
from random import randint
from mmdet.datasets import DATASETS

def listdir(path):
    return sorted([d.name for d in Path(path).iterdir()])
def listfile(path, pattern):
    return sorted([f.name for f in Path(path).glob(pattern=pattern)])

@DATASETS.register_module()
class StaticDataset(Dataset):
    def __init__(self,  transforms, num_frames, image_mask_root=None,video_root=None):
        assert image_mask_root is not None or video_root is not None
        if image_mask_root is not None:
            self.images = listfile(image_mask_root,'*.jpg')
            self.masks = listfile(image_mask_root, '*.png')
            assert len(self.images) == len(self.masks)
        else:
            self.images = []
            self.masks = []
            for v in listdir(video_root):
                image_mask_root = Path(video_root) / v
                images = listfile(image_mask_root,'*.jpg')
                masks = listfile(image_mask_root, '*.png')
                assert len(images) == len(masks)
                self.images += images
                self.masks += masks
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        

    def __iter__(self):
        idx = randint(0,len(self)-1)
        yield self[idx]


class VOSTrainDataset(Dataset):
    def __init__(self, image_root, mask_root, transforms, max_skip=10, num_frames=3, min_skip=1):
        mask_videos = listdir(mask_root)
        image_videos = listdir(image_root)
        self.videos = sorted(list(set(mask_videos) - set(image_videos)))
        self.pipeline = transforms
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.num_frames = num_frames

        self.frames = dict()
        for v in self.videos:
            mask_frames = listfile(Path(mask_root) / v, '*.png')
            image_frames = listfile(Path(image_root) / v, '*.jpg')
            assert len(mask_frames) == len(image_frames), f'nums of JPG & mask not match {v}'
            self.frames[v] = list(zip(image_frames,mask_frames))
        
    def __len__(self):
        return len(self.videos)
    
    def _random_choose_frames(self,frame_list):
        pass

    def __getitem__(self, index):
        v = self.videos[index]
        frames = self.frames[v]
        chosen_frames = self._random_choose_frames(frames)
        image_frames,mask_frames = zip(*chosen_frames)
        return image_frames,mask_frames


class VOSTestDataset(Dataset):

    def __init__(self, image_root, ref_mask_root, gt_mask_root = None, transforms = None):
        self.videos = listdir(image_root)
        self.mask_root = ref_mask_root
        self.gt_mask_root = gt_mask_root
        self.transforms = transforms
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
