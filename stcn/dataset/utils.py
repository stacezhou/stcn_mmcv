from os import path as osp
from pathlib import Path
import mmcv
from PIL import Image
import numpy as np

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

def read_labels(mask_path):
    labels = np.unique(Image.open(mask_path).__array__()).tolist()
    return set(labels)

def generate_meta(image_root, mask_root):
    mask_videos = listdir(mask_root, complete_path=False)
    image_videos = listdir(image_root, complete_path=False)

    videos = sorted(list(set(mask_videos) & set(image_videos)))
    data_infos = dict()
    # todo video target nums meta
    for v in videos:
        mask_frames = listfile(Path(mask_root) / v, '*.png', complete_path=False)
        image_frames = listfile(Path(image_root) / v, '*.jpg',complete_path=False)
        assert len(mask_frames) == len(image_frames), f'nums of JPG & mask not match {v}'

        H,W,C = Image.open(str(Path(mask_root) / image_frames[0])).__array__().shape

        labels_set = set()
        for mask in mask_frames:
            labels_set = labels_set | read_labels(Path(mask_root) / mask)
        frame_and_mask = list(zip(image_frames,mask_frames))
        data_infos[v] = {
            'img_height' : H,
            'img_width' : W,
            'nums_frame' : len(frame_and_mask),
            'labels' : labels_set,
            'nums_obj' : len(labels_set) - 1,
            'frame_and_mask':frame_and_mask
        }
    return data_infos