import math
import numpy as np
import cv2
from skimage.morphology import disk
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
import json
from tqdm import tqdm
import pickle
import pandas as pd

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / (union+1)
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

def metric_frame_JF(pred_frame, gt_frame):

    J = [
            db_eval_iou(pred_object,gt_object) # object H,W 
            for pred_object,gt_object in zip(pred_frame,gt_frame) # frame N,H,W
        ]
    F = [
            db_eval_boundary(pred_object,gt_object)
            for pred_object,gt_object in zip(pred_frame,gt_frame)
        ]
    return {
        'J' : np.array(J), # N
        'F' :np.array(F)
    }
def metric_video_JF(video_pred_masks,video_gt_masks):
    '''
    video_gt_masks (np.array) T,N,H,W bool
    '''
    assert video_pred_masks.shape == video_gt_masks.shape
    k = video_gt_masks.shape[1]
    J = [
        [
            db_eval_iou(pred_object,gt_object) # object H,W 
            for pred_object,gt_object in zip(pred_frame,gt_frame) # frame N,H,W
        ]
        for pred_frame,gt_frame in zip(video_pred_masks,video_gt_masks) # video T,N,H,W
    ]
    F = [
        [
            db_eval_boundary(pred_object,gt_object)
            for pred_object,gt_object in zip(pred_frame,gt_frame)
        ]
        for pred_frame,gt_frame in zip(video_pred_masks,video_gt_masks)
    ]
    return {
        'J' : np.array(J), # N,T
        'F' :np.array(F)
    }

def split_object_masks(pred_video_cls, gt_video_cls, label):
    pred_video_masks = np.stack([
        pred_video_cls == i
        for i in label
    ])
    gt_video_masks = np.stack([
        gt_video_cls == i
        for i in label
    ])
    return pred_video_masks, gt_video_masks

def read_video_pair(pred_video_path, gt_video_path):
    filenames = sorted([f.name for f in Path(pred_video_path).glob('*.png')])
    pred_video_cls = np.stack([
        np.array(Image.open(Path(pred_video_path) / img))
        for img in filenames
    ])
    
    gt_video_cls = np.stack([
        np.array(Image.open(Path(gt_video_path) / img))
        for img in filenames
    ])
    object_labels = np.unique(gt_video_cls)[1:]
    return split_object_masks(pred_video_cls, gt_video_cls, object_labels)


def metric_videos(pred_path, gt_path):
    pred_videos = [v for v in Path(pred_path).iterdir()]
    videos_name = [v.name for v in pred_videos]
    gt_videos = [v for v in Path(gt_path).iterdir() if v.name in videos_name]
    results_J = dict()
    results_F = dict()
    i = 0
    for name,pred_video,gt_video in tqdm(zip(videos_name,pred_videos,gt_videos)):
        pred_masks, gt_masks = read_video_pair(pred_video, gt_video)
        JF = metric_video_JF(pred_masks, gt_masks)
        results_J[name] = JF['J']
        results_F[name] = JF['F']

    return results_J, results_F

def video_average(results):
    res = []
    for k,v in results.items():
        res.append((k,v.mean()))
    return sorted(res,key= lambda x:x[1])

def object_average(results):
    res = []
    for k,v in results.items():
        for i,o in enumerate(v):
            res.append((k,i,o.mean()))
    return sorted(res,key= lambda x:x[2])

def frame_average(results):
    res = []
    for k,v in results.items():
        N,T = v.shape
        for j in range(T):
            res.append((k,j,v[:,j].mean()))
    return sorted(res, key = lambda x:x[2])

def main():
    parser = ArgumentParser()
    parser.add_argument('pred_videos_root')
    parser.add_argument('gt_videos_root')
    parser.add_argument('--save', default='.')
    args = vars(parser.parse_args())
    J, F = metric_videos(args['pred_videos_root'], args['gt_videos_root'])
    with open(Path(args['save']) / 'raw_result.pkl', 'wb') as fp:
        pickle.dump({'J':J,'F':F},fp)
    
    JF = dict()
    for k in J:
        JF[k] = (J[k] + F[k]) / 2 

    result = dict()
    result['bad_video'] = pd.DataFrame.from_records(video_average(JF), columns=['video_name','J&F'])
    result['bad_obj'] = pd.DataFrame.from_records(object_average(JF), columns=['video_name','obj_id','J&F'])
    result['bad_frame'] = pd.DataFrame.from_records(frame_average(JF), columns=['video_name','frame_id','J&F'])
    for k,v in result.items():
        print(k)
        print(v[:10])
        v.to_csv(str(Path(args['save']) / f'{k}.csv'))
        print('---------------')
        
if __name__ == '__main__':
    main()
