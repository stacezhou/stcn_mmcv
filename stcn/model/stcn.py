from mmcv.runner import BaseModule
from mmcv.utils import Registry
from collections import defaultdict
from mmdet.models import BACKBONES, LOSSES
from torch.nn import Parameter
import numpy as np
import torch
import random
from stcn.dataset.metric import db_eval_iou
import torch.nn.functional as F

VOSMODEL = Registry('vos_model')
from pathlib import Path

def safe_torch_cat(TensorList, *k, **kw):
    valid_tensor_list = []
    for T in TensorList:
        if T is not None:
            valid_tensor_list.append(T)
    if len(valid_tensor_list) != 0:
        return torch.cat(valid_tensor_list, *k, **kw)
    else:
        return None

def compute_bg_prob(prob_wo_bg):
    return torch.prod(1 - prob_wo_bg, dim=0, keepdim=True)

def compute_bg_logits(logits_wo_bg):
    prob_wo_bg = torch.sigmoid(logits_wo_bg)
    prob_bg = compute_bg_prob(prob_wo_bg).clamp(1e-7, 1-1e-7)
    logits_bg = torch.log(prob_bg / (1 - prob_bg))
    return logits_bg

@VOSMODEL.register_module()
class STCN(BaseModule):
    def __init__(self, 
                key_encoder,
                value_encoder,
                mask_decoder,
                memory,
                loss_fn,
                seg_background = False,
                max_objs_per_frame = 3,
                multi_scale_test = False,
                multi_scale_train = False,
                test_scales = [1, 1.3, 1.5, 2],
                train_scales = [1],
                align_corners = True,
                init_cfg=None):
        super().__init__(init_cfg)
        self.key_encoder = VOSMODEL.build(key_encoder)
        self.value_encoder = VOSMODEL.build(value_encoder)
        self.mask_decoder = VOSMODEL.build(mask_decoder)
        self.loss_fn = LOSSES.build(loss_fn)
        self.use_bg = seg_background
        self.sentry = Parameter(torch.Tensor(0))
        self.targets = []
        self.oi_groups = []
        self.max_objs_per_frame = max_objs_per_frame
        self.memory_module = memory
        self.test_sclaes = test_scales if multi_scale_test else [1]
        self.train_scales = train_scales if multi_scale_train else [1]
        self.align_corners = align_corners
    

    def update_targets(self, new_objs, batch_size):
        self.targets.extend(new_objs)
        
        # [0,0,1,1,2,2,3,3,1] -> 9 objs , 4 image
        fi_list=[i for i,l in self.targets]

        # [[0,1],[2,3,8],[4,5],[6,7]] -> 9 objs, 4 image
        oi_groups = [[] for _ in range(batch_size)]
        for oi,fi in enumerate(fi_list):
            oi_groups[fi].append(oi)
    
        for i,s in enumerate(self.multi_scales):
            self.memory[i].update_targets(fi_list)
        self.fi_list = fi_list
        self.oi_groups = oi_groups

    def forward(self,img=None, gt_mask=None, img_metas=None, return_loss=False,*k,**kw):

        pred_mask = [None for s in self.multi_scales]
        old_gt_mask = [None for s in self.multi_scales]
        new_gt_mask = [None for s in self.multi_scales]
        pred_logits = [None for s in self.multi_scales]
        mask = [None for s in self.multi_scales]
        V = [None for s in self.multi_scales]
        K = [None for s in self.multi_scales]
        feats = [None for s in self.multi_scales]
        new_size = [None for s in self.multi_scales]

        #! encode key
        for i,s in enumerate(self.multi_scales):
            *_,H,W = img.shape
            new_size[i] = int(H*s) // 16 * 16, int(W*s) // 16 * 16
            _img = F.interpolate(img, size=new_size[i], mode='bilinear', align_corners=self.align_corners) if s != 1 else img
            K[i], feats[i] = self.key_encoder(_img) #!
                

        if img_metas[0]['flag'] == 'new_video':
            for memory in self.memory:
                memory.reset()
            # List of (frame_index, object_label)
            self.targets = [] 
            # List of (object_index,...) of a frame
            # [[0,1],[2,3,8],[4,5],[6,7]] -> 9 objs, 4 image
            self.oi_groups = []
        
        #! read memory and decode mask
        if self.memory[0].is_init:
            for i,s in enumerate(self.multi_scales):
                V[i] = self.memory[i].read(K[i]) #!
                pred_logits[i], pred_mask[i] = self.mask_decoder(V[i], feats[i], self.fi_list) #!
                # fi_list : [0,0,1,1,2,2,3,3,1] -> 9 objs , 4 image


        #! parse new & gt object mask
        if gt_mask is not None:
            oi_groups = self.oi_groups.copy()
            _old_gt_mask, _new_gt_mask = self.parse_targets(gt_mask)
            for i,s in enumerate(self.multi_scales):
                old_gt_mask[i] = F.interpolate(
                    _old_gt_mask,
                    size=new_size[i],
                    mode='nearest'
                ) if s != 1 and _old_gt_mask is not None else _old_gt_mask
                new_gt_mask[i] = F.interpolate(
                    _new_gt_mask,
                    size=new_size[i],
                    mode='nearest'
                ) if s != 1 and _new_gt_mask is not None else _new_gt_mask


        #! concat predict and new_gt_mask
        for i,s in enumerate(self.multi_scales):
            mask[i] = safe_torch_cat([pred_mask[i],new_gt_mask[i]])

        if mask[0] is not None:
            ms_sizes = [m.shape[-2:] for m in mask]
            fuse_mask = torch.stack([
                    F.interpolate(
                        m,
                        size = ms_sizes[0],
                        mode='nearest'
                    ) if m.shape[-2:] != ms_sizes[0] else m
                for m in mask
                ]).float().mean(dim=0)
            #! multi scale interact
            mask = [
                    F.interpolate(
                        fuse_mask,
                        size = ms_size,
                        mode='nearest'
                    ) if fuse_mask.shape[-2:] != ms_size else fuse_mask
                for ms_size in ms_sizes
                ]

            #! encode object mask and write memory
            for i,s in enumerate(self.multi_scales):
                V[i] = self.value_encoder(mask[i], feats[i],self.fi_list)
                self.memory[i].write(K[i],V[i])



        if return_loss:
            loss = [torch.sum(self.sentry * 0)]
            for _old_gt_mask, _pred_logits in zip(old_gt_mask, pred_logits):
                _loss = torch.sum(self.sentry * 0) 
                iou = 0
                for oii in oi_groups:
                    if len(oii) == 0:
                        continue

                    cls_gt = self.compute_label(_old_gt_mask[oii])
                    logits = _pred_logits[oii]
                    if not self.use_bg:
                        bg_logits = compute_bg_logits(logits)
                        logits = torch.cat([bg_logits, logits], dim=0)

                    #! compute loss
                    _loss += self.loss_fn(logits.swapaxes(0,1), cls_gt, it=kw['runner'].iter)
                    loss.append(_loss)
            loss = sum(loss)
            output = {
                'loss': loss,
                'nums_frame': len(oi_groups),
                'bce_p' : self.loss_fn.this_p,
                }
        else:
            out_masks = []
            for oii in self.oi_groups:
                if len(oii) == 0:
                    continue

                #! pred mask
                out_mask = self.compute_label(fuse_mask[oii])
                out_mask = out_mask.cpu().numpy().astype(np.uint8).squeeze(0)
                out_masks.append(out_mask)

            if len(out_masks) == 0:
                out_mask = np.zeros((img_metas[0]['pad_shape'][:2])).astype(np.uint8)
                out_masks.append(out_mask)
                
            output = out_masks

        return output

    def train_step(self, data_series, optimizer, batch_size = None,*k,**kw):
        output = defaultdict(list)
        B = batch_size
        img_TB = data_series['img']
        BT,C,H,W = img_TB.shape
        T = BT // B
        img_TB = img_TB.view((T,B,C,H,W))
        gt_mask_TB = data_series['gt_mask'].view((T,B,1,H,W))
        img_metas_TB =[data_series['img_metas'][t*B:(t+1)*B] for t in range(T)]
        gt_mask_TB = self.random_filter(gt_mask_TB)
        # todo 当 labels 个数超过上限后， 随机抑制 labels
        for i in range(T):
            img_B = img_TB[i]
            gt_mask_B = gt_mask_TB[i]
            img_metas_B = img_metas_TB[i]
            flag = 'new_video' if i == 0 else ''
            img_metas_B[0]['flag'] = flag

            output[i] = self.forward(
                    img = img_B,
                    gt_mask = gt_mask_B,
                    img_metas=img_metas_B,
                    return_loss=True,
                    *k,**kw
                    )

        loss = sum([item['loss'] for key,item in output.items()])
        nums_frame = sum([item['nums_frame'] for key,item in output.items()])

        if 'runner' in kw:
            runner = kw['runner']
            # inject from config
            if 'inject' in runner.meta and runner.iter in runner.meta['inject']:
                command = runner.meta['inject'][runner.iter]
                try:
                    exec(command,
                    {'data':runner.data_loader.dataset,
                    'model':runner.model.module
                    })
                except:
                    pass
            # inject from file
            injection_command = Path(runner.work_dir) / f'inject@{runner.iter}'
            if injection_command.exists():
                with open(injection_command, 'r') as fp:
                    command = fp.read()
                try:
                    exec(command,
                    {'data':runner.data_loader.dataset,
                    'model':runner.model.module
                    })
                except:
                    pass
        return {
            'loss' : loss,
            'num_samples': nums_frame,
            'log_vars' : {
                'bce_p' : output[0]['bce_p'],
                'loss' : loss.detach().cpu(),
                # 'mem_num_objs' : self.memory.gate.shape[0],
            }
        }



    def parse_targets(self, gt_mask):
        # 出现了新的目标: 一般只在第一帧，Youtube 后续帧也会出现新目标
        this_objs = [
            (fi,label) 
            for fi,frame_gt in enumerate(gt_mask)
            for label in frame_gt.unique().tolist() 
            if label != 0 or self.use_bg
        ]
        new_objs = sorted(list(set(this_objs) - set(self.targets)))

        if len(self.targets) > 0:
            old_gt_mask = torch.stack([
                gt_mask[i] == label 
                for i,label in self.targets
            ]).float()
        else:
            old_gt_mask = None
        if len(new_objs) > 0:
            new_gt_mask = torch.stack([
                gt_mask[i] == label 
                for i,label in new_objs
            ]).float()
            self.update_targets(new_objs, gt_mask.shape[0])
        else:
            new_gt_mask = None
        return old_gt_mask, new_gt_mask
    
    def compute_label(self, mask_prob):
        p_mask = mask_prob.float()
        if not self.use_bg:
            bg_mask = compute_bg_prob(p_mask)
            p_mask = torch.cat([bg_mask, p_mask], dim=0)
        out_mask_label = torch.argmax(p_mask, dim=0)
        return out_mask_label


    def train(self, mode=True):
        if mode == True:
            self.multi_scales = self.train_scales
            self.memory = [VOSMODEL.build(self.memory_module) for s in self.multi_scales]
        else:
            self.multi_scales = self.test_sclaes
            self.memory = [VOSMODEL.build(self.memory_module) for s in self.multi_scales]

        for memory in self.memory:
            memory.train(mode)
        super().train(mode)

    def eval(self):
        self.train(False)
    
    def random_filter(self,gt_mask_TB):
        for i in range(gt_mask_TB.shape[1]):
            object_labels  = gt_mask_TB[:,i].unique().tolist()
            if 0 in object_labels:
                object_labels.remove(0)
            while len(object_labels) > self.max_objs_per_frame:
                random.shuffle(object_labels)
                l = object_labels.pop()
                gt_mask_TB[:,i] = torch.where(gt_mask_TB[:,i] == l,torch.zeros_like(gt_mask_TB[:,i]),gt_mask_TB[:,i])
        return gt_mask_TB
