from mmcv.runner import BaseModule
from mmcv.utils import Registry
from collections import defaultdict
from mmdet.models import BACKBONES, LOSSES
from torch.nn import Parameter
import numpy as np
import torch

VOSMODEL = Registry('vos_model')

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
                init_cfg=None):
        super().__init__(init_cfg)
        self.key_encoder = VOSMODEL.build(key_encoder)
        self.value_encoder = VOSMODEL.build(value_encoder)
        self.mask_decoder = VOSMODEL.build(mask_decoder)
        self.memory = VOSMODEL.build(memory)
        self.loss_fn = LOSSES.build(loss_fn)
        self.use_bg = seg_background
        self.sentry = Parameter(torch.Tensor(0))
        self.targets = []
    
    def train(self, mode=True):
        self.memory.train(False)
        super().train(mode)

    def eval(self):
        self.train(False)

    def forward(self,img, gt_mask=None, img_metas=None, return_loss=False,*k,**kw):
        pred_mask = None
        new_gt_mask = None
        #! encode key
        K, feats = self.key_encoder(img) 

        if img_metas[0]['flag'] == 'new_video':
            self.memory.reset()
            self.targets = []
            self.oi_groups = []
        
        if self.memory.is_init:
            #! read memory
            V = self.memory.read(K)
            #! deocde mask
            pred_logits, pred_mask = self.mask_decoder(V, feats, self.fi_list)

        if gt_mask is not None:
            oi_groups = self.oi_groups.copy()
            old_gt_mask, new_gt_mask = self.parse_targets(gt_mask)

        mask = safe_torch_cat([pred_mask, new_gt_mask],dim=0)
        if mask is not None:
            #! encode mask
            V = self.value_encoder(mask, feats, self.fi_list)
            #! write memory
            self.memory.write(K, V)

        if return_loss:
            loss = torch.sum(self.sentry * 0)
            for oii in oi_groups:
                if len(oii) == 0:
                    continue

                cls_gt = self.compute_label(old_gt_mask[oii])

                logits = pred_logits[oii]
                if not self.use_bg:
                    bg_logits = compute_bg_logits(logits)
                    logits = torch.cat([bg_logits, logits], dim=0)

                #! compute loss
                loss += self.loss_fn(logits.swapaxes(0,1), cls_gt)
            output = {
                'loss': loss,
                'nums_frame': len(oi_groups),
                }
        else:
            out_masks = []
            for oii in self.oi_groups:
                if len(oii) == 0:
                    continue

                #! pred mask
                out_mask = self.compute_label(mask[oii])
                out_mask = out_mask.cpu().numpy().astype(np.uint8).squeeze(0)
                out_masks.append(out_mask)
            if len(out_masks) == 0:
                out_mask = np.zeros((img_metas[0]['img_shape'][:2])).astype(np.uint8)
                out_masks.append(out_mask)
                
            output = {'mask': out_masks, 'img_metas': img_metas}

        return output

    def train_step(self, data_series, optimizer, batch_size = None,**kw):
        output = defaultdict(list)
        step = batch_size
        # todo 当 labels 个数超过上限后， 随机抑制 labels
        for i in range(0,len(data_series['img']),step):
            img = data_series['img'][i:i+step]
            gt_mask = data_series['gt_mask'][i:i+step]
            img_metas = data_series['img_metas'][i:i+step]
            flag = 'new_video' if i == 0 else ''
            img_metas[0]['flag'] = flag
            output[i] = self.forward(
                    img = img,
                    gt_mask = gt_mask,
                    img_metas=img_metas,
                    return_loss=True,
                    )

        loss = sum([item['loss'] for key,item in output.items()])
        nums_frame = sum([item['nums_frame'] for key,item in output.items()])
        return {
            'loss' : loss,
            'num_samples': nums_frame,
            'log_vars' : {
                'loss' : loss.detach().cpu(),
                # 'mem_num_objs' : self.memory.gate.shape[0],
            }
        }

    def update_targets(self, new_objs, batch_size):
        self.targets.extend(new_objs)
        
        # [0,0,1,1,2,2,3,3,1] -> 9 objs , 4 image
        fi_list=[i for i,l in self.targets]

        # [[0,1],[2,3,8],[4,5],[6,7]] -> 9 objs, 4 image
        oi_groups = [[] for _ in range(batch_size)]
        for oi,fi in enumerate(fi_list):
            oi_groups[fi].append(oi)
    
        self.memory.update_targets(fi_list)
        self.fi_list = fi_list
        self.oi_groups = oi_groups


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
            ])
        else:
            old_gt_mask = None
        if len(new_objs) > 0:
            new_gt_mask = torch.stack([
                gt_mask[i] == label 
                for i,label in new_objs
            ])
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