from mmcv.runner import BaseModule
from mmcv.utils import Registry
from collections import defaultdict
from mmdet.models import BACKBONES, LOSSES
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

@VOSMODEL.register_module()
class STCN(BaseModule):
    def __init__(self, 
                key_encoder,
                value_encoder,
                mask_decoder,
                memory,
                loss_fn,
                init_cfg=None):
        super().__init__(init_cfg)
        self.key_encoder = VOSMODEL.build(key_encoder)
        self.value_encoder = VOSMODEL.build(value_encoder)
        self.mask_decoder = VOSMODEL.build(mask_decoder)
        self.memory = VOSMODEL.build(memory)
        self.loss_fn = LOSSES.build(loss_fn)
        self.targets = []
    
    def update_targets(self, new_objs):
        self.targets.extend(new_objs)
        
        # [0,0,1,1,2,2,3,3,1] -> 9 objs , 4 image
        fi_list=[i for i,l in self.targets]

        # [[0,1],[2,3,8],[4,5],[6,7]] -> 9 objs, 4 image
        oi_groups = [[] for _ in range(len(set(fi_list)))]
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
            self.update_targets(new_objs)
        else:
            new_gt_mask = None
        return old_gt_mask, new_gt_mask

    def forward(self,img, gt_mask, flag, return_loss=False,*k,**kw):
        K, feats = self.key_encoder(img) 
        if flag == 'new_video':
            self.memory.reset()
            self.targets = []
            self.oi_groups = []
            pred_mask = None
            pred_logits = None
        else:
            # 先预测
            V = self.memory.read(K)
            pred_logits, pred_mask = self.mask_decoder(V, feats, self.fi_list)

        # 然后增加新 GT

        oi_groups = self.oi_groups.copy()
        old_gt_mask, new_gt_mask = self.parse_targets(gt_mask)
        mask = safe_torch_cat([pred_mask, new_gt_mask],dim=0)
        V = self.value_encoder(mask, feats, self.fi_list)
        self.memory.write(K, V)

        if return_loss:
            loss = 0
            for i,oii in enumerate(oi_groups):
                cls_gt = torch.topk(
                    old_gt_mask[oii].char(), 
                    k = 1, 
                    dim=0)[1].squeeze(0)
                logits = pred_logits[oii].swapaxes(0,1)
                loss += self.loss_fn(logits, cls_gt)
            output = {'loss': loss}
        else:
            output = {'mask': mask}

        return output

    def train_step(self, data_batch, optimizer, **kw):
        output = defaultdict(list)
        for i,data in enumerate(data_batch):
            img = data['img'].data
            gt_mask = data['gt_mask'].data
            flag = 'new_video' if i == 0 else ''
            output[i] = self.forward(
                    img = img,
                    gt_mask = gt_mask,
                    flag=flag,
                    return_loss=True,
                    )

        loss = sum([item['loss'] for key,item in output.items()])
        return {
            'loss' : loss,
            'num_samples': len(data_batch) * img.shape[0],
            'log_vars' : {
                'loss' : loss.detach().cpu(),
            }
        }

    
    def val_step(self,data_batch,**kw):
        # val_step 不会记录梯度
        return self.forward(**data_batch,return_loss=False)

        