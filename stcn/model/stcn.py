from mmcv.runner import BaseModule
from mmcv.utils import Registry
from collections import defaultdict
from mmdet.models import BACKBONES, LOSSES
import torch

VOSMODEL = Registry('vos_model')

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
    
    def update_targets(self, gt_mask):
        # 出现了新的目标: 一般只在第一帧，Youtube 后续帧也会出现新目标
        this_objs = [
            (img_id,obj_label) 
            for img_id,frame_gt in enumerate(gt_mask)
            for obj_label in frame_gt.unique().tolist() 
        ]
        new_objs = sorted(list(set(this_objs) - set(self.targets)))
        self.targets.extend(new_objs)
        
        # [0,0,1,1,2,2,3,3,1] -> 9 objs , 4 image
        broadcast_map=[i for i,l in self.targets]

        # [[0,1],[2,3,8],[4,5],[6,7]] -> 9 objs, 4 image
        aggregate_map = [[] for _ in range(len(set(broadcast_map)))]
        for oi,fi in enumerate(broadcast_map):
            aggregate_map[fi].append(oi)
    
        self.memory.update_targets(broadcast_map)
        self.key_encoder.update_targets(broadcast_map)
        self.mask_decoder.update_targets(aggregate_map)
        self.aggregate_map = aggregate_map


    def parse_targets(self, gt_mask):
        prob = torch.stack([
            gt_mask[i] == label 
            for i,label in self.targets
        ])
        return prob

    def forward(self,img, gt_mask, flag, return_loss=False,*k,**kw):
        if flag == 'new_video':
            # self.memory.reset()
            self.memory.reset()
            self.update_targets(gt_mask)
            K, feats = self.key_encoder(img) 
            mask_prob = self.parse_targets(gt_mask)
        else:
            K, feats = self.key_encoder(img) 
            V = self.memory.read(K)
            logits, mask_prob = self.mask_decoder(V, feats)

        V = self.value_encoder(mask_prob, feats)
        self.memory.write(K, V)

        if return_loss and flag == 'new_video':
            output = {'loss': 0}
        elif return_loss and flag != 'new_video':
            loss = 0
            for i,objs_per_frame in enumerate(self.aggregate_map):
                cls_gt = gt_mask[i].long()
                logits_one_frame = logits[objs_per_frame].swapaxes(0,1)
                loss += self.loss_fn(logits_one_frame, cls_gt)
            output = {'loss': loss}
        else:
            output = {'mask': mask_prob}

        return output

    def train_step(self, data_batch, optimizer, **kw):
        output = defaultdict(list)
        for i,data in enumerate(data_batch):
            img = data['img'].data[0]
            gt_mask = data['gt_mask'].data[0]
            flag = 'new_video' if i == 0 else ''
            output[i] = self.forward(
                    img = img,
                    gt_mask = gt_mask,
                    flag=flag,
                    return_loss=True,
                    )

        loss = sum([item['loss'] for key,item in output.items()])
        return {
            'loss' : loss
        }

    
    def val_step(self,data_batch,**kw):
        # val_step 不会记录梯度
        return self.forward(**data_batch,return_loss=False)

        