import torch
import torch.nn.functional as F
import math
from model.network import STCN
from model.losses import LossComputer
from model.modules import ValueEncoder,ValueEncoderSO




class STCNModel(STCN):

    def __init__(self):
        super().__init__(single_object=True)
        self.value_encoder = ValueEncoderSO()
        # self.loss_fn = LossComputer()
        self.MAX_NUM_MEMORY = 5
    
    def init_obj(self,cls_gt):
        B,H,W = cls_gt.shape
        obj_labels = [ 
                [i, f.unique().tolist()] 
                for i,f in enumerate(cls_gt) 
        ]
        # for i,labels in obj_labels:
        #     labels.remove(0) # remove background

        obj_masks = torch.stack([
                cls_gt[i] == label 
                for i,labels in obj_labels
                for label in labels
        ])
        N = obj_masks.shape[0]
        broadcast_map = [
                i
                for i,labels in obj_labels
                for label in labels
        ]

        aggregate_map = []
        for i,labels in obj_labels:
            aggregate_map.append([i]*len(labels))
        
        self.B = B
        self.N = N
        self.H = H
        self.W = W
        self.broadcast_map = broadcast_map
        self.aggregate_map = aggregate_map
        self.obj_labels = obj_labels
        self._memory_has_init = False
        
        return obj_masks # N,H,W

    def split_masks(self, cls_gt):
        obj_labels = self.obj_labels
        obj_masks = torch.stack([
                cls_gt[i] == label 
                for i,labels in obj_labels
                for label in labels
        ])
        return obj_masks

    def add_memory(self,key,value,mask):
        if self._memory_has_init:
            self.mem_keys = torch.cat([self.mem_keys, key.unsqueeze(2)],dim=2)
            self.mem_values = torch.cat([self.mem_values, value.unsqueeze(2)],dim=2)
            self.mem_masks = torch.cat([self.mem_masks, mask.unsqueeze(0)],dim=0)
        else:
            self.mem_keys = key.unsqueeze(2) # B,C,T,H,W
            self.mem_values = value.unsqueeze(2) # N,C,T,H,W
            self.mem_masks = mask.unsqueeze(0) # N,H,W
            self._memory_has_init = True


    def read_memory(self,qk):
        # qk:B,C,H,W
        mk = self.mem_keys 
        mv = self.mem_values
        
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2) # B,C,THW
        qk = qk.flatten(start_dim=2) # B,C,HW

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2) # B,THW,1
        ab = mk.transpose(1, 2) @ qk # B,THW,HW

        frame_affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(frame_affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(frame_affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum  # B,THW,HW

        obj_affinity = frame_affinity[self.broadcast_map] # N,THW,HW
        N, CV, T, H, W = mv.shape

        mo = mv.view(N, CV, T*H*W) 
        obj_read_values = torch.bmm(mo, obj_affinity) # N, CV, HW
        obj_read_values = obj_read_values.view(N, CV, H, W)
        return obj_read_values

    def mask_decoder(self,obj_read_values,kf16_thin, kf8, kf4):
        ii = self.broadcast_map
        obj_mask_value = torch.cat([obj_read_values, kf16_thin[ii]],dim=1)
        logits = self.decoder(obj_mask_value,kf8[ii],kf4[ii])
        prob = torch.sigmoid(logits)
        return logits.squeeze(1),prob.squeeze(1)

    def forward(self,current_frame):
        image = current_frame['rgb'] 
        frame_key, kf16_thin, kf16, kf8, kf4  = self.encode_key(image)
        # 除非特殊说明，所有Tensor的维度都是 B,C,H,W
        if current_frame['is_ref']:
            cls_gt = current_frame['cls_gt'] # B,H,W
            obj_logits = None
            obj_masks = self.init_obj(cls_gt)
        else:
            obj_read_values = self.read_memory(frame_key) 
            obj_masks,obj_logits = self.mask_decoder(obj_read_values, kf16_thin, kf8, kf4)
        ii = self.broadcast_map # broadcast B,C,H,W to N,C,H,W ,from B frames to N objects
        obj_mem_values = self.value_encoder(image[ii],kf16[ii],obj_masks.unsqueeze(1))
        self.add_memory(frame_key,obj_mem_values,obj_masks)
        return {
            'pred_mask':obj_masks,
            'pred_logits':obj_logits,
            'gt_mask':self.split_masks(current_frame['cls_gt']),
            **current_frame
        }

    def train_step(self,data,optimizer):
        B,T,C,H,W = data['rgb'].shape
        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)
        loss = 0
        from tqdm import tqdm
        for i in tqdm(range(T)):
            currents_frame = {
                'rgb':data['rgb'][:,i],  # B,3,H,W
                'cls_gt':data['cls_gt'][:,i], # B,H,W \in 0,1,2...
                'name':data['info']['name'],
                'frames': [x[i] for x in data['info']['frames']],
                'is_ref': True if i == 0 else False
            }
            output = self.forward(currents_frame)
            # loss += self.loss_fn(output)

        return {
            'loss':loss
        }
            
    def val_step(self,data,optimizer):
        with torch.no_grad():
            output = self.forward(data)
        return output