import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math
from util.tensor_util import compute_tensor_iu
from model.modules import ValueEncoder,ValueEncoderSO,KeyEncoder,KeyProjection,Decoder

class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p
    
class LossComputer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BootstrappedCE()

    def forward(self, output, it=0):
        losses = defaultdict(int)
        pred_logits = output['pred_logits']
        cls_gt = output['cls_gt']
        obj_map = output['obj_map']
        for i,per_frame in enumerate(obj_map):
            losses['total_loss'] = losses['total_loss'] + self.bce(pred_logits[per_frame].unsqueeze(0),
                                    cls_gt[i].unsqueeze(0),it)[0]
        
        return losses

class STCNModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.key_encoder = KeyEncoder()
        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.value_encoder = ValueEncoderSO()
        # Compress f16 a bit to use in decoding later on
        self.mask_decoder = Decoder()
        self.loss_fn = LossComputer()
    
    def encode_key(self, frame): 
        # input: b*c*h*w
        b = frame.shape[0]

        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*H*W
        k16 = k16.view(b, *k16.shape[-3:]).contiguous()

        # B*C*H*W
        f16_thin = f16_thin.view(b, *f16_thin.shape[-3:])
        f16 = f16.view(b, *f16.shape[-3:])
        f8 = f8.view(b, *f8.shape[-3:])
        f4 = f4.view(b, *f4.shape[-3:])

        return k16, f16_thin, f16, f8, f4

    def encode_value(self,image,kf16,obj_masks):
        ii = self.broadcast_map # broadcast B,C,H,W to N,C,H,W ,from B frames to N objects
        obj_mem_values = self.value_encoder(image[ii],kf16[ii],obj_masks.unsqueeze(1))
        return obj_mem_values

    def decode_mask(self,obj_read_values,kf16_thin, kf8, kf4):
        ii = self.broadcast_map # broadcast B,C,H,W to N,C,H,W ,from B frames to N objects
        obj_mask_value = torch.cat([obj_read_values, kf16_thin[ii]],dim=1)
        logits = self.mask_decoder(obj_mask_value,kf8[ii],kf4[ii]).squeeze(1)
        prob_ = torch.sigmoid(logits).clamp(1e-7, 1-1e-7)
        logits = torch.log(prob_ / (1 - prob_))
        prob = torch.empty_like(prob_)
        for obj_per_frame in self.aggregate_map:
            prob[obj_per_frame] = F.log_softmax(logits[obj_per_frame],dim=0)

        return prob,logits

    def init_memory(self,cls_gt):
        B,H,W = cls_gt.shape
        obj_labels = [ 
                [i, sorted(f.unique().tolist())] 
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
        k = 0
        for i,labels in obj_labels:
            obj = []
            for j in labels:
                obj.append(k)
                k += 1
            aggregate_map.append(obj)

        
        self.B = B
        self.N = N
        self.H = H
        self.W = W
        self.broadcast_map = broadcast_map
        self.aggregate_map = aggregate_map
        self.obj_labels = obj_labels
        self._memory_has_init = False # see self.add_memory
        
        return obj_masks # N,H,W

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
        frame_affinity = x_exp / x_exp_sum  # B,THW,HW
        del maxes,x_exp,x_exp_sum,ab,a_sq,mk,qk # save memory

        obj_affinity = frame_affinity[self.broadcast_map] # N,THW,HW
        del frame_affinity
        N, CV, T, H, W = mv.shape

        mo = mv.view(N, CV, T*H*W) 
        obj_read_values = torch.bmm(mo, obj_affinity) # N, CV, HW
        obj_read_values = obj_read_values.view(N, CV, H, W)
        return obj_read_values

    def forward(self,current_frame=dict(),return_loss=True,**data):
        if len(current_frame) == 0 and len(data) != 0 and return_loss == False:
            return self.train_step(data,None,return_loss=False)

        image = current_frame['rgb'] 
        frame_key, kf16_thin, kf16, kf8, kf4  = self.encode_key(image)
        # 除非特殊说明，所有Tensor的维度都是 B,C,H,W
        if current_frame['is_ref']:
            cls_gt = current_frame['cls_gt'] # B,H,W
            obj_logits = None
            obj_masks = self.init_memory(cls_gt)
        else:
            obj_read_values = self.read_memory(frame_key) 
            obj_masks,obj_logits = self.decode_mask(obj_read_values, kf16_thin, kf8, kf4)
        obj_mem_values = self.encode_value(image,kf16,obj_masks)
        self.add_memory(frame_key,obj_mem_values,obj_masks)
        return {
            'pred_mask':obj_masks,
            'pred_logits':obj_logits,
            'gt_mask':self.split_masks(current_frame['cls_gt']),
            'obj_map':self.aggregate_map,
            **current_frame
        }

    def train_step(self,data,optimizer,return_loss=True):
        B,T,C,H,W = data['rgb'].shape
        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)
        loss = 0
        total_i = 0
        total_u = 1
        for i in range(T):
            currents_frame = {
                'rgb':data['rgb'][:,i],  # B,3,H,W
                'cls_gt':data['cls_gt'][:,i], # B,H,W \in 0,1,2...
                'name':data['info']['name'],
                'frames': [x[i] for x in data['info']['frames']],
                'is_ref': True if i == 0 else False
            }
            output = self.forward(currents_frame)
            if output['is_ref']:
                continue
            ti,tu = compute_tensor_iu(output['pred_logits'] > 0.5, output['gt_mask'] > 0.5)
            total_i += ti.detach().cpu()
            total_u += tu.detach().cpu()
            loss = loss  +  self.loss_fn(output)['total_loss']

        if return_loss == False:
            return [(total_i / total_u).cpu().numpy()]

        return {
            'loss':loss,
            'log_vars':{
                'loss':loss.detach().cpu(),
                'iou': total_i / total_u
            },
            'num_samples' : B
        }
            
    def val_step(self,data,optimizer):
        pass
    
    def split_masks(self, cls_gt):
        obj_labels = self.obj_labels
        obj_masks = torch.stack([
                cls_gt[i] == label 
                for i,labels in obj_labels
                for label in labels
        ])
        return obj_masks