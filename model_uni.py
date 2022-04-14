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
        self.this_p = 0

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
        self.this_p = this_p
        return loss.mean()
    
class LossComputer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BootstrappedCE()

    def forward(self, pred_logits,cls_gt,obj_map, it=0):
        loss = 0
        for i,per_frame in enumerate(obj_map):
            c_gt = cls_gt[i].unsqueeze(0)
            logits = pred_logits[per_frame].unsqueeze(0)
            loss += self.bce(logits,c_gt,it)[0]

        return loss
    
    @property
    def p(self):
        return self.bce.this_p

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
        prob_ = torch.sigmoid(logits)
        new_prob = torch.zeros_like(prob_)
        for obj_per_frame in self.aggregate_map:
            bg,*objs = obj_per_frame
            new_prob[objs] = prob_[objs]
            new_prob[bg] = torch.prod(1-new_prob[objs],dim=0,keepdim=False).clamp(1e-7, 1-1e-7)


        logits = torch.log(new_prob / (1 - new_prob))
        prob = torch.empty_like(new_prob)
        for obj_per_frame in self.aggregate_map:
            prob[obj_per_frame] = F.log_softmax(logits[obj_per_frame],dim=0)

        return prob,logits

    def init_memory(self,cls_gt,obj_labels):
        B,H,W = cls_gt.shape
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

    def forward(self,return_loss=False,**data_batch):
        assert 'rgb' in data_batch
        assert 'cls_gt' in data_batch
        assert len(data_batch['rgb'].shape) == 5, "rgb must be a (B,T,3,H,W) Tensor"
        assert len(data_batch['cls_gt'].shape) == 4, "cls_gt must be a (B,t,H,W) int Tensor"

        # data to cuda
        for k, v in data_batch.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data_batch[k] = v.cuda(non_blocking=True)

        if return_loss:
            loss = 0
            total_i = 0
            total_u = 1
        else:
            batch_masks = []

        B,T,C,H,W = data_batch['rgb'].shape
        obj_labels_w_bg = [[idx,x.unique().tolist()] for idx,x in enumerate(data_batch['cls_gt'])]

        for i in range(T):
            image = data_batch['rgb'][:,i] # B,3,H,W ,默认所有Tensor都是 B,C,H,W
            frame_key, kf16_thin, kf16, kf8, kf4  = self.encode_key(image)
            if i == 0: # todo, always first frame as ref frame 
                cls_gt = data_batch['cls_gt'][:,i] # B,H,W
                obj_masks = self.init_memory(cls_gt, obj_labels_w_bg)
            else:
                obj_read_values = self.read_memory(frame_key) 
                obj_masks,obj_logits = self.decode_mask(obj_read_values, kf16_thin, kf8, kf4)

            obj_mem_values = self.encode_value(image,kf16,obj_masks)
            self.add_memory(frame_key,obj_mem_values,obj_masks)

            if return_loss:
                if i == 0 : 
                    continue
                cls_gt = data_batch['cls_gt'][:,i] # B,H,W \dtype int
                gt_mask = self.split_masks(cls_gt)
                ti,tu = compute_tensor_iu(obj_logits > 0.5, gt_mask > 0.5)
                total_i += ti.detach().cpu()
                total_u += tu.detach().cpu()
                if 'iter' in data_batch:
                    it = data_batch['it']
                else:
                    it = 0
                loss = loss  +  self.loss_fn(obj_masks,cls_gt,self.aggregate_map,it)
            else:
                # todo
                batch_masks.append(obj_masks.detach().cpu())

        
        if return_loss:
            return { 'loss':loss,
                'p' : self.loss_fn.p,
                'num_samples':B,
                'iou' : total_i / total_u }
        else:
            return {
                'pred_masks': batch_masks,
                ** data_batch
            }


    def train_step(self,data_batch,optimizer,**kw):
        output = self.forward(return_loss=True,**data_batch)
        loss = output['loss']
        return {
            'loss':loss,
            'log_vars':{
                'loss':loss.detach().cpu(),
                'iou': output['iou'],
                'p': output['p']
            },
            'num_samples' : output['num_samples']
        }
            
    def val_step(self,data_batch,**kw):
        return self.train_step(data_batch)

    def split_masks(self, cls_gt):
        obj_labels = self.obj_labels
        obj_masks = torch.stack([
                cls_gt[i] == label 
                for i,labels in obj_labels
                for label in labels
        ])
        return obj_masks