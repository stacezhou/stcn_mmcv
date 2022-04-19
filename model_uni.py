import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from util.tensor_util import compute_tensor_iu
from model.modules import ValueEncoderSO,KeyEncoder,KeyProjection,Decoder

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

    def forward(self, pred_logits,cls_gt,a_map, it=0):
        loss = 0
        for i,per_frame in enumerate(a_map):
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
        self.mem_objs = [] # (No.frame, cls_label)
    
    def forward(self,return_loss=False,**data_batch):
        assert 'rgb' in data_batch # B,T,3,H,W
        assert 'cls_gt' in data_batch # B,T,H,W

        # 测试时没有 gt，没法计算 loss 
        if return_loss:
            loss = 0
            total_i = 0
            total_u = 1
        else:
            out_masks = []


        # 默认一个 batch 清空 memory
        if 'reset_memory' not in data_batch or data_batch['reset_memory']:
            self.mem_objs = []
            self.mem_keys = None


        # 按时间依次处理 Batch 内各帧
        B,T,C,H,W = data_batch['rgb'].shape
        for i in range(T):

            image = data_batch['rgb'][:,i] # B,3,H,W 
            cls_gt = data_batch['cls_gt'][:,i] # B,H,W


            # 出现了新的目标: 一般只在第一帧，Youtube 后续帧也会出现新目标
            this_objs = [
                (img_id,obj_label) 
                for img_id,frame_gt in enumerate(cls_gt)
                for obj_label in frame_gt.unique().tolist() 
            ]
            new_objs = sorted(list(set(this_objs) - set(self.mem_objs)))
            self.get_new_objs = len(new_objs) > 0
            if self.get_new_objs:
                # 从 cls_gt 得到每个 object 的 mask
                new_obj_masks = torch.stack([
                    cls_gt[i] == label
                    for i,label in new_objs
                ])
                _new_obj_prob = new_obj_masks.float().clamp(1e-7, 1-1e-7)
                # 以及相应的 logtis, 主要用于 Youtube 时和预测 object 统一
                new_obj_logits = torch.log(_new_obj_prob / (1 - _new_obj_prob))


            #######################! 预测目标  ############################
            frame_key, kf16_thin, kf16, kf8, kf4  = self.encode_key(image)
            if self.has_memory:
                obj_mem_out = self.read_memory(frame_key) 
                pred_obj_masks,pred_obj_logits = self.decode_mask(obj_mem_out, kf16_thin, kf8, kf4)


            # 组合新gt目标和预测目标
            if self.get_new_objs and self.has_memory:
                obj_masks = torch.cat([pred_obj_masks,new_obj_masks])
                obj_logits = torch.cat([pred_obj_logits,new_obj_logits])
            elif self.get_new_objs:
                obj_masks, obj_logits = new_obj_masks, new_obj_logits
            elif self.has_memory:
                obj_masks, obj_logits = pred_obj_masks,pred_obj_logits
            self.mem_objs.extend(new_objs)


            ########################! 存记忆   ############################
            obj_mem_in = self.encode_value(image,kf16,obj_masks)
            self.add_memory(frame_key,obj_mem_in)


            # obj_masks = F.interpolate(obj_masks, (H,W), mode='bilinear', align_corners=False)
            # cls_pred = torch.argmax(obj_masks, dim=0)
            # 计算 Loss 或者 直接返回预测结果
            if return_loss:
                if i == 0 : 
                    continue
                loss = loss  +  self.loss_fn(obj_logits,cls_gt,self.aggregate_map,data_batch['_iter'])

                gt_masks = torch.stack([
                    cls_gt[i] == label
                    for i,label in self.mem_objs
                ])
                # 不计算背景的 iou
                # non_bg = [i for i,label in self.mem_objs if label != 0]
                # test_masks = torch.empty_like(obj_masks)
                # for obj_per_frame in self.aggregate_map:
                #     pass
                # ti,tu = compute_tensor_iu(test_masks,  gt_masks[non_bg])
                ti,tu = compute_tensor_iu(obj_masks > 0.4,  gt_masks)
                total_i += ti.tolist()
                total_u += tu.tolist()
            else:
                # todo
                out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)
                out_masks.append(obj_masks.detach().cpu())

        
        # 计算 Loss 或者 直接返回预测结果
        if return_loss:
            return { 'loss':loss,
                'p' : self.loss_fn.p,
                'num_samples':B,
                'iou' : total_i / total_u }
        else:
            return {
                'pred_masks': out_masks,
                ** data_batch
            }

    def train_step(self,data_batch,optimizer,**kw):

        # data to cuda
        for k, v in data_batch.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data_batch[k] = v.cuda(non_blocking=True)

        output = self.forward(return_loss=True,_iter=kw['_iter'],**data_batch)
        loss = output['loss']
        return {
            # train_step 时，此处回传的 loss 会自动调用 backward
            'loss':loss,
            # log vars 会被记录
            'log_vars':{ 
                'loss':loss.detach().cpu(),
                'iou': output['iou'],
                'p': output['p']
            },
            # 用于对 log vars 计算平均时加权
            'num_samples' : output['num_samples']
        }
            

    def val_step(self,data_batch,**kw):
        # val_step 不会记录梯度
        return self.train_step(data_batch,optimizer=None,**kw)


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

    @property
    def broadcast_map(self):
        # [0,0,1,1,2,2,3,3,1] -> 9 objs , 4 image
        return [i for i,l in self.mem_objs]
    @property
    def aggregate_map(self):
        # [[0,1],[2,3,8],[4,5],[6,7]] -> 9 objs, 4 image
        B = len(set(self.broadcast_map))
        a_map = [[]]*B
        for oi,fi in enumerate(self.broadcast_map):
            a_map[fi].append(oi)
        return a_map

    @property
    def has_memory(self):
        return len(self.mem_objs) > 0

    def encode_value(self,image,kf16,obj_masks):
        ii = self.broadcast_map # broadcast B,C,H,W to N,C,H,W ,from B frames to N objects
        obj_mem_in = self.value_encoder(image[ii],kf16[ii],obj_masks.unsqueeze(1))
        return obj_mem_in

    def decode_mask(self,obj_mem_out,kf16_thin, kf8, kf4):
        ii = self.broadcast_map # broadcast B,C,H,W to N,C,H,W ,from B frames to N objects
        obj_mask_value = torch.cat([obj_mem_out, kf16_thin[ii]],dim=1)
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
            prob[obj_per_frame] = F.softmax(logits[obj_per_frame],dim=0)

        return prob,logits

    def add_memory(self,key,value):

        if self.mem_keys is None:
            self.mem_keys = key.unsqueeze(2) # B,C,T,H,W
            self.mem_values = value.unsqueeze(2) # N,C,T,H,W
        else:
            new_N,C,H,W = value.shape
            N,C,T,H,W = self.mem_values.shape
            self.mem_values = F.pad(self.mem_values,(0,0,0,0,0,0,0,0,new_N - N,0))
            self.mem_values = torch.cat([self.mem_values, value.unsqueeze(2)],dim=2)
            self.mem_keys = torch.cat([self.mem_keys, key.unsqueeze(2)],dim=2)

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
        obj_mem_out = torch.bmm(mo, obj_affinity) # N, CV, HW
        obj_mem_out = obj_mem_out.view(N, CV, H, W)
        return obj_mem_out
