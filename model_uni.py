import torch
import torch.nn.functional as F
import math
from model.network import STCN
from model.losses import LossComputer
from model.modules import ValueEncoder

class STCNModel(STCN):

    def __init__(self):
        super().__init__(single_object=False)
        self.value_encoder = ValueEncoder()
        self.loss = None
        # self.loss_fn = LossComputer()
        self.MAX_NUM_MEMORY = 5
        self.C = 128 # value embedding dimension
    
    def reset_memory(self,mask):
        self.K = len(mask.unique()) # num_objects
        self.num_frames = 0
        B,N,H,W = mask.shape
        h,w = H//16,W//16
        T = self.MAX_NUM_MEMORY
        self.mem_keys = torch.zeros((B,T,H,W))
        self.mem_value = torch.zeros((B,T,self.C,h,w))
        self.mem_masks = torch.zeros((B,T,N,H,W))
    
    def add_memory(self,key,value,mask):
        if self.num_frames < self.MAX_NUM_MEMORY:
            i = self.num_frames
            self.mem_keys[i] = key
            self.mem_values[i] = value
            self.mem_mask[i] = mask
        else:
            self.mem_keys = torch.cat([self.mem_keys,key])

        self.num_frames += 1

    def read_memory(self,key):
        mk = self.mem_keys[:self.num_frames]
        mv = self.mem_value[:self.num_frames]
        qk = key
        
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum 

        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem_out = mem.view(B, CV, H, W)
        return mem_out

    def mask_decoder(self,read_value,kf16_thin, kf16, kf8, kf4):
        mask_value = torch.cat([read_value, kf16_thin])
        logits = self.decoder(mask_value,kf8,kf4)
        prob = torch.sigmoid(logits)
        logits = self.aggregate(prob)
        prob = F.softmax(logits,dim=1)[:,1:]
        return logits, prob

    def forward(self,current_frame):
        image = current_frame['rgb']  # BxCxHxW
        key, *fine_features = self.encode_key(image.unsqueeze(1))
        # kf16_thin, kf16, kf8, kf4 
        if current_frame['is_ref']:
            mask = current_frame['gt'] # BxHxW
            logits = None
            self.reset_memory(mask)
        else:
            read_value = self.read_memory(key)
            mask,logits = self.mask_decoder(read_value, *fine_features)

        mem_value = self.value_encoder(image,key,mask.squeeze(),mask.squeeze())
        self.add_memory(key,mem_value,mask)
        return {
            'pred_mask':mask,
            'pred_logits':logits,
            **current_frame
        }

    def train_step(self,data,optimizer):
        B,T,C,H,W = data['rgb'].shape
        loss = 0
        for i in range(T):
            currents_frame = {
                'rgb':data['rgb'][:,i],
                'gt':data['gt'][:,i],
                'cls_gt':data['cls_gt'][:,i],
                'sec_gt':data['sec_gt'][:,i],
                'selector':data['selector'],
                'name':data['info']['name'],
                'frames': [x[i] for x in data['info']['frames']],
                'is_ref': True if i == 0 else False
            }
            output = self.forward(currents_frame)
            loss += self.loss_fn(output)

        return loss
            
    def val_step(self,data,optimizer):
        with torch.no_grad():
            output = self.forward(data)
        return output