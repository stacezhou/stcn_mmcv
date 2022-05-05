from .stcn import VOSMODEL
import torch
import torch.nn.functional as F
import math

@VOSMODEL.register_module()
class AffinityMemoryBank():
    def __init__(self, 
            thin_reading_scale = 8,
            top_k = 20,
            mem_every = 5,
            include_last = False,
            train_memory_strategy = False,
            ) -> None:
        self.top_k = top_k
        self.test_mode = False
        self.mem_every = mem_every
        self.include_last = include_last
        self.thin_reading_scale = thin_reading_scale
        self.train_memory_strategy = train_memory_strategy
        self.reset()
    
    def train(self, mode=True):
        if mode == True:
            self.test_mode = False 
        else:
            self.test_mode = True 

    def update_targets(self, fi_list):
        if self.is_init:
            obj_new = len(fi_list) - len(self.ii)
            self.ii = fi_list
            pad = (0,0,0,0,0,0, 0,0,0,obj_new)
            self.Vs = F.pad(self.Vs, pad)
            self.gate = F.pad(self.gate, (0,0,0,obj_new))
        else:
            self.ii = fi_list


    def reset(self):
        self.is_init = False
        self.ii = []
        self.Ks = []
        self.Vs = []

    def _read(self, K):
        B,Ck,h,w = K.shape
        N,Cv,T,H,W = self.Vs.shape

        gate = self.gate.view((N,1,T,1,1))
        Ks = self.Ks[self.ii] * gate
        K = K[self.ii]

        Ks = Ks.flatten(start_dim=2) # B,C,THW
        K = K.flatten(start_dim=2) #B,C,HW

        # See supplementary material
        a_sq = Ks.pow(2).sum(1).unsqueeze(2) # B,THW,1
        ab = Ks.transpose(1, 2) @ K # B,THW,HW
        affinity = (2*ab-a_sq) / math.sqrt(Ks.shape[1])   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        if self.test_mode and 0 < self.top_k < affinity.shape[1]:
            maxes = torch.max(affinity, dim=1, keepdim=True)[0]
            values, indices = torch.topk(affinity - maxes, k=self.top_k, dim=1)
            x_exp = values.exp_()
            x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
            # The types should be the same already
            # some people report an error here so an additional guard is added
            affinity.zero_().scatter_(1, indices, x_exp.type(affinity.dtype)) # B * THW * HW
            
        else: # softmax
            maxes = torch.max(affinity, dim=1, keepdim=True)[0]
            x_exp = torch.exp(affinity - maxes)
            x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
            affinity = x_exp / x_exp_sum  # B,THW,HW

        Vs = self.Vs.flatten(start_dim=2)
        V = torch.bmm(Vs, affinity).view((N,Cv,h,w))
        return V
    
    def read(self, K):
        B,C,H,W = K.shape
        step = (W+self.thin_reading_scale) // self.thin_reading_scale
        V_ = []
        for i in range(0,W,step):
            V_.append(self._read(K[:,:,:,i:i+step]))
        V = torch.cat(V_, dim=3)
        return V



    def write(self,K,V):
        K = K.unsqueeze(2) # B,C,T,H,W
        V = V.unsqueeze(2) # N,C,T,H,W
        if not self.is_init:
            self.Ks = K
            self.Vs = V
            self.gate = torch.ones((V.shape[0],1)).to(V.device)
            self.is_init = True
        elif not self.test_mode:
            self.Ks = torch.cat([self.Ks, K], dim=2)
            self.Vs = torch.cat([self.Vs, V], dim=2)
            self.gate = F.pad(self.gate, (0,1,0,0),value=1) # N,T
        else:
            T = self.gate.shape[1]
            get_new_mem_frame = ((T-1) % self.mem_every == 0)
            if self.include_last and not get_new_mem_frame:
                self.Ks[:,:,-1] = K.squeeze(2)
                self.Vs[:,:,-1] = V.squeeze(2)
            elif self.include_last and get_new_mem_frame:
                self.Ks = torch.cat([self.Ks, K], dim=2)
                self.Vs = torch.cat([self.Vs, V], dim=2)
                self.gate = F.pad(self.gate, (0,1,0,0),value=1) # N,T
            else:
                return