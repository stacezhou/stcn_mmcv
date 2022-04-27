from .stcn import VOSMODEL
import torch
import torch.nn.functional as F
import math

@VOSMODEL.register_module()
class AffinityMemoryBank():
    def __init__(self) -> None:
        self.reset()

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

    def read(self, K):
        Ks = self.Ks.flatten(start_dim=2) # B,C,THW
        K = K.flatten(start_dim=2) #B,C,HW

        # See supplementary material
        a_sq = Ks.pow(2).sum(1).unsqueeze(2) # B,THW,1
        ab = Ks.transpose(1, 2) @ K # B,THW,HW
        frame_affinity = (2*ab-a_sq) / math.sqrt(Ks.shape[1])   # B, THW, HW
        # del Ks, K
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(frame_affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(frame_affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        frame_affinity = x_exp / x_exp_sum  # B,THW,HW
        # del maxes,x_exp,x_exp_sum,ab,a_sq # save memory

        N,C,T,H,W = self.Vs.shape
        obj_affinity = frame_affinity[self.ii] # N,THW,HW
        obj_affinity = obj_affinity.view((N,T,H*W*H*W)) * self.gate.unsqueeze(2)
        obj_affinity = obj_affinity.view((N,T*H*W,H*W))

        # del frame_affinity

        Vs = self.Vs.flatten(start_dim=2)
        V = torch.bmm(Vs, obj_affinity).view_as(self.Vs[:,:,0])
        return V


    def write(self,K,V):
        K = K.unsqueeze(2) # B,C,T,H,W
        V = V.unsqueeze(2) # N,C,T,H,W
        if not self.is_init:
            self.Ks = K
            self.Vs = V
            self.gate = torch.ones((V.shape[0],1)).to(V.device)
            self.is_init = True
        else:
            self.Ks = torch.cat([self.Ks, K], dim=2)
            self.Vs = torch.cat([self.Vs, V], dim=2)
            self.gate = F.pad(self.gate, (0,1,0,0),value=1)
