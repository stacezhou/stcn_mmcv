from .stcn import VOSMODEL
import torch
import math

@VOSMODEL.register_module()
class AffinityMemoryBank():
    def __init__(self) -> None:
        self.targets = []
        self.Ks = []
        self.Vs = []
        self.is_init = False

    def update_targets(self, broadcast_map):
        self.ii = broadcast_map

    def reset(self):
        self.is_init = False

    def read(self, K):
        Ks = self.Ks.flatten(start_dim=2) # B,C,THW
        Vs = self.Vs.flatten(start_dim=2)
        K = K.flatten(start_dim=2) #B,C,HW

        # See supplementary material
        a_sq = Ks.pow(2).sum(1).unsqueeze(2) # B,THW,1
        ab = Ks.transpose(1, 2) @ K # B,THW,HW
        frame_affinity = (2*ab-a_sq) / math.sqrt(Ks.shape[1])   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(frame_affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(frame_affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        frame_affinity = x_exp / x_exp_sum  # B,THW,HW
        del maxes,x_exp,x_exp_sum,ab,a_sq,mk,qk # save memory

        obj_affinity = frame_affinity[self.ii] # N,THW,HW
        del frame_affinity

        V = torch.bmm(Vs, obj_affinity).view_as(self.Vs[:,:,0])
        return V


    def write(self,K,V):
        K = K.unsqueeze(2) # B,C,T,H,W
        V = V.unsqueeze(2) # N,C,T,H,W
        if not self.is_init:
            self.Ks = K
            self.Vs = V
            self.is_init = True
        else:
            self.Ks = torch.cat([self.Ks, K], dim=2)
            self.Vs = torch.cat([self.Vs, V], dim=2)
