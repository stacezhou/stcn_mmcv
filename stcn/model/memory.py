from .stcn import VOSMODEL

@VOSMODEL.register_module()
class AffinityMemoryBank():
    def __init__(self) -> None:
        self.v_feats = None
        pass

    def reset(self):
        pass

    def read(self, k_feats):
        return self.v_feats

    def write(self,k_feats, v_feats):
        self.v_feats = v_feats
