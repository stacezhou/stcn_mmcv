from mmcv.runner import BaseModule
from collections import defaultdict

class STCN(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.key_encoder = None
        self.value_encoder = None
        self.memory = None
        self.mask_decoder = None
        self.loss_fn = None
    
    def forward(self,img, flag, return_loss=False,*k,**kw):
        if flag == 'new_video':
            self.memory.reset()

        k_feats = self.key_encoder(img)
        v_feats = self.memory.read(k_feats)
        mask = self.mask_decoder(img, v_feats)
        v_feats = self.value_encoder(mask, v_feats)
        self.memory.write(k_feats, v_feats)

        if return_loss:
            loss = self.loss_fn(mask, kw['gt_mask'])
            output = {'loss': loss}
        else:
            output = {'mask': mask}

        return output

    def train_step(self, data_batch, optimizer, **kw):
        output = defaultdict(list)
        for i,data in enumerate(data_batch):
            flag = 'new_video' if i == 0 else ''
            output[i] = self.forward(**data,return_loss=True,flag=flag)

        loss = sum([item['loss'] for key,item in output.items()])
        return {
            'loss' : loss
        }

    
    def val_step(self,data_batch,**kw):
        # val_step 不会记录梯度
        return self.forward(**data_batch,return_loss=False)