from mmcv.runner import BaseModule
from mmcv.utils import Registry
from collections import defaultdict
from mmdet.models import BACKBONES

VOSMODEL = Registry('vos_model')

@VOSMODEL.register_module()
class STCN(BaseModule):
    def __init__(self, 
                key_encoder,
                key_proj,
                key_comp,
                value_encoder,
                mask_decoder,
                memory,
                loss_fn,
                init_cfg=None):
        super().__init__(init_cfg)
        self.key_encoder = BACKBONES.build(key_encoder)
        self.value_encoder = VOSMODEL.build(value_encoder)
        self.mask_decoder = VOSMODEL.build(mask_decoder)
        self.memory = VOSMODEL.build(memory)
        self.loss_fn = loss_fn
    
    def forward(self,img, gt_mask, flag, return_loss=False,*k,**kw):
        K, f16_thin, f16, f8, f4 = self.key_encoder(img) 
        if flag == 'new_video':
            # self.memory.reset()
            V = self.value_encoder(img, gt_mask, f16)
            self.memory.write(K, V)

        V = self.memory.read(K)
        mask = self.mask_decoder(mask, K, f16_thin, f8, f4, V)
        V = self.value_encoder(img, mask, f16)
        self.memory.write(K, V)

        if return_loss:
            loss = self.loss_fn(mask, kw['gt_mask'])
            output = {'loss': loss}
        else:
            output = {'mask': mask}

        return output

    def train_step(self, data_batch, optimizer, **kw):
        output = defaultdict(list)
        for i,data in enumerate(data_batch):
            img = data['img'].data[0]
            gt_mask = data['gt_mask']
            flag = 'new_video' if i == 0 else ''
            output[i] = self.forward(
                    img = img,
                    gt_mask = gt_mask,
                    flag=flag,
                    return_loss=True,
                    )

        loss = sum([item['loss'] for key,item in output.items()])
        return {
            'loss' : loss
        }

    
    def val_step(self,data_batch,**kw):
        # val_step 不会记录梯度
        return self.forward(**data_batch,return_loss=False)

@VOSMODEL.register_module()
class PASS:
    def __init__(self):
        pass

    def __call__(self,*k,**kw):
        return (k,kw)
        