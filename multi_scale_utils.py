import torch.nn.functional as F 
import torch

class MultiScaleTest():

    def __init__(self,size,scales,align_corners = True) -> None:
        self.size = size
        self.out_sizes = [(int(size[0]*sc),int(size[1]*sc)) for sc in scales]
        self.align_corners = align_corners
        
    def scales(self, tensor,mode='nearest',out_sizes = None):
        '''
        get multi scale output from single input tensor
        '''
        if out_sizes is None:
            out_sizes = self.out_sizes
        *B,C,H,W = tensor.shape
        if mode in ['linear', 'bilinear', 'bicubic','trilinear']:
            return [
                F.interpolate(tensor.reshape((-1,C,H,W)), size=out_size, 
                    mode=mode, align_corners=self.align_corners).reshape((*B,C,*out_size))
                for out_size in out_sizes
            ]
        else:
            return [
                F.interpolate(tensor.reshape((-1,C,H,W)), size=out_size, 
                    mode=mode).reshape((*B,C,*out_size))
                for out_size in out_sizes
            ]

    def unscales(self, tensores,mode='bilinear', size = None):
        '''
        resize all input [tensorer] to [size]
        '''
        if size is None:
            size = self.size
        if mode in ['linear', 'bilinear', 'bicubic','trilinear']:
            return [
                F.interpolate(tensor, size=size,mode=mode,align_corners=self.align_corners)
                for tensor in tensores
            ]
        else:
            return [
                F.interpolate(tensor, size=size,mode=mode)
                for tensor in tensores
            ]
        

    def interact(self, gen_out_mask_list):
        multi_scale_mean_mask = [None] * len(gen_out_mask_list)
        finish = False
        while not finish:
            multi_scale_out_mask = []
            for i,gen_out_mask in enumerate(gen_out_mask_list):
                try:
                    out_mask = gen_out_mask.send(multi_scale_mean_mask[i])
                except StopIteration:
                    finish = True
                else:
                    multi_scale_out_mask.append(out_mask)
            if finish:
                break
            out_sizes = [tuple(out.shape[-2:]) for out in multi_scale_out_mask]
            out_mask = self.unscales(multi_scale_out_mask,mode='bilinear',size=out_sizes[0])
            mean_mask = torch.stack(out_mask).mean(dim=0)
            multi_scale_mean_mask = self.scales(mean_mask,mode='nearest',out_sizes=out_sizes) 
            for send,get in zip(multi_scale_mean_mask,multi_scale_out_mask):
                assert tuple(send.shape) == tuple(get.shape)
