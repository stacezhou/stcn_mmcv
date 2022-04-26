from stcn import dataset
from functools import partial
from mmdet.datasets import DATASETS,build_dataloader
from mmcv import Config
from mmcv.parallel import collate
from torch.utils.data import DataLoader,DistributedSampler

cfg = Config.fromfile('stcn/config/dataset/youtube.py')

data = DATASETS.build(cfg.data)
def vos_collate(batchs, samples_per_gpu=1):    
    transposed = zip(*batchs)
    out_batchs = [
            collate(t,samples_per_gpu=samples_per_gpu) 
            for t in transposed
            ]
    return out_batchs
        
loader = DataLoader(data, batch_size= 8, 
            collate_fn=partial(vos_collate, samples_per_gpu=8),
            )

for x in loader:
    break

print(len(x))