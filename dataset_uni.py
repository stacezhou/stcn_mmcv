from torch.utils.data import ConcatDataset
from dataset.static_dataset import StaticTransformDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataset.vos_dataset import VOSDataset
from util.hyper_para import HyperParameters
from os import path
from util.load_subset import load_sub_davis, load_sub_yv

# Parse command line arguments
para = HyperParameters()
para.parse()

static_root = path.expanduser(para['static_root'])
yv_root = path.join(path.expanduser(para['yv_root']), 'train_480p')
davis_root = path.join(path.expanduser(para['davis_root']), '2017', 'trainval')
bl_root = path.join(path.expanduser(para['bl_root']))

skip_values = [10, 15, 20, 25, 5]

def get_dataset(stage=3,max_skip=5,valset=False):
    print('Renewed with skip: ', max_skip)
    if valset == True:
        if stage != 0:
            yv_dataset = VOSDataset(path.join(yv_root, 'JPEGImages'), 
                                path.join(yv_root, 'Annotations'), max_skip//5, is_bl=False,val=True, subset=load_sub_yv())
            davis_dataset = VOSDataset(path.join(davis_root, 'JPEGImages', '480p'), 
                                path.join(davis_root, 'Annotations', '480p'), max_skip, is_bl=False,val=True, subset=load_sub_davis())
            eval_dataset = ConcatDataset([davis_dataset]*25 + [yv_dataset]*5)
            return eval_dataset
        else:
            duts_te_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TE'), method=1)
            return duts_te_dataset



    if stage == 0:
        # fss_dataset = StaticTransformDataset(path.join(static_root, 'fss'), method=0)
        duts_tr_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TR'), method=1)
        ecssd_dataset = StaticTransformDataset(path.join(static_root, 'ecssd'), method=1)
        big_dataset = StaticTransformDataset(path.join(static_root, 'BIG_small'), method=1)
        hrsod_dataset = StaticTransformDataset(path.join(static_root, 'HRSOD_small'), method=1)

        # BIG and HRSOD have higher quality, use them more
        train_dataset = ConcatDataset([duts_tr_dataset, ecssd_dataset]
                + [big_dataset, hrsod_dataset]*5)

        print('Static dataset size: ', len(train_dataset))
        return train_dataset
    elif stage == 1:
        train_dataset = VOSDataset(path.join(bl_root, 'JPEGImages'), 
                            path.join(bl_root, 'Annotations'), max_skip, is_bl=True)

        print('Blender dataset size: ', len(train_dataset))
        return train_dataset
    else: # stage 2 or 3
        yv_dataset = VOSDataset(path.join(yv_root, 'JPEGImages'), 
                            path.join(yv_root, 'Annotations'), max_skip//5, is_bl=False, subset=load_sub_yv())
        davis_dataset = VOSDataset(path.join(davis_root, 'JPEGImages', '480p'), 
                            path.join(davis_root, 'Annotations', '480p'), max_skip, is_bl=False, subset=load_sub_davis())
        train_dataset = ConcatDataset([davis_dataset]*50 + [yv_dataset]*10)

        print('YouTube dataset size: ', len(yv_dataset))
        print('DAVIS dataset size: ', len(davis_dataset))
        print('Concat dataset size: ', len(train_dataset))
        return train_dataset

increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.8, 1.0]

def get_dataloader(dataset,world_size,local_rank=0,**kw):
    batch_size = 4
    if world_size > 1:
        train_sampler = DistributedSampler(dataset, rank=local_rank,shuffle=True)
        data_loader = DataLoader(dataset,batch_size, sampler= train_sampler,num_workers=para['num_workers'],drop_last=True, pin_memory=True)
    else:
        data_loader = DataLoader(dataset,batch_size,num_workers=para['num_workers'],drop_last=True, pin_memory=True)
    return data_loader
