from stcn.para import get_config
from stcn.trainer import train_model
from stcn.dataset import ConcatDataset
from stcn import VOSMODEL
from mmdet.datasets import build_dataset

def main():
    cfg, meta, timestamp, distributed = get_config()
    model = VOSMODEL.build(cfg.model)
    model.init_weights()
    if isinstance(cfg.data.train, (list,tuple)):
        datasets = ConcatDataset([build_dataset(ds) for ds in cfg.data.train])
    else:
        datasets = build_dataset(cfg.data.train)

    train_model(
        model,
        datasets,
        cfg,
        validate = cfg.validate,
        distributed=distributed,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
