from stcn.para import get_config
from stcn.trainer import train_model
from stcn import VOSMODEL
from mmdet.datasets import DATASETS

def main():
    cfg, meta, timestamp, distributed = get_config()
    model = VOSMODEL.build(cfg.model)
    model.init_weights()
    datasets = [DATASETS.build(cfg.data.train)]

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
