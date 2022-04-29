from stcn.para import get_config
from stcn.trainer import train_model
from stcn import VOSMODEL
from mmdet.datasets import build_dataset

def main():
    cfg, meta, timestamp, distributed = get_config()
    model = VOSMODEL.build(cfg.model)
    model.init_weights()
    datasets = [build_dataset(cfg.data.train)]

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
