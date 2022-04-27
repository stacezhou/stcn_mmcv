from mmdet.apis import train_detector as train_model
from stcn.para import get_config
from stcn import VOSMODEL,VOSDATASETS

def main():
    cfg, args, meta, timestamp, distributed = get_config()
    model = VOSMODEL.build(cfg.model)
    model.init_weights()
    datasets = [VOSDATASETS.build(cfg.data.train)]

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
