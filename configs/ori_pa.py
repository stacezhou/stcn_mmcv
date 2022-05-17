from configs.stcn_origin import *
model['mask_decoder'].update(dict(
    use_PA = True,
    pa_config = dict(
        require_grad=False,
        init_cfg = dict(
            type='Pretrained',
            checkpoint='/home/zh21/STCN/GGN/PA.pth',
        )
    )
))