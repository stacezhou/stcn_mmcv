model = dict(
    type = 'STCN',
    key_encoder = dict(
        type = 'KeyEncoder',
        backbone = dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
        key_proj = dict(
            type='KeyProjection', 
            indim = 1024,
            keydim = 64,
            ortho_init = True,
        ),
        key_comp = dict(
            type='KeyProjection',
            indim = 1024,
            keydim = 512,
        ),
    ),
    value_encoder = dict(
        type = 'ValueEncoder',
        backbone=dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=2,
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', 
                    checkpoint='torchvision://resnet50')),
        feature_fusion = dict(
            type = 'FeatureFusionBlock',
            indim = 1024 + 512,
            outdim = 512,
        )
    ),
    mask_decoder = dict(
        type = 'MaskDecoder',
    ),
    memory = dict(type= 'AffinityMemoryBank'),
    loss_fn = dict(type = 'StcnBCELoss'),
)