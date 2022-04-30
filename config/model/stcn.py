key_dim = 64
value_dim = 512
custom_imports = dict(
    imports=['stcn.loss.bce','stcn.model'],
    allow_failed_imports=False)
model = dict(
    type = 'STCN',
    init_cfg = dict(type='Kaiming', layer='Conv2d'),
    seg_background = False,
    key_encoder = dict(
        type = 'KeyEncoder',
        backbone = dict(
            type='ResNet',
            depth=50,
            out_indices=(0, 1, 2),
            frozen_stages=1,
            init_cfg=dict(type='Pretrained', 
                checkpoint='torchvision://resnet50')
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
            in_channels=4,
            out_indices=(2,),
            frozen_stages=1,
            # init_cfg=dict(type='Pretrained', 
                # checkpoint='torchvision://resnet18')
        ),
        feature_fusion = dict(
            type = 'FeatureFusionBlock',
            indim = 1024 + 256,
            outdim = 512,
        )
    ),
    mask_decoder = dict(
        type = 'MaskDecoder',
        indim = 512,
    ),
    memory = dict(
            type= 'AffinityMemoryBank',
            top_k = -1,
            mem_every = 5,
            include_last = False,
            thin_reading_scale = 8,
    ),
    loss_fn = dict(type = 'BootstrappedCE'),
)