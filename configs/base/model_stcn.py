K_dim = 64
V_dim = 512
custom_imports = dict(
    imports=['stcn.loss.bce','stcn.model','stcn.nni_hook'],
    allow_failed_imports=False)
model = dict(
    type = 'STCN',
    init_cfg = dict(type='Kaiming', layer='Conv2d'),
    seg_background = False,
    max_objs_per_frame = 3,
    multi_scale_train = False,
    train_scales = [1],
    multi_scale_test = False,
    test_scales = [1, 1.3, 2],
    align_corners = True,
    key_encoder = dict(
        type = 'KeyEncoder',
        backbone = dict(
            type='ResNet',
            depth=50,
            out_indices=(0, 1, 2),
            frozen_stages=3,
            init_cfg=dict(type='Pretrained', 
                checkpoint='torchvision://resnet50')
        ),
        key_proj = dict(
            type='KeyProjection', 
            indim = 1024,
            keydim = K_dim, 
            ortho_init = True,
        ),
        key_comp = dict(
            type='KeyProjection',
            indim = 1024,
            keydim = V_dim,
        ),
    ),
    value_encoder = dict(
        type = 'ValueEncoder',
        backbone=dict(
            type='ResNet',
            depth=18,
            in_channels=4,
            out_indices=(2,),
        ),
        feature_fusion = dict(
            type = 'FeatureFusionBlock',
            indim = 1024 + 256,
            outdim = V_dim,
        )
    ),
    mask_decoder = dict(
        type = 'MaskDecoder',
        indim = V_dim,
    ),
    memory = dict(
        type= 'AffinityMemoryBank',
        thin_reading_scale = 8,
        top_k = 20,
        mem_every = 5,
        include_last = True,
        train_memory_strategy = False,
    ),
    loss_fn = dict(
        type = 'BootstrappedCE',
        start_warm=10000, 
        end_warm=40000, 
        top_p=0.15
    ),
)