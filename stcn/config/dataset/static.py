pipeline= [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile'),
]
data = dict(
    type = 'StaticDataset',
    transforms = pipeline,
    num_frames = 3,
    image_root = '/data/static/BIG_small',
)