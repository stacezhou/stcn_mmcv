youtube_path = dict(
    train = dict(
        image_root = '/data/YouTube/train_480p/JPEGImages',
        mask_root = '/data/YouTube/train_480p/Annotations',
    ),
    val = dict(
        image_root = '/data/YouTube/valid/JPEGImages',
        mask_root = '/data/YouTube/valid/Annotations',
        palette = '/data/YouTube/valid/Annotations/0a49f5265b/00000.png',
    ),
    mini_val = dict(
        image_root = '/data/YouTube/debug/JPEGImages',
        mask_root = '/data/YouTube/debug/valid_Annotations',
        palette = '/data/YouTube/valid/Annotations/0a49f5265b/00000.png',
    ),
)
davis_path = dict(
    image_root = '/data/DAVIS/2017/trainval/JPEGImages/480p',
    mask_root = '/data/DAVIS/2017/trainval/Annotations/480p',
)
ovis_path = dict(
    image_root = '/data/OVIS_img/train',
    mask_root = '/data/OVIS_anno/OVIS_anno',
)