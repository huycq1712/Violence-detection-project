from torchvision.transforms import RandAugment
from . import transforms as T


def build_transforms(cfg, is_train=True):
    height = cfg.INPUT.HEIGHT
    width = cfg.INPUT.WIDTH
    ratio = cfg.INPUT.RATIO

    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.CenterCrop((ratio*height, ratio*width)),
            T.ToTensor(),
            RandAugment(),
            normalize_transform
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.CenterCrop((ratio * height, ratio * width)),
            T.ToTensor(),
            normalize_transform
        ])

    return transform