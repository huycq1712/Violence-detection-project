from torchvision.transforms import RandAugment
from . import transforms as T


def build_transforms(cfg, is_train=True):
    height = cfg.INPUT.HEIGHT
    width = cfg.INPUT.WIDTH
    #ratio = cfg.INPUT.RATIO

    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    #normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])

    if is_train:
        transform = T.Compose([
            T.RandomBrightness(),
            T.RandomContrast(),
            T.RandomSaturation(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Resize((height, width)),
            #T.CenterCrop((ratio*height, ratio*width)),
            #T.ToTensor(),
            #RandAugment(),
            normalize_transform
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            #T.CenterCrop((ratio * height, ratio * width)),
           # T.ToTensor(),
            normalize_transform
        ])

    optical_trainform = T.Compose([
        T.Resize((height, width)),
        #T.CenterCrop((ratio * height, ratio * width)),
        #T.ToTensor(),
        normalize_transform
    ])

    return transform, optical_trainform


if __name__ == "__main__":
    #from config import cfg
    import torch
    transform = build_transforms(cfg=None)

    a = torch.rand(4, 12, 3, 224, 224)
    print(transform(a).shape)