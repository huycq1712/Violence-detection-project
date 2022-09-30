import torch
import torchvision
import random
import math
import numpy as np
from PIL import Image
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw

from torchvision.transforms import functional as F
from torchvision.transforms import RandAugment
from torch.nn.functional import interpolate


class Compose:
    
    def __init__(self, transforms) -> None:
        self.transforms = transforms
        
    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
            
        return image, mask
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize:
    
    def __init__(self, size, interpolation=Image.BILINEAR) -> None:
        self.img_size =  size
        self.interpolation = interpolation
        
    def __call__(self, image, mask=None):
        image = F.resize(image, self.img_size, self.interpolation)
        if mask is not None:
            mask = F.resize(mask, self.img_size, self.interpolation)
        return image, mask


class RandomBrightness:
    
    def __init__(self, brightness_factor=0.3, prob=0.5) -> None:
        self.brightness_factor = brightness_factor
        self.prob = prob
        
    def __call__(self, image, mask=None):
        if random.random() < self.prob:
            image = F.adjust_brightness(image, self.brightness_factor)
        return image, mask

class RandomContrast:
    
    def __init__(self, contrast_factor=0.3, prob=0.5) -> None:
        self.contrast_factor = contrast_factor
        self.prob = prob
        
    def __call__(self, image, mask=None):
        if random.random() < self.prob:
            image = F.adjust_contrast(image, self.contrast_factor)
        return image, mask
    

class RandomSaturation:
    
    def __init__(self, saturation_factor=0.3, prob=0.5) -> None:
        self.saturation_factor = saturation_factor
        self.prob = prob
        
    def __call__(self, image, mask=None):
        if random.random() < self.prob:
            image = F.adjust_saturation(image, self.saturation_factor)
        return image, mask

class RandomHorizontalFlip:
    
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask=None):
        if random.random() < self.prob:
            image = F.hflip(image)
            mask = F.hflip(mask) if mask is not None else None
            
        return image, mask


class RandomVerticalFlip:
    
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask=None):
        if random.random() < self.prob:
            image = F.vflip(image)
            mask = F.vflip(mask) if mask is not None else None
            
        return image, mask


class CenterCrop:
    
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image):
        return F.center_crop(image, self.size)
    
    
class ToTensor:
    def __call__(self, image, mask=None):
        image = F.to_tensor(image)
        mask = F.to_tensor(mask) if mask is not None else None
        return image, mask


class ToPILImage:
    
    def __init__(self, mode=None) -> None:
        self.mode = mode
    
    def __call__(self, image):
        
        return F.to_pil_image(image, self.mode)
    
    
class Normalize:
    def __init__(self, mean, std, to_bgr255=False):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, mask=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        
        image = image.type(torch.float32)
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask


if __name__ == "__main__":
    m =  Resize(224)
    image = torch.rand(4, 5, 3, 256, 256)
    print(m(image).shape) 