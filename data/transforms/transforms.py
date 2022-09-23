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
        
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
            
        return image
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize:
    
    def __init__(self, size, interpolation=Image.BILINEAR) -> None:
        self.img_size = size
        self.interpolation = interpolation
        
    def __call__(self, image):
        image = F.resize(image, self.img_size, self.interpolation)
        return image
    
    
class RandomHorizontalFlip:
    
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = F.hflip(image)
            
        return image


class RandomHorizontalFlip:
    
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = F.vflip(image)
            
        return image


class CenterCrop:
    
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image):
        return F.center_crop(image, self.size)
    
    
class ToTensor:
    def __call__(self, image):
        image = F.to_tensor(image)
        return image


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

    def __call__(self, image):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image