from typing import Tuple

import numpy as np
import torch
import torchvision

target_img_means = [0.485, 0.456, 0.406]
target_img_stds = [0.229, 0.224, 0.225]

train_transforms: torchvision.transforms.Compose
dummy_transforms: torchvision.transforms.Compose
test_transforms: torchvision.transforms.Compose


def init_transforms(target_img_size: Tuple[int, int]) -> None:
    global train_transforms, dummy_transforms, test_transforms

    # train_transforms should be called in prepare-training-data.py
    # to save runtime CPU use (a lot)
    dummy_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            size=target_img_size,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True
        ),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2, saturation=0.2),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomRotation(5),
        torchvision.transforms.ToTensor(),
        # Why we use different means/std here?:
        # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        # torchvision.transforms.Normalize(mean=target_img_means, std=target_img_stds)
    ])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            size=target_img_size,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True
        ),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=target_img_means, std=target_img_stds)
    ])

def get_cuda_device(cuda_device: str = 'cuda') -> torch.device:
    if torch.cuda.is_available():
        return torch.device(cuda_device)
    raise RuntimeError('CUDA device not available')
