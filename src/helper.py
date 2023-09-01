from PIL import Image
from typing import Any, Dict

import argparse
import json
import os
import torch
import torchvision

target_img_size = (224, 426)  # size is in (h,w)
target_img_means = [0.485, 0.456, 0.406]
target_img_stds = [0.229, 0.224, 0.225]
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=target_img_size),
    # torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, hue=0.25, saturation=0.25),
    torchvision.transforms.RandomGrayscale(p=0.5),
    torchvision.transforms.RandomRotation(5),
    torchvision.transforms.ToTensor(),
    # Why we use different means/std here?:
    # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    torchvision.transforms.Normalize(mean=target_img_means, std=target_img_stds)
])
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=target_img_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=target_img_means, std=target_img_stds)
])


def get_cuda_device() -> torch.device:
    # Check if GPU is available
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        raise RuntimeError('CUDA device not available')


def preprocess_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)
    image_tensor = test_transforms(image).unsqueeze(0)
    assert isinstance(image_tensor, torch.Tensor)
    return image_tensor
