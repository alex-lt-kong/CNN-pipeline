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
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1, saturation=0.1),
    torchvision.transforms.RandomGrayscale(p=0.1),
    torchvision.transforms.RandomRotation(3),
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


def read_config_file() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', '-c', dest='config', required=True,
                    help='the path of the JSON config file')
    args = vars(ap.parse_args())
    config_path = args['config']
    if os.path.isfile(config_path) is False:
        raise FileNotFoundError(f'File [{config_path}] not found')
    with open(config_path, 'r') as json_file:
        json_str = json_file.read()
        settings = json.loads(json_str)
        assert isinstance(settings, Dict)
    return settings


def preprocess_image(image_path: str) -> Image:
    image = Image.open(image_path)
    image = test_transforms(image).unsqueeze(0)
    return image


# def main() -> None:
#    settings = read_config_file()
#    v16mm = model.VGG16MinusMinus(2)
#    v16mm.load_state_dict(torch.load(settings['model']['parameters']))
#    image_path = input("Enter the path of image to inference:\n")
#    output = v16mm(preprocess_image(image_path))
#    print(output)
#    y_pred = torch.argmax(output, dim=1)
#    print(y_pred)


# if __name__ == '__main__':
#    main()
