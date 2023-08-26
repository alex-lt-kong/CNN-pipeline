from PIL import Image
from typing import Any, Dict

import argparse
import json
import model
import os
import torch
import torchvision

# This transforms should be similar to the transform defined in model.py
# but withOUT unnecessary distortions.
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 426)),  # size is in (h,w)
    torchvision.transforms.ToTensor(),
    # Why we use different means/std here?:
    # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])


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
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 426)),  # size is in (h,w)
        torchvision.transforms.ToTensor(),
        # Why we use different means/std here?:
        # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
    image = transforms(image).unsqueeze(0)
    return image


def main() -> None:
    settings = read_config_file()
    v16mm = model.VGG16MinusMinus(2)
    v16mm.load_state_dict(torch.load(settings['model']['parameters']))
    image_path = input("Enter the path of image to inference:\n")
    output = v16mm(preprocess_image(image_path))
    print(output)
    y_pred = torch.argmax(output, dim=1)
    print(y_pred)


if __name__ == '__main__':
    main()
