from PIL import Image
from typing import Any, List

import argparse
import helper
import logging
import json
import model
import os
import PIL
import sys
import time
import torch

curr_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tensor_from_img_dir(images_dir: str) -> torch.Tensor:
    images: List[Image.Image] = []
    for filename in sorted(os.listdir(images_dir)):
        file_path = os.path.join(images_dir, filename)
        if os.path.isfile(file_path):
            image = Image.open(file_path)
            images.append(image)
            assert isinstance(image, Image.Image)

    tensor_images = torch.empty(
        (len(images), 3, helper.target_img_size[0], helper.target_img_size[1])
    )
    for i in range(len(images)):
        assert images[i] is not None
        tensor_images[i] = helper.test_transforms(images[i])
        assert isinstance(tensor_images[i], torch.Tensor)

    return tensor_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-dir', '-d', dest='image-dir', required=True,
                    help='Directory that is full of images!')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d | %(levelname)7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    args = vars(ap.parse_args())
    image_path = str(args['image-dir'])

    with open(os.path.join(curr_dir, '..', 'config.json')) as j:
        settings = json.load(j)
    v16mm = model.VGG16MinusMinus(2)
    logging.info(f'Loading parameters from {settings["model"]["parameters"]}')
    v16mm.load_state_dict(torch.load(settings['model']['parameters']))

    logging.info('Parameters loaded')

    preview_ele_num = 5
    layer_count = 0
    logging.info("Sample values from some layers:")
    for name, param in v16mm.named_parameters():
        layer_count += 1
        if layer_count % 5 != 0:
            continue
        if param.requires_grad:
            if len(param.data.shape) <= 1:
                logging.info(
                    f'{name}({param.data.shape}): {param.data[:preview_ele_num]}'
                )
            elif len(param.data.shape) <= 2:
                logging.info(
                    f'{name}({param.data.shape}): {param.data[0][:preview_ele_num]}'
                )
            elif len(param.data.shape) <= 3:
                logging.info(
                    f'{name}({param.data.shape}): {param.data[0][0][:preview_ele_num]}'
                )
            elif len(param.data.shape) <= 4:
                logging.info(
                    f'{name}({param.data.shape}): {param.data[0][0][0][:preview_ele_num]}'
                )
            else:
                logging.info(
                    f'{name}({param.data.shape}): {param.data[0][0][0][0][:preview_ele_num]}'
                )

    v16mm.to(device)
    v16mm.eval()
    logging.info(f'Loading and transforming image from {image_path}')
    images_tensor = get_tensor_from_img_dir(image_path)

    rounded_unix_time = int(time.time() / 100000)
    h = rounded_unix_time % helper.target_img_size[0]
    w = rounded_unix_time % (helper.target_img_size[1] - preview_ele_num)
    logging.info(
        f'Image tensor ready, tensor shape: {images_tensor.shape}, '
        f'sample vlaues start from (w{w}, h{h}):'
    )

    for i in range(0, images_tensor.shape[1]):
        logging.info(images_tensor[0][i][h][w: w + preview_ele_num])

    images_tensor = images_tensor.to(device)
    logging.info('Running inference')
    output = v16mm(images_tensor)
    assert isinstance(output, torch.Tensor)
    logging.info(f'Done, raw output:\n{output}')
    y_pred = torch.argmax(output, dim=1)
    logging.info(f'y_pred: {y_pred}')


if __name__ == '__main__':
    main()
