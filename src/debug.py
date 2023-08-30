from typing import Any, Dict

import argparse
import helper
import logging
import json
import model
import os
import sys
import torch

settings: Dict[str, Any]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', '-c', dest='config', required=True,
                    help='Path of the JSON config file')
    ap.add_argument('--image-path', '-p', dest='image-path', required=True,
                    help='Path of image to be inferenced')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d | %(levelname)7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    args = vars(ap.parse_args())
    config_path = args['config']
    image_path = args['image-path']

    if os.path.isfile(config_path) is False:
        raise FileNotFoundError(f'File [{config_path}] not found')
    with open(config_path, 'r') as json_file:
        json_str = json_file.read()
        settings = json.loads(json_str)
        assert isinstance(settings, Dict)
    v16mm = model.VGG16MinusMinus(2)
    logging.info(f'Loading parameters from {settings["model"]["parameters"]}')
    v16mm.load_state_dict(torch.load(settings['model']['parameters']))
    logging.info('Parameters loaded')

    preview_ele_num = 5
    for name, param in v16mm.named_parameters():
        if param.requires_grad:
            if len(param.data.shape) <= 1:
                logging.info(f'{name}({param.data.shape}): {param.data[:preview_ele_num]}')
            elif len(param.data.shape) <= 2:
                logging.info(f'{name}({param.data.shape}): {param.data[0][:preview_ele_num]}')
            elif len(param.data.shape) <= 3:
                logging.info(f'{name}({param.data.shape}): {param.data[0][0][:preview_ele_num]}')
            elif len(param.data.shape) <= 4:
                logging.info(f'{name}({param.data.shape}): {param.data[0][0][0][:preview_ele_num]}')
            else:
                logging.info(f'{name}({param.data.shape}): {param.data[0][0][0][0][:preview_ele_num]}')

    v16mm.eval()
    logging.info(f'Loading and transforming image from {image_path}')
    input = helper.preprocess_image(image_path)
    logging.info('Image ready')
    logging.info('Running inference')
    output = v16mm(input)
    logging.info('Done')
    logging.info(f'Raw output: {output}')
    y_pred = torch.argmax(output, dim=1)
    logging.info(f'y_pred: {y_pred}')


if __name__ == '__main__':
    main()
