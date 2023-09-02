import argparse
import helper
import logging
import json
import model
import os
import sys
import torch

curr_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-path', '-p', dest='image-path', required=True,
                    help='Path of image to be inferenced')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d | %(levelname)7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    args = vars(ap.parse_args())
    image_path = args['image-path']

    with open(os.path.join(curr_dir, '..', 'config.json')) as j:
        settings = json.load(j)
    v16mm = model.VGG16MinusMinus(2)
    logging.info(f'Loading parameters from {settings["model"]["parameters"]}')
    v16mm.load_state_dict(torch.load(settings['model']['parameters']))

    logging.info('Parameters loaded')

    preview_ele_num = 5
    layer_count = 0
    for name, param in v16mm.named_parameters():
        layer_count += 1
        if layer_count % 4 != 0:
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
    tensor_image = helper.preprocess_image(image_path)

    logging.info(
        f'Image tensor ready, tensor shape: {tensor_image.shape}, '
        f'sample vlaues:'
    )
    for i in range(0, tensor_image.shape[1]):
        logging.info(tensor_image[0][i][i][:preview_ele_num])

    tensor_image = tensor_image.to(device)
    logging.info('Running inference')
    output = v16mm(tensor_image)
    logging.info('Done')
    logging.info(f'Raw output: {output}')
    y_pred = torch.argmax(output, dim=1)
    logging.info(f'y_pred: {y_pred}')


if __name__ == '__main__':
    main()
