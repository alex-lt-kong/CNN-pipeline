from PIL import Image
from typing import Tuple

import os


def resize_images(src_dir: str, dst_dir: str, dst_size: Tuple[int, int]) -> None:

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for filename in os.listdir(src_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(src_dir, filename)
            image = Image.open(image_path)

            if image.size[0] == dst_size[0] and image.size[1] == dst_size[1]:
                print(f'{filename}\'s size is already the same as dst_size, skipping')
                continue
            resized_image = image.resize(dst_size)
            output_path = os.path.join(dst_dir, filename)
            resized_image.save(output_path)

            print(f'Resized [{filename}] and saved to [{output_path}]')


src_dir = '/mnt/models/vgg16-based-pipeline/data/input/1_src/'
dst_dir = '/mnt/models/vgg16-based-pipeline/data/input/1/'
size = (426, 224)  # size in in (w, h)

resize_images(src_dir, dst_dir, size)
