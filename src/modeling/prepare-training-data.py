from typing import Any, Dict
from PIL import Image

import argparse
import datetime as dt
import helper
import json
import logging
import os
import random
import shutil
import sys
import threading
import time
import torchvision


curr_dir = os.path.dirname(os.path.abspath(__file__))


def apply_transform_and_save(
    source_dir: str, image_filename: str, image_ext: str,
    target_dir: str, variant_no: int, transforms: torchvision.transforms.Compose
):

    # Load the image
    image_path = os.path.join(source_dir, image_filename)
    image = Image.open(image_path)
    try:
        transformed_image = transforms(image)
    except Exception as ex:
        logging.error(f'Error transforming [{image_path}]: {ex}')
        return
    finally:
        image.close()

    # Save the transformed image to the target directory
    target_path = os.path.join(target_dir, image_filename)
    from torchvision.utils import save_image
    save_image(
        transformed_image,
        target_path + f'_{variant_no}.{image_ext}'
    )


def prepare_files(
        input_dir: str,
        train_dir: str, val_dir: str, image_ext: str,
        split_ratio: float, synthetic_multiplier: int, seed: int = 97381
) -> None:
    if os.path.exists(train_dir):
        logging.warning(
            f'[{train_dir}] exists '
            f'(file count: {len(os.listdir(train_dir)):,}), it will be removed'
        )
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)
    if os.path.exists(val_dir):
        logging.warning(
            f'[{val_dir}] exists '
            f'(file count: {len(os.listdir(val_dir)):,}), it will be removed'
        )
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)

    start_ts = time.time()
    files = os.listdir(input_dir)

    random.seed(seed)

    num_files = len(files)
    logging.info(f'{input_dir} | {num_files:,} samples found, will prepare {
                 num_files * synthetic_multiplier:,} synthetic samples')
    num_files_dir_1 = int(num_files * split_ratio)
    random.shuffle(files)

    for i, file in enumerate(files):
        if (i+1) % 100 == 0 or (i+1) == num_files:
            eta = start_ts + (time.time() - start_ts) * (num_files / (i+1))
            total_sec = eta - start_ts
            logging.info(
                f"{input_dir} | Processing {(i+1) * synthetic_multiplier:,}/{num_files * synthetic_multiplier:,}\tsamples ({int((i+1) / num_files * 100):2d}%), "
                f"ETA: {dt.datetime.fromtimestamp(eta).astimezone().strftime('%F %T')}"
                f'({total_sec/3600:.1f} hrs)'
                f', speed: {int(num_files * synthetic_multiplier / total_sec)} samples / sec'
            )
        for j in range(synthetic_multiplier):
            if i < num_files_dir_1:
                apply_transform_and_save(
                    input_dir, file, image_ext, train_dir, j, helper.train_transforms)
            else:
                apply_transform_and_save(
                    input_dir, file, image_ext, val_dir, j, helper.train_transforms)

    logging.info(
        f'{input_dir} | Done. '
        f'[{train_dir}] file count: {len(os.listdir(train_dir)):,}, '
        f'[{val_dir}] file count: {len(os.listdir(val_dir)):,}'
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d | %(levelname)7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-path', '-c', dest='config-path', required=True,
                    help='Config file path')
    ap.add_argument(
        '--split-ratio', '-r', help='Ratio of the training set',
        dest='split-ratio', type=float, default='0.9'
    )
    # Pass JPG is harddisk I/O is likely to be the bottleneck;
    # pass BMP is CPU is likely to be the bottleneck
    ap.add_argument(
        '--image-extension', '-e', dest='image-extension', default='jpg',
        help='Extension of images without the preceding dot.'
    )
    ap.add_argument(
        '--synthetic-multiplier', '-m',
        help=(
            'Number of raw samples times synthetic-multiplier will be the '
            'count of training+validation samples'
        ),
        dest='synthetic-multiplier', type=float, default='2'
    )
    args = vars(ap.parse_args())

    config = Dict[str, Any]
    with open(args['config-path']) as j:
        config = json.load(j)

    random_seed = 16888
    helper.init_transforms((
        config['model']['input_image_size']['height'],
        config['model']['input_image_size']['width']
    ))
    threads = []
    for cat in range(config['model']['num_classes']):
        cat = str(cat)
        input_dir = os.path.join(config["dataset"]['raw'], cat)
        training_dir = os.path.join(config["dataset"]['training'], cat)
        validation_dir = os.path.join(config["dataset"]['validation'], cat)
        thread = threading.Thread(target=prepare_files, args=(
            input_dir, training_dir, validation_dir, args['image-extension'],
            args['split-ratio'], int(args['synthetic-multiplier']), random_seed
        ))
        thread.start()
        threads.append(thread)
        time.sleep(5)

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
