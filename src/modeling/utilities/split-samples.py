from typing import Any, Dict

import argparse
import json
import os
import random
import shutil

curr_dir = os.path.dirname(os.path.abspath(__file__))


def split_files(
        input_dir: str,
        output_dir_1: str, output_dir_2: str,
        split_ratio: float, seed: int = 97381
) -> None:

    if os.path.exists(output_dir_1):
        print(
            f'[{output_dir_1}] exists '
            f'(file count: {len(os.listdir(output_dir_1)):,}), it will be removed'
        )
        shutil.rmtree(output_dir_1)
    os.makedirs(output_dir_1)
    if os.path.exists(output_dir_2):
        print(
            f'[{output_dir_2}] exists '
            f'(file count: {len(os.listdir(output_dir_2)):,}), it will be removed'
        )
        shutil.rmtree(output_dir_2)
    os.makedirs(output_dir_2)

    files = os.listdir(input_dir)
    random.seed(seed)

    num_files = len(files)
    num_files_dir_1 = int(num_files * split_ratio)
    random.shuffle(files)

    for i, file in enumerate(files):
        src = os.path.join(input_dir, file)
        if i < num_files_dir_1:
            dst = os.path.join(output_dir_1, file)
        else:
            dst = os.path.join(output_dir_2, file)
        shutil.copy2(src, dst)

    print(
        f'Splitting files from [{input_dir}] to '
        f'[{output_dir_1}]/[{output_dir_2}] completed successfully.\n'
        f'[{output_dir_1}] file count: {len(os.listdir(output_dir_1)):,}\n'
        f'[{output_dir_2}] file count: {len(os.listdir(output_dir_2)):,}\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--split-ratio', '-r', help='Ratio of the training set',
        dest='split-ratio', type=float, default='0.9'
    )
    ap.add_argument(
        '--categories', '-c', required=True, dest='categories',
        help='Comma-separated list of categories, typically something like 0,1,2')
    args = vars(ap.parse_args())

    config = Dict[str, Any]
    with open(os.path.join(curr_dir, '..', '..', '..', 'config.json')) as j:
        config = json.load(j)

    random_seed = 16888
    # breakpoint()
    for cat in args['categories'].split(','):
        input_dir = os.path.join(config["dataset"]["raw"], cat)
        training_dir = os.path.join(config["dataset"]["training"], cat)
        validation_dir = os.path.join(config["dataset"]["validation"], cat)
        split_files(
            input_dir, training_dir, validation_dir,
            args['split-ratio'], random_seed
        )


if __name__ == '__main__':
    main()
