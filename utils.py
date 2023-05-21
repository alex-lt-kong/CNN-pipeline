from subprocess import Popen, PIPE
from typing import Tuple, Dict, Any

import argparse
import json
import logging
import os
import sys
import tensorflow as tf


def set_environment_vars() -> None:
    os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8/targets/x86_64-linux/lib/"
    #os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib/"


def remove_invalid_samples(sample_path):
    num_skipped = 0
    for folder_name in ("0", "1"):
        logging.info(f'Now checking samples in {folder_name} directory')
        folder_path = os.path.join(sample_path, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            is_jpeg_ok = False
            try:
                fobj = open(fpath, "rb")
                is_jpeg_ok = tf.compat.as_bytes("JFIF") in fobj.peek(20)
                is_jpeg_ok = (is_jpeg_ok or tf.compat.as_bytes("Lavc") in fobj.peek(20))
                check_chars = fobj.read()[-2:]
                is_jpeg_ok = (is_jpeg_ok and check_chars == b'\xff\xd9')
            except Exception as ex:
                is_jpeg_ok = False
                logging.error(f'{ex}')
            finally:
                fobj.close()

            if not is_jpeg_ok:
                num_skipped += 1                
                os.remove(fpath)
                logging.info(f'{fpath} seems INvalid and is removed from filesystem')
            else:
                logging.debug(f'{fpath} seems valid')

    logging.info(f"Deleted {num_skipped} images")


def read_config_file() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', dest='config', required=True,
        help='the path of the JSON format configuration file to be used by the model'        
    )
    args = vars(ap.parse_args())
    config_path = args['config']
    if os.path.isfile(config_path) is False:
        raise FileNotFoundError(f'File [{config_path}] not found')
    with open(config_path, 'r') as json_file:
        json_str = json_file.read()
        settings = json.loads(json_str)
    return settings


def check_gpu() -> None:
    process = Popen(['nvidia-smi'], stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    logging.info(output.decode('utf8'))
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logging.info(gpus)


def initialize_logger() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)


def prepare_dataset(
    sample_path, image_size, batch_size, seed=168
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        sample_path,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        #color_mode='grayscale'
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        sample_path,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        #color_mode='grayscale'
    )

    # This buffer_size is for hard drive IO only, 
    # not the number of images sent to the model in one go.
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    return train_ds, val_ds