from subprocess import Popen, PIPE

import argparse
import json
import logging
import os
import sys
import tensorflow as tf


def remove_invalid_samples():
  num_skipped = 0
  for folder_name in ("0", "1"):
    folder_path = os.path.join("data-in", folder_name)
    for fname in os.listdir(folder_path):
      fpath = os.path.join(folder_path, fname)
      try:
        fobj = open(fpath, "rb")
        is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(20)
        is_jfif = (is_jfif or tf.compat.as_bytes("Lavc") in fobj.peek(20))
      except Exception as ex:
        logging.error(f'{ex}')
      finally:
        fobj.close()

      if not is_jfif:
        num_skipped += 1        
        os.remove(fpath)
        logging.info(f'{fpath} seems INvalid and is removed from filesystem')
      else:
        logging.debug(f'{fpath} seems valid')

  logging.info(f"Deleted {num_skipped} images")


def read_config_file():
  ap = argparse.ArgumentParser()
  ap.add_argument(
    '--config', 
    dest='config',
    help='the path of the JSON format configuration file to be used by the model',
    default='./config.json'
  )
  args = vars(ap.parse_args())
  config_path = args['config']
  if os.path.isfile(config_path) is False:
    raise FileNotFoundError(f'File [{config_path}] not found')
  with open(config_path, 'r') as json_file:
    json_str = json_file.read()
    settings = json.loads(json_str)
  return settings


def check_gpu():
  process = Popen(['nvidia-smi'], stdout=PIPE)
  (output, err) = process.communicate()
  exit_code = process.wait()
  logging.info(output.decode('utf8'))
  logging.info(tf.config.list_physical_devices('GPU'))


def initialize_logger(log_path: str):
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
    
  formatter = logging.Formatter('%(asctime)s | %(levelname)6s | %(message)s')
  file_handler = logging.FileHandler(log_path)
  file_handler.setLevel(logging.DEBUG)
  file_handler.setFormatter(formatter)
  if (logger.hasHandlers()):
    logger.handlers.clear()
  logger.addHandler(file_handler)


def prepare_dataset(image_size, batch_size, seed=2333):

  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data-in",
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
  )
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data-in",
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
  )
  return train_ds, val_ds