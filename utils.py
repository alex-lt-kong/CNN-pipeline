from subprocess import Popen, PIPE

import argparse
import json
import logging
import os
import sys
import tensorflow as tf


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


def initialize_logger():
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s | %(levelname)6s | %(message)s')
  handler.setFormatter(formatter)
  logger.addHandler(handler)

def prepare_dataset(image_size, batch_size):

  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data-in",
    validation_split=0.2,
    subset="training",
    seed=2333,
    image_size=image_size,
    batch_size=batch_size,
  )
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data-in",
    validation_split=0.2,
    subset="validation",
    seed=2333,
    image_size=image_size,
    batch_size=batch_size,
  )
  return train_ds, val_ds