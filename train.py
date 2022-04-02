from subprocess import Popen, PIPE
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers

import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
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


def initialize_logger():
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s | %(levelname)6s | %(message)s')
  handler.setFormatter(formatter)
  logger.addHandler(handler)


def check_gpu():
  process = Popen(['nvidia-smi'], stdout=PIPE)
  (output, err) = process.communicate()
  exit_code = process.wait()
  logging.info(output.decode('utf8'))
  logging.info(tf.config.list_physical_devices('GPU'))


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
        logging.info(f'{fpath} seems INvalid')
        #os.remove(fpath)
      else:
        logging.debug(f'{fpath} seems valid')

  logging.info(f"Deleted {num_skipped} images")


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


def preview_samples(dest_dir, train_ds, batch_size):

  for images, labels in train_ds.take(1):
    for i in range(len(images)):
      label_dir = os.path.join(dest_dir, str(labels[i]))
      if os.path.isdir(label_dir) is False:
        os.mkdir(label_dir)

      tf.keras.utils.save_img(
        os.path.join(label_dir, f'{i}.jpg'),
        images[i].numpy().astype("uint8")
      )

   # print(type(images[0]))

def make_model(input_shape, num_classes):

    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1)
      ]
    )

    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
      x = layers.Activation("relu")(x)
      x = layers.SeparableConv2D(size, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.Activation("relu")(x)
      x = layers.SeparableConv2D(size, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

      # Project residual
      residual = layers.Conv2D(size, 1, strides=2, padding="same")(
          previous_block_activation
      )
      x = layers.add([x, residual])  # Add back residual
      previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def get_confusion_matrix(model, val_ds, train_ds):
  y_pred = []
  y_true = []
  for x, y in val_ds:
      y_pred.extend(model.predict(x))
      y_true.extend(y.numpy())
  y_pred_cat = np.rint([item for sublist in y_pred for item in sublist]) 

  disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred_cat)
  disp.figure_.suptitle("Confusion Matrix")
  print(f"Confusion matrix:\n{disp.confusion_matrix}")


def main():
  settings = read_config_file()
  image_size = (
    settings['dataset']['image']['height'],
    settings['dataset']['image']['width']
  )
  batch_size = settings['model']['batch_size']
  
  initialize_logger()
  check_gpu()
  #remove_invalid_samples()
  train_ds, val_ds = prepare_dataset(image_size=image_size, batch_size=batch_size)
 # preview_samples(
 #   dest_dir=settings['dataset']['preview_save_to'],
 #   train_ds=train_ds,
 #   batch_size=batch_size
 # )
  
  train_ds = train_ds.prefetch(buffer_size=32)
  val_ds = val_ds.prefetch(buffer_size=32)

  model = make_model(input_shape=image_size + (3,), num_classes=2)
  keras.utils.plot_model(
    model,
    show_shapes=True,
    to_file=settings['model']['plot_save_to']
  )

  model.compile(
      optimizer=keras.optimizers.Adam(1e-3),
      loss="binary_crossentropy",
      metrics=["accuracy", 'AUC'],
  )

  model.fit(train_ds, epochs=settings['model']['epochs'], validation_data=val_ds)
  model.save(settings['model']['save_to'])  
  get_confusion_matrix(model=model, val_ds=val_ds, train_ds=train_ds)


if __name__ == '__main__':
  main()