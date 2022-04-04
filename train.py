from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers

import utils
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
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
        logging.info(f'{fpath} seems INvalid')
        #os.remove(fpath)
      else:
        logging.debug(f'{fpath} seems valid')

  logging.info(f"Deleted {num_skipped} images")


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


def main():
  settings = utils.read_config_file()
  image_size = (
    settings['dataset']['image']['height'],
    settings['dataset']['image']['width']
  )
  batch_size = settings['model']['batch_size']
  
  utils.initialize_logger()
  utils.check_gpu()
  #remove_invalid_samples()
  train_ds, val_ds = utils.prepare_dataset(image_size=image_size, batch_size=batch_size)
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
  


if __name__ == '__main__':
  main()