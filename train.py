from tensorflow import keras
from tensorflow.keras import layers

import utils
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil
import tensorflow as tf


def preview_samples(dest_dir, dataset):

  for images, labels in dataset.take(1):
    for i in range(len(images)):
      label_dir = os.path.join(dest_dir, str(labels[i].numpy()))
      if os.path.isdir(label_dir) is False:
        os.mkdir(label_dir)

      tf.keras.utils.save_img(
        os.path.join(label_dir, f'{i}.jpg'), images[i].numpy().astype("uint8")
      )


def make_model(input_shape, num_classes, dropout=0.5):

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
    x = layers.Rescaling(1.0 / 255)(x) # Shrink RGB channel values from [0, 255] range to [0, 1] range.
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
      x = layers.Dropout(dropout)(x)
      # Dropout added per https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/

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

    x = layers.Dropout(dropout)(x) # Original dropout layer in Keras example.
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def main():
  settings = utils.read_config_file()
  image_size = (
    settings['dataset']['image']['height'],
    settings['dataset']['image']['width']
  )
  batch_size = settings['model']['batch_size']
  
  utils.initialize_logger(settings['misc']['log_path'])
  logging.info(settings)
  utils.check_gpu()
  utils.remove_invalid_samples(settings['dataset']['path'])
  train_ds, val_ds = utils.prepare_dataset(settings['dataset']['path'], image_size=image_size, batch_size=batch_size)
  preview_samples(dest_dir=settings['dataset']['preview_save_to'], dataset=train_ds)
  
  # This buffer_size is for hard drive IO only, not the number of images send to the model in one go.
  train_ds = train_ds.prefetch(buffer_size=32)
  val_ds = val_ds.prefetch(buffer_size=32)

  model = make_model(input_shape=image_size + (3,), num_classes=2, dropout=settings['model']['dropout'])
  keras.utils.plot_model(
    model,
    show_shapes=True,
    to_file=settings['model']['save_to']['model_plot']
  )

  with open(settings['model']['save_to']['summary'], 'w') as f:    
    model.summary(print_fn=lambda x: f.write(x + '\n'))

  model.compile(
      optimizer=keras.optimizers.Adam(1e-3),
      loss="binary_crossentropy",
      metrics=["accuracy"],
  )

  history = model.fit(train_ds, epochs=settings['model']['epochs'], validation_data=val_ds)
  if os.path.isdir(settings['model']['save_to']['model']):
    shutil.rmtree(settings['model']['save_to']['model'])
  model.save(settings['model']['save_to']['model'])
  df = pd.DataFrame(data=history.history)
  fig = df[['accuracy', 'val_accuracy']].plot(kind='line', figsize=(16, 9), fontsize=18).get_figure()
  fig.savefig(settings['model']['save_to']['history_plot'])

  df.to_csv(settings['model']['save_to']['history'])
  

if __name__ == '__main__':
  main()