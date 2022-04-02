from tensorflow import keras
from tensorflow.keras import layers
from subprocess import Popen, PIPE


import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

image_size = (144, 256)
batch_size = 8

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


def prepare_dataset():

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


def preview_samples():
  count = 0
  for images, labels in train_ds.take(1):
    for i in range(batch_size):
      count += 1
      tf.keras.utils.save_img(
        f'data-out/sample-preview/{labels[i]}/{count}.jpg',
        images[i].numpy().astype("uint8")
      )

   # print(type(images[0]))

def make_model(input_shape, num_classes):
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




initialize_logger()
check_gpu()
remove_invalid_samples()
train_ds, val_ds = prepare_dataset()
preview_samples()

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
  ]
)

if os.path.exists('./model.dat'):
  model = keras.models.load_model('./model.dat')
else:
  model = make_model(input_shape=image_size + (3,), num_classes=2)
  keras.utils.plot_model(model, show_shapes=True, to_file='model.png')

  epochs = 100

  callbacks = [
    #  keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
  ]
  model.compile(
      optimizer=keras.optimizers.Adam(1e-3),
      loss="binary_crossentropy",
      metrics=["accuracy"],
  )

  model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds)
  model.save('./model.dat')  


y_pred = []
y_true = []
for x, y in val_ds:
    y_pred.extend(model.predict(x))
    y_true.extend(y.numpy())
for x, y in train_ds:
    y_pred.extend(model.predict(x))
    y_true.extend(y.numpy())
y_pred_cat = np.rint([item for sublist in y_pred for item in sublist])

from sklearn import metrics

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred_cat)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")