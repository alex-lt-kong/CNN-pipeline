from tensorflow import keras

import utils
import logging
import os
import pandas as pd
import shutil
import sys
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

  sys.path.insert(1, settings['model']['path'])
  import definition
  model = definition.make_model(input_shape=image_size + (3,), num_classes=2)
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
  # epochs: an epoch is an iteration over the entire x and y data provided
  # (unless the steps_per_epoch flag is set to something other than None)
  if os.path.isdir(settings['model']['save_to']['model']):
    shutil.rmtree(settings['model']['save_to']['model'])
  model.save(settings['model']['save_to']['model'])
  df = pd.DataFrame(data=history.history)
  fig = df[['accuracy', 'val_accuracy']].plot(kind='line', figsize=(16, 9), fontsize=18).get_figure()
  fig.savefig(settings['model']['save_to']['history_plot'])

  df.to_csv(settings['model']['save_to']['history'])
  

if __name__ == '__main__':
  main()