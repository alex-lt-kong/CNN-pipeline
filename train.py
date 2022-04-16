from tensorflow import keras

import utils
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd

import shutil
import sys
import tensorflow as tf


def preview_samples(dest_dir, dataset, data_augmentation):

  count = {
    '0': 0,
    '1': 0
  }
  for images, labels in dataset:
    augmented_images = data_augmentation(images)
    for i in range(len(augmented_images)):
      label = str(labels[i].numpy())
      count[label] += 1
      label_dir = os.path.join(dest_dir, label)
      if os.path.isdir(label_dir) is False:
        os.mkdir(label_dir)

      tf.keras.utils.save_img(
        os.path.join(label_dir, f'{count[label]}.jpg'), augmented_images[i].numpy().astype("uint8")
      )

    enough_sample = 0
    for key in count.keys():
      if count[key] > 5:
        enough_sample += 1
    if enough_sample >= len(count.keys()):
      break


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

  sys.path.insert(1, settings['model']['path'])
  import definition

  func = definition.data_augmentation()
  preview_samples(
    dest_dir=settings['dataset']['preview_save_to'],
    dataset=train_ds,
    data_augmentation=func)
  
  # This buffer_size is for hard drive IO only, not the number of images send to the model in one go.
  train_ds = train_ds.prefetch(buffer_size=32)
  val_ds = val_ds.prefetch(buffer_size=32)

  input_shape = image_size + (3,)  # + (1,) for grayscale, + (3,) for rgb
  model = definition.make_model(input_shape=input_shape, data_augmentation=func, num_classes=2)
  
  #keras.utils.plot_model(
  #  model,
  #  show_shapes=True,
  #  to_file=settings['model']['save_to']['model_plot']
  #)

  #with open(settings['model']['save_to']['summary'], 'w') as f:    
  #  model.summary(print_fn=lambda x: f.write(x + '\n'))

  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


  history = model.fit(train_ds, epochs=settings['model']['epochs'], validation_data=val_ds)
  # epochs: an epoch is an iteration over the entire x and y data provided
  # (unless the steps_per_epoch flag is set to something other than None)
  if os.path.isdir(settings['model']['save_to']['model']):
    shutil.rmtree(settings['model']['save_to']['model'])
  tf.keras.models.save_model(model=model, filepath=settings['model']['save_to']['model'])
  df = pd.DataFrame(data=history.history)
  df['accuracy_ma'] = df['accuracy'].rolling(window=5).mean()
  df['val_accuracy_ma'] = df['val_accuracy'].rolling(window=5).mean()
  plt.figure(figsize = (16, 9/2))
  #fig = df[['accuracy', 'accuracy_ma', 'val_accuracy', 'val_accuracy_ma']].plot(kind='line', figsize=(16, 9/2), fontsize=12).get_figure()
  plt.plot(df['accuracy'],        linewidth=0.8, label="Accuracy",               color='C0', alpha=0.8)
  plt.plot(df['accuracy_ma'],     linewidth=1.2, label="Accuracy MA",            color='C0', alpha=1.0)
  plt.plot(df['val_accuracy'],    linewidth=0.8, label="Validation Accuracy",    color='C1', alpha=0.8)
  plt.plot(df['val_accuracy_ma'], linewidth=1.2, label="Validation Accuracy MA", color='C1', alpha=1.0)
  plt.legend()
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.savefig(settings['model']['save_to']['history_plot'], bbox_inches='tight')

  df.to_csv(settings['model']['save_to']['history'])
  

if __name__ == '__main__':
  main()