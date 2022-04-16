from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import classification_report, plot_confusion_matrix
from tensorflow import keras

import seaborn as sn
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import utils
import pandas as pd


def get_predictions(dataset, model, misclassified_dest):
  # X = np.concatenate([X for X, y in val_ds], axis=0)  
  # y_true = np.concatenate([y for x, y in val_ds], axis=0)
  # cannot use this approach as it loads all the samples into memory, which 
  # gets exhausted very fast.
  
  y_true = []
  y_pred = []
  y_pred_cat = []
  count = 0

  for x, y in dataset:
    # Note that here x and y do not come one after another--they come in batches
    # whose size is defined by the batch_size parameter passed to the prepare_dataset
    # method.
    count += 1
    logging.info(f'Predicting {count}-th sample')
    y_true.extend(y.numpy().tolist())
    predictions = model.predict(x)
    
    for i in range(len(predictions)):
      y_pred.append(np.argmax(tf.nn.softmax(predictions[i])))
      y_pred_cat.append(0 if y_pred[-1] < 0.5 else 1)

      if y_pred_cat[-1] != y[i]:
        label_dir = os.path.join(misclassified_dest, str(y[i].numpy()))
        if os.path.isdir(label_dir) is False:
          os.mkdir(label_dir)
        tf.keras.utils.save_img(
          os.path.join(label_dir, f'{count}.jpg'), x[i].numpy().astype("uint8")
        )

  return np.array(y_true), np.array(y_pred), np.array(y_pred_cat)


def plot_confusion_matrix(y_true, y_pred_cat, classes, path):

  cm = metrics.confusion_matrix(y_true, y_pred_cat)
  
  df_cm = pd.DataFrame(cm, index=classes, columns=classes)

  plt.figure(figsize = (16/2, 9/2))
  plt.rcParams.update({'font.size': 15})
  sn.heatmap(df_cm, cmap="YlGnBu", annot=True, fmt='g')  
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.savefig(path, bbox_inches='tight')


def plot_roc_curve(y_true, y_pred, path):
  fpr, tpr, thresholds = roc_curve(y_true, y_pred)
  plt.figure(figsize = (16/2, 9/2))
  plt.rcParams.update({'font.size': 15})
  plt.plot(fpr, tpr) 
  plt.axis([0, 1, 0, 1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.savefig(path, bbox_inches='tight')


def main():
  settings = utils.read_config_file()
  image_size = (
    settings['dataset']['image']['height'], settings['dataset']['image']['width']
  )
  batch_size = settings['model']['batch_size']
    
  utils.initialize_logger(settings['misc']['log_path'])
  train_ds, val_ds = utils.prepare_dataset(settings['dataset']['path'], image_size=image_size, batch_size=batch_size)
  model = tf.keras.models.load_model(settings['model']['save_to']['model'])
  y_true, y_pred, y_pred_cat = get_predictions(
    dataset=val_ds, model=model, misclassified_dest=settings['diagnostics']['misclassified']
  )
  print(y_true)
  print(y_pred_cat)
  classes = ['Not Detected', 'Detected']

  plot_roc_curve(y_true, y_pred, settings['diagnostics']['roc'])
  plot_confusion_matrix(y_true, y_pred_cat, classes, settings['diagnostics']['confusion_matrix'])
  with open(settings['diagnostics']['report'], 'w') as f:
    f.write(classification_report(y_true, y_pred_cat, target_names=classes))

if __name__ == '__main__':
  main()