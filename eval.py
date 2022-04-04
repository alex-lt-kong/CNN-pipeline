from sklearn import metrics
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import numpy as np
import utils


def get_confusion_matrix(model, val_ds, train_ds):
  X = np.concatenate([X for X, y in train_ds], axis=0)
  y_true = np.concatenate([y for x, y in train_ds], axis=0)
  y_pred = model.predict(X).ravel()
  y_pred_cat = np.where(y_pred > 0.1, 1,0)
  # model.predict(X) is different between binary and mutli-class Classification
  # In a word, binary classification does not need argmax()
  print(y_true)
  print(y_pred_cat)
  cm = metrics.confusion_matrix(y_true, y_pred_cat)
  print(cm)
  print(classification_report(y_true, y_pred_cat, target_names=['0','1']))
  return


settings = utils.read_config_file()
image_size = (
  settings['dataset']['image']['height'],
  settings['dataset']['image']['width']
)
batch_size = settings['model']['batch_size']
  
utils.initialize_logger()
train_ds, val_ds = utils.prepare_dataset(image_size=image_size, batch_size=batch_size)
model = keras.models.load_model(settings['model']['save_to'])

get_confusion_matrix(model=model, val_ds=val_ds, train_ds=train_ds)
