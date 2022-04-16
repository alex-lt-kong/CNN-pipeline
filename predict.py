import glob
import json
import logging
import numpy as np
import os
import shutil
import subprocess
import tensorflow as tf
import utils



def video_to_frames(video_path, frames_dir):
  logging.info(f'Extracting frames from video [{video_path}] to [{frames_dir}]')

  tmp_dir = '/tmp/frames/'
  if os.path.isdir(tmp_dir):
    shutil.rmtree(tmp_dir)
  os.mkdir(tmp_dir)

  ffmpeg_cmd = [
    '/usr/local/bin/ffmpeg', '-i', video_path, '-r', '1', os.path.join(tmp_dir, '%05d.jpg')
  ]
  p = subprocess.Popen(args=ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
  stdout, stderr = p.communicate()

  if p.returncode != 0:
      # ffmpeg output will only be saved if debug mode is enabled;
      # otherwise there will be too much of it.
      logging.info(f'ffmpeg non-zero exist code: {p.returncode}')
      logging.debug(f'stdout: {stderr.decode("utf-8")}')


def predict_frames(model, frames_dir, img_size):

  predictions = []
  listing = glob.glob(frames_dir)
  print(f'{len(listing)},', end=' ')
  for file_path in listing:    
    if os.path.isfile(file_path) is False:
      continue
    img = tf.keras.utils.load_img(
      file_path, target_size=img_size
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    prediction_raw = model.predict(img_array)
    prediction = np.argmax(tf.nn.softmax(prediction_raw[0]))
    predictions.append(prediction)

  return predictions


def generate_final_prediction(predictions):
  predictions.sort(reverse=True) 
  if len(predictions) == 0:
    print('Error')
    return
  print(f'{predictions[0:25]}, ({100 * sum(predictions)/len(predictions):.2f}%)')


def main():

  settings = utils.read_config_file()
  utils.initialize_logger(settings['misc']['log_path'])
  videos = []

  listing = glob.glob(settings['prediction']['source'])
  listing.sort(reverse=True) 
  for file_path in listing:
    if os.path.isfile(file_path):
      videos.append(file_path)

  try:
    with open(settings['prediction']['results']) as json_file:
      prediction_records = json.load(json_file)
  except Exception as ex:
    logging.error(f'Unable to read from [{settings["prediction"]["results"]}]. Reason: {ex}')
    prediction_records = {}
    print(prediction_records)
  model = tf.keras.models.load_model(settings['model']['save_to']['model'])
  for video in videos:
    if video in prediction_records:
      print(f'{video} in predictions')
    else:
      print(f'{video} NOT in predictions', end='')
      video_to_frames(video, settings["prediction"]["frames_dir"])
      predictions = predict_frames(
        model,
        settings["prediction"]["frames_dir"] + '*.jpg',
        (settings['dataset']['image']['height'], settings['dataset']['image']['width'])
      )
      generate_final_prediction(predictions)

  

if __name__ == '__main__':
  main()