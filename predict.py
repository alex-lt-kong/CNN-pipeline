from flask import Flask, request, Response
from threading import Thread

import numpy as np
import sys
import tensorflow as tf
import time
import utils
import signal
import subprocess
import waitress
import shutil

app = Flask(__name__)
prediction_interval = 60
ev_flag = True

@app.route("/")
def set_prediction_interval():
    global prediction_interval
    try:
        prediction_interval = float(request.args.get('prediction_interval', '1'))
    except Exception as ex:
        return Response(f"{ex}", status=400)
    return Response(f'prediction_interval: {prediction_interval} sec', status=200)

def signal_handler(signum, frame):
    global ev_flag
    print('Signal handler called with signal', signum)
    ev_flag = False


def limit_gpu_memory_usage() -> None:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    else:
        raise RuntimeError('How come?')


def predict_frames(model, img_path, img_size):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    prediction_raw = model.predict(img_array)
    prediction = np.argmax(tf.nn.softmax(prediction_raw[0]))

    return prediction


def prediction_thread() -> None:
    global prediction_interval, ev_flag
    settings = utils.read_config_file()
    img_path = settings['prediction']['input_file_path']
    sys.path.insert(1, settings['model']['path'])
    
    import definition    

    model = tf.keras.models.load_model(settings['model']['save_to']['model'])
    while ev_flag:
        shutil.copyfile(img_path, '/tmp/corr-sample.jpg')
        prediction = predict_frames(
            model, '/tmp/corr-sample.jpg',definition.target_image_size
        )
        if prediction == 1:
            subprocess.run([settings['prediction']['on_detected']])
            while count < 60 and ev_flag:
                count += 1
                time.sleep(1)
        count = 0
        while count < prediction_interval and ev_flag:
            count += 1
            time.sleep(1)


def main():
    utils.set_environment_vars()
    limit_gpu_memory_usage()
    signal.signal(signal.SIGINT, signal_handler)
    utils.initialize_logger()
    th = Thread(target=prediction_thread)
    th.start()
    
    waitress.serve(app, host='127.0.0.1', port='4386')



if __name__ == '__main__':
    main()