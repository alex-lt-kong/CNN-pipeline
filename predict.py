from flask import Flask, request, Response
from threading import Thread, Lock


import logging
import numpy as np
import os
import sys
import tensorflow as tf
import time
import utils
import signal
import subprocess
import waitress
import shutil
import zmq
  

app = Flask(__name__)
prediction_interval = 3600
ev_flag = True
image_queue_mutex = Lock()
image_kinda_queue = []
image_kinda_queue_min_len = 12
image_kinda_queue_max_len = image_kinda_queue_min_len * 2

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
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
            )
    else:
        raise RuntimeError('How come?')
    return


def predict_frames(model, img_path, img_size):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    return model.predict(img_array)


def zeromq_thread() -> None:

    context = zmq.Context()

    #  Socket to talk to server
    print("Connecting to publisher endpoint")
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:4242")
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    print("Connected to endpoint")


    while ev_flag:
        #  Get the reply.
        message = socket.recv()
        image_queue_mutex.acquire()
        while len(image_kinda_queue) >= image_kinda_queue_max_len:
            image_kinda_queue.pop(0)
        image_kinda_queue.append(message)
        image_queue_mutex.release()
        #print(f"Image received, size {len(message)} bytes")
        #with open("/tmp/test.jpg", "wb") as binary_file:
        #    # Write bytes to file
        #    binary_file.write(message)


def prediction_thread() -> None:
    logging.info('prediction_thread() started')
    global prediction_interval, ev_flag
    settings = utils.read_config_file()
    img_path = settings['prediction']['input_file_path']
    sys.path.insert(1, settings['model']['path'])
    import definition

    model = tf.keras.models.load_model(settings['model']['save_to']['model'])
    while ev_flag:
        logging.info("Iterating prediction loop...")
        image_queue_mutex.acquire()
        if len(image_kinda_queue) < image_kinda_queue_min_len:
            logging.warn(
                f'len(image_kinda_queue): {len(image_kinda_queue)}, '
                'waiting for more frames...'
            )
            image_queue_mutex.release()
            time.sleep(5)
            continue

        with open(img_path, "wb") as binary_file:
            binary_file.write(image_kinda_queue[3])
        
        prediction = predict_frames(
            model, img_path, definition.target_image_size
        )
        logging.info(f'prediction: {prediction}')
        if prediction == 1:
            logging.info('Target detected, preparing context frames')
            for i in range(image_kinda_queue_min_len):
                with open(f'/tmp/frame{i}.jpg', "wb") as binary_file:
                    binary_file.write(image_kinda_queue[i])
            image_queue_mutex.release()
            logging.info('Calling downstream program...')
            result = subprocess.run(
                [settings['prediction']['on_detected']],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            logging.info(f'stdout: {result.stdout}')
            logging.info(f'stderr: {result.stderr}')
            
            count = 0
            while count < 60 * 2 and ev_flag:
                count += 1
                time.sleep(0.5)
        else:
            image_queue_mutex.release()

        count = 0
        while count < prediction_interval and ev_flag:
            count += 1
            time.sleep(1)
    logging.info('prediction thread exited')

def main():
    utils.initialize_logger()
    utils.set_environment_vars()
    limit_gpu_memory_usage()

    #signal.signal(signal.SIGINT, signal_handler)
    th_pred = Thread(target=prediction_thread)
    th_pred.start()
    th_zmq = Thread(target=zeromq_thread)
    th_zmq.start()
    
    waitress.serve(app, host='127.0.0.1', port='4386')



if __name__ == '__main__':
    main()
