from flask import Flask, request, Response
from threading import Thread, Lock

import definition
import datetime as dt
import logging
import os
import sys
import tensorflow as tf
import time
import utils
import signal
import sqlite3
import subprocess
import waitress
import zmq
  

app = Flask(__name__)
curr_dir = os.path.dirname(os.path.realpath(__file__))
prediction_interval = 600
ev_flag = True
image_queue_mutex = Lock()
image_kinda_queue = []
image_kinda_queue_min_len = 12
image_kinda_queue_max_len = image_kinda_queue_min_len * 2
db_path = os.path.join(curr_dir, 'predict.sqlite')


def signal_handler(signum, frame) -> None:
    global ev_flag
    print('Signal handler called with signal', signum)
    ev_flag = False
    signal.signal(signum, signal.SIG_DFL)


@app.route("/")
def set_prediction_interval() -> Response:
    global prediction_interval
    try:
        prediction_interval = float(request.args.get('prediction_interval', '1'))
    except Exception as ex:
        return Response(f"{ex}", status=400)
    return Response(f'prediction_interval: {prediction_interval} sec', status=200)


def prepare_database() -> None:

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            prediction REAL
    )''')
    conn.commit()
    
    cutoff_date = dt.datetime.now() - dt.timedelta(days=15)

    cur.execute(
        "DELETE FROM prediction_results WHERE timestamp < ?", (cutoff_date,)
    )
    conn.commit()


def config_tf() -> None:
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
    tf.compat.v1.disable_eager_execution()
    return


def predict_frames(model, img_path, img_size) -> float:

    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    pred = model.predict(img_array, steps=1)[0][0].item()
    assert isinstance(pred, float)
    return pred


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

    logging.info('zeromq_thread() exited gracefully')


def insert_prediction_to_db(conn: sqlite3.Connection, pred: float) -> None:

    cur = conn.cursor()
    sql = "INSERT INTO prediction_results(timestamp, prediction) VALUES (?, ?)"
    cur.execute(sql, (dt.datetime.now().isoformat(), pred))
    conn.commit()


def prediction_thread() -> None:
    logging.info('prediction_thread() started')
    global prediction_interval, ev_flag
    settings = utils.read_config_file()
    img_path = settings['prediction']['input_file_path']

    conn = sqlite3.connect(db_path)

    model = tf.keras.models.load_model(settings['model']['model'])
    while ev_flag:
        logging.info("Iterating prediction loop...")
        image_queue_mutex.acquire()
        if len(image_kinda_queue) < image_kinda_queue_min_len:
            logging.warning(
                f'len(image_kinda_queue): {len(image_kinda_queue)}, '
                'waiting for more frames...'
            )
            image_queue_mutex.release()
            time.sleep(5)
            continue

        with open(img_path, "wb") as binary_file:
            binary_file.write(image_kinda_queue[3])
        
        pred = predict_frames(model, img_path, definition.target_image_size)
        insert_prediction_to_db(conn, pred)
        if pred > 0.5:
            logging.warning('Target detected, preparing context frames')
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
            while count < 90 * 2 and ev_flag:
                count += 1
                time.sleep(0.5)
        else:
            image_queue_mutex.release()

        count = 0
        while count < prediction_interval and ev_flag:
            count += 1
            time.sleep(1)
    logging.info('prediction_thread() exited gracefully')
    conn.close()


def waitress_thread() -> None:
   # server = 
    
    app.run(host='0.0.0.0', port=4386)

def main() -> None:
    utils.initialize_logger()
    utils.set_environment_vars()
    config_tf()
    prepare_database()

    signal.signal(signal.SIGINT, signal_handler)
    th_pred = Thread(target=prediction_thread)
    th_pred.start()
    th_zmq = Thread(target=zeromq_thread)
    th_zmq.start()
    #th_waitress = Thread(target=waitress_thread)
    #th_waitress.start()
    waitress.serve(app, host='127.0.0.1', port='4386')
    logging.info('main() exited gracefully')

if __name__ == '__main__':
    main()
