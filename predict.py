from flask import Flask, request, Response
from threading import Thread, Lock
from typing import List, Any, Dict
from PIL import Image


import argparse
import datetime as dt
import io
import json
import logging
import model
import os
import time
#import utils
import signal
import sqlite3
import subprocess
import torch
import waitress
import sys
import zmq


app = Flask(__name__)
curr_dir = os.path.dirname(os.path.realpath(__file__))
prediction_interval = 600
ev_flag = True
image_queue_mutex = Lock()
image_queue: List[Any] = []
image_queue_min_len = 16
image_queue_max_len = image_queue_min_len * 2
db_path = os.path.join(curr_dir, 'predict.sqlite')


def signal_handler(signum: int, frame: Any) -> None:
    global ev_flag
    print('Signal handler called with signal', signum)
    ev_flag = False
    signal.signal(signum, signal.SIG_DFL)


@app.route("/")
def set_prediction_interval() -> Response:
    global prediction_interval
    try:
        prediction_interval = float(request.args.get('prediction_interval', '1'))
        logging.info(f'prediction_interval changed to {prediction_interval}')
    except Exception as ex:
        logging.exception('Unabled to set new prediction_interval')
        return Response(f"{ex}", status=400)
    return Response(f'prediction_interval: {prediction_interval} sec', status=200)


def prepare_database() -> None:

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS inference_records (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            model_output TEXT,
            prediction INTEGER,
            elapsed_time_ms REAL
    )''')
    conn.commit()

    cutoff_date = dt.datetime.now() - dt.timedelta(days=15)

    cur.execute(
        "DELETE FROM inference_records WHERE timestamp < ?", (cutoff_date,)
    )
    conn.commit()


def zeromq_thread() -> None:

    context = zmq.Context()
    url = "tcp://127.0.0.1:4241"
    #  Socket to talk to server
    logging.info(f"Connecting to publisher endpoint [{url}]")
    socket = context.socket(zmq.SUB)
    socket.connect(url)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    logging.info("Connected to endpoint")

    while ev_flag:
        #  Get the reply.
        message = socket.recv()
        image_queue_mutex.acquire()
        while len(image_queue) >= image_queue_max_len:
            image_queue.pop(0)
        image_queue.append(message)
        image_queue_mutex.release()

    logging.info('zeromq_thread() exited gracefully')


def insert_prediction_to_db(cur: sqlite3.Cursor, model_output: str,
                            prediction: int, elapsed_time_ms: float
) -> None:
    sql = """
        INSERT INTO inference_records(
            timestamp, model_output, prediction, elapsed_time_ms
        )
        VALUES (?, ?, ?, ?)
    """
    cur.execute(sql,
        (dt.datetime.now().isoformat(), model_output,
         prediction, elapsed_time_ms)
    )

def initialize_logger() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s | %(name)9s | %(levelname)8s | %(message)s'
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

def read_config_file() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', dest='config', required=True,
        help='the path of the JSON format configuration file to be used by the model'        
    )
    args = vars(ap.parse_args())
    config_path = args['config']
    if os.path.isfile(config_path) is False:
        raise FileNotFoundError(f'File [{config_path}] not found')
    with open(config_path, 'r') as json_file:
        json_str = json_file.read()
        settings = json.loads(json_str)
        assert isinstance(settings, Dict)
    return settings

def prediction_thread() -> None:
    logging.info('prediction_thread() started')
    global prediction_interval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = read_config_file()
    img_path = settings['prediction']['input_file_path']

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    v16mm = model.VGG16MinusMinus(2)
    v16mm.to(device)
    logging.info(f'Loading weights to model from: {settings["model"]["model"]}')
    v16mm.load_state_dict(torch.load(settings['model']['model']))
    logging.info('Weights loaded')
    iter_count = 0
    while ev_flag:
        iter_count += 1
        logging.info("Iterating prediction loop...")
        image_queue_mutex.acquire()
        if len(image_queue) < image_queue_min_len:
            logging.warning(
                f'len(image_queue): {len(image_queue)}, '
                'waiting for more frames...'
            )
            image_queue_mutex.release()
            count = 0
            while count < 5 and ev_flag:
                count += 1
                time.sleep(1)
            continue

        start_time = time.time()
        image = Image.open(io.BytesIO(image_queue[3]))
        input_data = v16mm.transforms(image).unsqueeze(0).to(device)
        # Classify the image using the model
        with torch.no_grad():
            output = v16mm(input_data)
            _, pred_tensor = torch.max(output.data, 1)
        elapsed_time = time.time() - start_time
        insert_prediction_to_db(cur, str(output), int(pred_tensor[0]),
                                round(elapsed_time * 1000.0, 1))
        
        if iter_count > 120:
            # commit() could be a very expensive operation
            # profiling shows it takes 1+ sec to complete            
            conn.commit()
            logging.info(f'SQLite commit()ed')
            iter_count = 0

        if pred_tensor[0] == 1:
            logging.warning('Target detected, preparing context frames')
            for i in range(image_queue_min_len):
                with open(f'/tmp/frame{i}.jpg', "wb") as binary_file:
                    binary_file.write(image_queue[i])
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
        while count < prediction_interval * 10.0 and ev_flag:
            count += 1
            time.sleep(0.1)
    conn.commit()
    logging.info(f'SQLite commit()ed')
    logging.info('prediction_thread() exited gracefully')
    conn.close()


def main() -> None:
    initialize_logger()
    prepare_database()
    signal.signal(signal.SIGINT, signal_handler)
    th_pred = Thread(target=prediction_thread)
    th_pred.start()
    th_zmq = Thread(target=zeromq_thread)
    th_zmq.start()
    waitress.serve(app, host='127.0.0.1', port='4386')
    logging.info('main() exited gracefully')


if __name__ == '__main__':
    main()
