from flask import Flask, request, Response
from threading import Thread, Lock
from typing import List, Any, Dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import argparse
import datetime as dt
import io
import json
import logging
import model
import os
import time
import signal
import sqlite3
import subprocess
import torch
import torchvision
import waitress
import sys
import zmq


app = Flask(__name__)
curr_dir = os.path.dirname(os.path.realpath(__file__))
prediction_interval = 600
ev_flag = True
image_queue_mutex = Lock()
image_queue: List[Any] = []
image_queue_min_len = 48
image_queue_max_len = image_queue_min_len * 2
image_context_start = -3
image_context_end = 12
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
        '%(asctime)s | %(name)8s | %(levelname)7s | %(message)s'
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

class CustomDataset(Dataset):
    def __init__(
        self, images, transform:torchvision.transforms.Compose=None
    ) -> None:
        self.images = images
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

def prediction_thread() -> None:
    logging.info('prediction_thread() started')
    global prediction_interval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = read_config_file()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    v16mm = model.VGG16MinusMinus(2)
    v16mm.to(device)
    logging.info(f'Loading weights to model from: {settings["model"]["model"]}')
    v16mm.load_state_dict(torch.load(settings['model']['model']))
    total_params = sum(p.numel() for p in v16mm.parameters())
    logging.info(f"Weights loaded, number of parameters: {total_params:,}")

    DATASET_SIZE = 8
    assert DATASET_SIZE + image_context_start > 0
    iter_count = 0    
    while ev_flag:
        iter_count += 1
        image_queue_mutex.acquire()
        logging.info("Iterating prediction loop "
                     f"(len(image_queue): {len(image_queue)})...")
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
        
        # Create an instance of your custom dataset class
        dataset = CustomDataset([
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 0])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 1])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 2])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 3])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 4])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 5])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 6])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 7]))
        ], transform=v16mm.transforms)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

        # Classify the image using the model
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                output = v16mm(batch)
                # output = v16mm(input_data)
                _, pred_tensor = torch.max(output.data, 1)
        elapsed_time = time.time() - start_time
        for i in range(DATASET_SIZE):
            insert_prediction_to_db(cur, str(output[i]), int(pred_tensor[i]),
                                    round(elapsed_time * 1000.0, 1))
        
        if iter_count > 15:
            # commit() could be a very expensive operation
            # profiling shows it takes 1+ sec to complete            
            conn.commit()
            logging.info(f'SQLite commit()ed')
            iter_count = 0

        nonzero_preds = torch.nonzero(pred_tensor)
        if len(nonzero_preds) > 0:
            logging.warning(
                f'Target detected at {nonzero_preds[0].item()}-th frame in '
                f'a batch of {DATASET_SIZE} frames '
                f'({nonzero_preds[0].item() + DATASET_SIZE}-th frame in the '
                f'queue of {len(image_queue)} frames), '
                'preparing context frames')
            logging.warning(f'The entire output is: {pred_tensor}')
            for i in range(image_context_end - image_context_start):
                temp_img_path = f'/tmp/frame{i}.jpg'
                with open(temp_img_path, "wb") as binary_file:
                    image_idx = DATASET_SIZE + nonzero_preds[0].item() + \
                                image_context_start + i * 2
                    logging.info(f'Writing {image_idx}-th frame from the queue '
                                 f'to path [{temp_img_path}]')
                    binary_file.write(image_queue[image_idx])

            for i in range(DATASET_SIZE):
                image_queue.pop(0)
            image_queue_mutex.release()
            logging.info('Calling downstream program...')
            result = subprocess.run(
                [settings['prediction']['on_detected']],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            logging.info(f'stdout: {result.stdout}')
            logging.info(f'stderr: {result.stderr}')

            cooldown_sec = 90
            logging.info(f'Entered cooldown period ({cooldown_sec} sec)')
            count = 0
            while count < cooldown_sec * 2 and ev_flag:
                count += 1
                time.sleep(0.5)
        else:

            for i in range(DATASET_SIZE):
                image_queue.pop(0)
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
