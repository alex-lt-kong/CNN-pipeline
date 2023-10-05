from typing import Any, Dict
from flask import Flask, request, Response
from threading import Thread, Lock
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import List, Any

import datetime as dt
import helper
import io
import logging
import json
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
image_queue_min_len = 64
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
                            prediction: int, elapsed_time_ms: float) -> None:
    sql = """
        INSERT INTO inference_records(
            timestamp, model_output, prediction, elapsed_time_ms
        )
        VALUES (?, ?, ?, ?)
    """
    cur.execute(sql, (dt.datetime.now().isoformat(), model_output,
                      prediction, elapsed_time_ms))


def initialize_logger() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(name)8s | %(levelname)7s | %(message)s',
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class CustomDataset(Dataset):
    def __init__(
        self, images, transform: torchvision.transforms.Compose = None
    ) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Image:
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image


def prediction_thread() -> None:

    logging.info('prediction_thread() started')
    global prediction_interval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings: Dict[str, Any]
    with open(os.path.join(curr_dir, '..', '..', 'config.json')) as j:
        settings = json.load(j)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    model_ids = ['0', '1', '2']
    v16mms = []
    for i in range(len(model_ids)):
        v16mms.append(model.VGG16MinusMinus(2))
        v16mms[i].to(device)
        model_path = settings['model']['parameters'].replace(
            r'{id}', model_ids[i]
        )
        logging.info(f'Loading weights to model from: {model_path}')
        v16mms[i].load_state_dict(torch.load(model_path))
        total_params = sum(p.numel() for p in v16mms[i].parameters())
        logging.info(f"Weights loaded, number of parameters: {total_params:,}")
        v16mms[i].eval()

    DATASET_SIZE = 16
    assert DATASET_SIZE + image_context_start > 0
    # assert image_context_end - image_context_start == DATASET_SIZE
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
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 7])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 8])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 9])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 10])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 11])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 12])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 13])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 14])),
            Image.open(io.BytesIO(image_queue[DATASET_SIZE + 15]))
        ], transform=helper.test_transforms)
        assert DATASET_SIZE == len(dataset.images)
        dataloader = DataLoader(
            dataset, batch_size=DATASET_SIZE, shuffle=False, num_workers=4
        )

        # Classify the image using the model
        torch.set_grad_enabled(False)
        outputs: List[torch.Tensor] = []
        output = torch.zeros(
            [DATASET_SIZE, v16mms[0].num_classes], dtype=torch.float32
        )
        output = output.to(device)
        # Per current, batch_size is always equal to DATASET_SIZE
        # so there will always be only one batch
        for batch in dataloader:
            batch = batch.to(device)
            for i in range(len(v16mms)):
                outputs.append(v16mms[i](batch))
                output += outputs[i]
            _, pred_tensor = torch.max(output.data, 1)
        torch.set_grad_enabled(True)
        elapsed_time = time.time() - start_time
        for i in range(DATASET_SIZE):
            insert_prediction_to_db(cur, str(output[i]), int(pred_tensor[i]),
                                    round(elapsed_time * 1000.0, 1))

        if iter_count > 15:
            # commit() could be a very expensive operation
            # profiling shows it takes 1+ sec to complete
            conn.commit()
            logging.info('SQLite commit()ed')
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
            logging.warning(f'Raw results from {len(outputs)} models are:')
            for i in range(len(outputs)):
                output_str = ''
                for j in range(len(nonzero_preds)):
                    output_str += f'{nonzero_preds[j]}]: ' + str(outputs[i][nonzero_preds[j]]) + '\n'
                logging.warning(f'\n[{output_str}')
            output_str = ''
            for i in range(len(nonzero_preds)):
                output_str += f'{nonzero_preds[i]}]: ' + str(output[nonzero_preds[i]]) + '\n'
            logging.warning(f'and arithmetic average of raw results is:\n{output_str}')
            for i in range(image_context_end - image_context_start):
                temp_img_path = f'/tmp/frame{i}.jpg'
                with open(temp_img_path, "wb") as binary_file:
                    image_idx = int(
                        DATASET_SIZE + nonzero_preds[0].item() +
                        image_context_start + i * 2
                    )
                    logging.info(f'Writing {image_idx}-th frame from the queue '
                                 f'to path [{temp_img_path}]')
                    binary_file.write(image_queue[image_idx])

            for i in range(DATASET_SIZE):
                image_queue.pop(0)
            image_queue_mutex.release()
            logging.info(
                f'Calling program [{settings["inference"]["on_detected"]}]...'
            )
            result = subprocess.run(
                settings['inference']['on_detected']['external_program_py'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            logging.info(f'stdout: {result.stdout}')
            logging.info(f'stderr: {result.stderr}')

            cooldown_sec = 150
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
    logging.info('SQLite commit()ed')
    logging.info('prediction_thread() exited gracefully')
    conn.close()


def main() -> None:
    initialize_logger()
    logging.info('predict.py started')
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