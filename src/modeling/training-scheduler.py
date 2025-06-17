from typing import Any, Dict

import argparse
import json
import logging
import signal
import subprocess
import sys
import threading
import traceback
import time
import zlib


ev_flag = False
schedule: Dict[str, Any] = {}
schedule_file_path = ""
schedule_file_crc = ""
schedule_mutex = threading.Lock()


def signal_handler(signal, frame):
    print('SIGINT(Ctrl+C) received, program will exit in a min')
    global ev_flag
    ev_flag = True


def ev_schedule_monitor():
    global schedule_file_crc, schedule, schedule_mutex, ev_flag
    while ev_flag is not True:
        if schedule_file_crc != "":
            time.sleep(10)
        try:
            with open(schedule_file_path, 'rb') as f:
                data = f.read()
                crc = hex(zlib.crc32(data) & 0xffffffff)

                if crc == schedule_file_crc:
                    continue
                logging.warning(f'New CRC32({crc}) is different from the existing one'
                                f'({schedule_file_crc}), will reload the schedule file')
                schedule_mutex.acquire()
                schedule = json.loads(data)
            schedule_file_crc = crc
            logging.info(f'New schedule is:\n{json.dumps(schedule, indent=True)}')
            logging.info(f'next_task_idx is updated to {schedule["next_task_idx"]}')
        except Exception as ex:
            logging.error(f'Error loading: [{schedule_file_path}]: {ex}')
            # Need to set schedule_file_crc to something here so that it triggers the 10-sec sleep()
            schedule_file_crc = "NA"
            continue
        finally:
            if schedule_mutex.locked():
                schedule_mutex.release()


def ev_training_driver() -> None:
    global schedule_file_crc, ev_flag
    while ev_flag is not True:
        # Do NOT shortern/remove this sleep(), this gives us some valuable time to kill the process
        time.sleep(10)
        if ev_flag:
            break
        try:
            schedule_mutex.acquire()
            task_idx = schedule["next_task_idx"]
            if task_idx > schedule["end_task_idx_inclusive"]:
                raise IndexError(f"end_task_idx_inclusive {schedule['end_task_idx_inclusive']} reached")
            schedule["next_task_idx"] += 1
            task_args = schedule["training_tasks"][task_idx]
            assert isinstance(task_args, Dict)
            if 'skip' in task_args and task_args['skip']:
                logging.warning("skip is set to ture, this task will be skipped")
                continue
            items = [
                'prepare_training_data_script', 'python_interpreter',
                'training_script', 'output_pipe_to', 'model_id',
                'config_file', 'cuda_device', 'epochs', 'synthetic_multiplier',
                'batch_size', 'split_ratio', 'image_ext', 'revision',
                'prepare_training_data', 'model_base_name', 'dropout_rate',
                'load_parameters', 'source_dir', 'training_data_dir',
                'validation_data_dir'
            ]
            for item in items:
                if item not in task_args:
                    task_args[item] = schedule['defaults'][item]
            for item in items:
                for also_item in items:
                    if type(task_args[item]) == str and type(task_args[also_item]) == str:
                        task_args[item] = task_args[item].replace(f'{{{also_item}}}', task_args[also_item])
        except IndexError as ex:
            logging.error(f'task_idx {task_idx} reaches the end of the list, exiting...')
            ev_flag = True
            break
        except Exception as ex:
            logging.error(
                f'Error loading config of {task_idx}-th task. This task will be skipped:\n{ex}\n{traceback.format_exc()}')
            continue
        finally:
            if schedule_mutex.locked():
                schedule_mutex.release()
        logging.info(f'About to start the {task_idx}-th task, arguments are:\n{json.dumps(task_args, indent=True)}')
        stdout_file = None
        try:
            stdout_file = open(task_args['output_pipe_to'], "w", buffering=1)
            rc = 0
            cmd = [
                task_args['python_interpreter'], task_args['prepare_training_data_script'],
                "--config-path", task_args['config_file'],
                "--split-ratio", task_args['split_ratio'],
                "--synthetic-multiplier", task_args['synthetic_multiplier'],
                "--image-extension", task_args['image_ext'],
                "--source-dir", task_args['source_dir'],
                "--dest-dir-training", task_args['training_data_dir'],
                "--dest-dir-validation", task_args['validation_data_dir']
            ]
            if task_args['prepare_training_data']:
                logging.info(f'Command to prepare training data is: {cmd}')
                process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stdout_file)
                sout, serr = process.communicate()
                rc = process.returncode
            else:
                logging.info(f'prepare_training_data is False, skipping training data preparation')
            if rc != 0:
                logging.error(f'Returncode from {cmd} is non-zero ({rc}), training will be skipped')
                if (sout is not None and len(sout) > 0) or (serr is not None and len(serr) > 0):
                    logging.error(f'Uncaught stdout (bytes): {sout}; uncaught stderr (bytes): {serr}')
                continue
            cmd = [
                task_args['python_interpreter'], task_args['training_script'],
                "--config-path", task_args['config_file'],
                "--epochs", task_args['epochs'],
                "--dropout-rate", task_args['dropout_rate'],
                "--batch-size", task_args['batch_size'],
                "--model-name", task_args['model_base_name'],
                "--model-id", task_args['model_id'],
                "--load-parameters", task_args['load_parameters'],
                "--cuda-device-id", task_args['cuda_device'],
                "--training-data-dir", task_args['training_data_dir'],
                "--validation-data-dir", task_args['validation_data_dir']
            ]
            logging.info(f'Training data is ready, now running the training process: {cmd}')
            process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stdout_file)
            process.communicate()
            rc = process.returncode
            logging.info(f'Returncode from training process is {rc}')
        except Exception as ex:
            logging.error(f"Error running data preparation/training program:\n{ex}\n{traceback.format_exc()}")
        finally:
            if stdout_file is not None:
                stdout_file.close()


def main() -> None:
    signal.signal(signal.SIGINT, signal_handler)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--schedule-file', '-f', dest='schedule-file', required=True
    )
    args = vars(ap.parse_args())
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d | %(levelname)7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    global schedule_file_path
    schedule_file_path = args['schedule-file']
    th_schedule_monitor = threading.Thread(target=ev_schedule_monitor)
    th_schedule_monitor.start()
    th_training_driver = threading.Thread(target=ev_training_driver)
    th_training_driver.start()
    th_schedule_monitor.join()
    th_training_driver.join()


if __name__ == '__main__':
    main()
