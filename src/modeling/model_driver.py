from model_definitions import *
from sklearn import metrics
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
from typing import Any, Dict, Tuple
from distutils.util import strtobool

import argparse
import datetime as dt
import helper
import logging
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import time


curr_dir = os.path.dirname(os.path.abspath(__file__))
device: torch.device
config: Dict[str, Any]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Prevents PyTorch from using the cudnn auto-tuner to find the fastest
    # convolution algorithms, which can result in non-deterministic behavior.
    torch.backends.cudnn.benchmark = False


def get_data_loaders(
    training_data_dir: str, test_data_dir: str, batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = ImageFolder(root=training_data_dir, transform=helper.dummy_transforms)
    test_ds = ImageFolder(root=test_data_dir, transform=helper.dummy_transforms)
    assert len(train_ds) >= len(test_ds), "How come len(train_ds) < len(test_ds)??"
    num_samples = int(len(test_ds) * 0.1)
    train_ds_random_sampler = RandomSampler(train_ds, num_samples=num_samples)
    test_ds_random_sampler = RandomSampler(test_ds, num_samples=num_samples)
    shuffle = True

    # not setting num_workers disables sample prefetching,
    # which drags down performance a lot
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, pin_memory=True
    )
    sampled_train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        num_workers=4, pin_memory=True, sampler=train_ds_random_sampler
    )
    sampled_test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        num_workers=4, pin_memory=True,
        sampler=test_ds_random_sampler
    )
    return (train_loader, sampled_train_loader, sampled_test_loader)


def write_metrics_to_csv(filename: str, metrics_dict: Dict[str, float]) -> None:

    csv_dir = os.path.join(config['model']['diagnostics_dir'], 'training')
    if not os.path.isdir(csv_dir):
        os.makedirs(csv_dir)
    csv_path = os.path.join(csv_dir, filename)
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()
    # logging.info(f'Appending:\n{metrics_dict}\n--- to ---\n{df}')
    df = pd.concat([df, pd.DataFrame([metrics_dict])], ignore_index=True)
    # df = df.append(metrics_dict, ignore_index=True)
    df.to_csv(csv_path, index=False)


def evalute_model_classification(
    model: nn.Module, num_classes: int, data_loader: DataLoader,
    ds_name: str
) -> None:
    # initialize the number of correct predictions, total number of samples,
    # and true positives, false positives, and false negatives for each class
    num_correct = 0
    total_samples = 0
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)

    y_trues_total = []
    y_preds_total = []
    # Disabling gradient calculation is useful for inference, when you are
    # sure that you will not call Tensor.backward(). It will reduce memory
    # consumption for computations that would otherwise have requires_grad=True.
    torch.set_grad_enabled(False)
    for images, y_trues in data_loader:
        # logging.info(batch_idx)
        images, y_trues = images.to(device), y_trues.to(device)
        # forward pass
        output = model(images)
        # compute the predicted labels
        y_preds = torch.argmax(output, dim=1)

        y_trues_total.extend(y_trues.tolist())
        y_preds_total.extend(y_preds.tolist())
        # update the number of correct predictions, total number of
        # samples, and true positives, false positives, and false
        # negatives for each class
        num_correct += (y_preds == y_trues).sum().item()
        total_samples += y_trues.size(0)
        for i in range(y_trues.size(0)):
            if y_preds[i] == y_trues[i]:
                true_positives[y_trues[i]] += 1
            else:
                false_positives[y_preds[i]] += 1
                false_negatives[y_trues[i]] += 1
        # logging.info('Iter done')
    torch.set_grad_enabled(True)
    # compute the accuracy
    # accuracy = num_correct / total_samples
    # logging.info('Accuracy: {:.2f}%'.format(accuracy * 100))
    # compute the precision, recall, and f-score for each class
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    fscore = np.zeros(num_classes)
    for i in range(num_classes):
        precision[i] = true_positives[i] / \
            (true_positives[i] + false_positives[i])
        recall[i] = true_positives[i] / \
            (true_positives[i] + false_negatives[i])
        beta = 1  # set beta to 1 for f1-score
        fscore[i] = (1 + beta**2) * (precision[i] * recall[i]
                                     ) / (beta**2 * precision[i] + recall[i])

    logging.info(f'Metrics from dataset: {ds_name}')

    metrics_dict = {}
    logging.info('Class\tPrecision\tRecall\t\tF-Score')
    for i in range(num_classes):
        metrics_dict[f'{i}_precision'] = precision[i]
        metrics_dict[f'{i}_recall'] = recall[i]
        metrics_dict[f'{i}_fscore'] = fscore[i]
        logging.info('{}    \t{:.2f}%\t\t{:.2f}%\t\t{:.2f}%'.format(
            i, precision[i] * 100, recall[i] * 100, fscore[i] * 100))
    write_metrics_to_csv(f'{ds_name}.csv', metrics_dict)

    generate_curves(ds_name)
    generate_curves(ds_name, 4)
    generate_curves(ds_name, 16)
    generate_curves(ds_name, 32)

    cm = metrics.confusion_matrix(y_trues_total, y_preds_total)
    logging.info(f'Confusion matrix (true-by-pred):\n{cm}')


def save_params(m: nn.Module, model_id: str) -> None:
    model_params_path = config['model']['parameters'].replace(
        r'{id}', model_id
    )
    if os.path.exists(model_params_path):
        os.remove(model_params_path)
    torch.save(m.state_dict(), model_params_path)
    logging.info(f'Model weights saved to [{model_params_path}]')


def save_ts_model(m: nn.Module, model_id: str) -> None:
    assert isinstance(m, nn.Module)
    logging.info('Serializing model to Torch Script file')
    ts_serialization_path = config['model'][
        'ts_model_path'
    ].replace(r'{id}', model_id)
    if os.path.exists(ts_serialization_path):
        os.remove(ts_serialization_path)
    m_ts = torch.jit.script(m)
    logging.info('Torch Script model created')
    m_ts.save(ts_serialization_path)
    logging.info(f'Torch Script model saved to [{ts_serialization_path}]')


def save_transformed_samples(dataloader: DataLoader,
                             save_dir: str, num_samples: int) -> None:
    logging.info(f'Saving transformed samples to {save_dir}')
    from torchvision.utils import save_image
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    dataset_size = len(dataloader.dataset)  # type: ignore
    logging.info(
        f"{dataset_size:,} images are in this dataset and we save "
        f"{num_samples} from it to [{save_dir}] for preview."
    )
    for i in range(num_samples):
        image_dst_path = f"{save_dir}/sample_{i}.jpg"
        sample_idx = random.randint(0, dataset_size - 1)
        save_image(
            dataloader.dataset[sample_idx][0],
            # dataloader.dataset[sample_idx][1] is label
            image_dst_path
        )


def train(
    load_parameters: bool, model_name: str, model_id: str, training_dir: str,
    validation_dir: str, dropout_rate: float = 0.001, lr: float = 0.001,
    epochs: int = 10, batch_size: int = 64
) -> nn.Module:

    m = globals()[model_name](config, dropout_rate)
    assert isinstance(m, nn.Module)
    m = m.to(device)

    if load_parameters:
        logging.warning(
            'Loading existing model parameters to continue training')
        m.load_state_dict(torch.load(
            config['model']['parameters'].replace(r'{id}', model_id)
        ))

    logging.info('Name                  |       Params | Structure')
    total_parameters = 0
    for name, module in m.named_modules():
        if isinstance(module, nn.Sequential):
            # Sequential() is like a wrapper module, we will print layers twice
            # if we don't skip it.
            continue
        if len(name) == 0:
            # Will print the entire model as a layer, let's skip it
            continue
        layer_parameters = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_parameters += layer_parameters
        logging.info(f'{name.ljust(21)} | {layer_parameters: >12,} | {module}')
    logging.info(f'{"Total".ljust(21)} | {total_parameters: >12,} | NA')

    training_samples_dir = training_dir
    test_samples_dir = validation_dir
    logging.info(f'Loading samples from [{training_samples_dir}] and [{test_samples_dir}]')

    # Define the dataset and data loader for the training set
    train_loader, sampled_train_loader, sampled_test_loader = get_data_loaders(
        training_samples_dir, test_samples_dir, batch_size
    )
    save_transformed_samples(
        train_loader,
        os.path.join(config['model']['diagnostics_dir'], 'preview', f'training_samples_{model_id}'),
        30
    )

    save_transformed_samples(
        sampled_test_loader,
        os.path.join(config['model']['diagnostics_dir'], 'preview', f'test_samples_{model_id}'),
        5
    )

    # Define the loss function, optimizer and learning rate scheduler
    loss_fn = nn.CrossEntropyLoss()

    # weight_decay is L2 regularization's lambda
    optimizer = optim.Adam(
        m.parameters(),
        lr=(0.001 if lr is None else lr),
        weight_decay=3e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    start_ts = time.time()
    # Train the model
    for epoch in range(epochs):
        logging.info('\n========================================\n'
                     f'Epoch {epoch + 1} / {epochs} started, '
                     f'lr: {scheduler.get_last_lr()}'
                     '\n========================================')
        # hook_handle = m.fc.register_forward_hook(
        #    lambda m, inp, out: torch.nn.functional.dropout(
        #        out, p=dropout_rate, training=m.training
        # ))
        m.train()   # switch to training mode
        for batch_idx, (images, y_trues) in enumerate(train_loader):
            # Every data instance is an input + label pair
            images, y_trues = images.to(device), y_trues.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()
            # Make predictions for this batch
            y_preds = m(images)

            # Compute the loss
            loss = loss_fn(y_preds, y_trues)

            # l1_lambda = 1e-4
            # l1_norm = sum(torch.linalg.norm(p, 1) for p in v16mm.parameters())
            # weight_decay is L2 reg already
            # l2_lambda = 1e-3
            # l2_norm = sum(p.pow(2.0).sum() for p in v16mm.parameters())

            # loss = loss + l1_lambda * l1_norm  # + l2_lambda * l2_norm
            # Computes the gradient of current tensor w.r.t. graph leaves.
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            if (((batch_idx + 1) % 500 == 0) or
                    (batch_idx + 1 == len(train_loader))):
                logging.info(f"Step {batch_idx+1}/{len(train_loader)}, "
                             f"loss: {loss.item():.5f}")

        scheduler.step()
        logging.info('Evaluating model after this epoch')

        # switch to evaluation mode
        m.eval()
        time.sleep(1)
        evalute_model_classification(
            m, config['model']['num_classes'], sampled_train_loader, f'training_{model_id}'
        )
        time.sleep(1)
        evalute_model_classification(
            m, config['model']['num_classes'], sampled_test_loader, f'test_{model_id}'
        )
        time.sleep(1)

        save_params(m, model_id)
        # hook_handle.remove()
        save_ts_model(m, model_id)
        eta = start_ts + (time.time() - start_ts) / ((epoch + 1) / epochs)
        logging.info(
            f'ETA: {dt.datetime.fromtimestamp(eta).astimezone().isoformat()}'
            f', estimated training duration: {(eta - start_ts)/3600:.1f} hrs'
        )
        # seems the server can overheat, let's give it a short break after each iteration...
        time.sleep(10)
    return m


def generate_curves(filename: str, mv_window: int = 1) -> None:
    csv_path = os.path.join(
        config['model']['diagnostics_dir'], 'training', f'{filename}.csv'
    )
    img_path = os.path.join(
        config['model']['diagnostics_dir'], 'training',
        f'{filename}_mv{mv_window}.png'
    )
    if os.path.isfile(csv_path) is not True:
        raise FileNotFoundError(f'{csv_path} not found')
    df = pd.read_csv(csv_path)

    plt.clf()
    for col in df.columns:
        df[col] = df[col].rolling(window=mv_window).mean()
        plt.plot(df.index, df[col], label=col)

    # Customize chart
    plt.title(f'{filename}_{mv_window}')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.grid(True)
    plt.legend()
    plt.savefig(img_path)


def main() -> None:

    global config, device
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d | %(levelname)7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    ap = argparse.ArgumentParser()
    # Not using store_true is to make training-scheduler's command construction easier.
    ap.add_argument('--load-parameters', '-l', dest='load_parameters', default=False,
                    type=strtobool, help='load existing parameters for continue training')
    ap.add_argument('--config-path', '-c', dest='config-path', required=True,
                    help='Config file path')
    ap.add_argument('--learning-rate', '-lr', dest='learning_rate', type=float,
                    help='Specify a learning rate', default=0.001)
    ap.add_argument('--epochs', '-e', dest='epochs', type=int, default=10)
    ap.add_argument('--model-id', '-i', dest='model_id', required=True)
    ap.add_argument('--model-name', '-n', dest='model_name', required=True)
    ap.add_argument('--dropout-rate', '-d', dest='dropout_rate',
                    default=0.5, type=float)
    ap.add_argument('--batch-size', '-b', dest='batch_size', default=64, type=int)
    ap.add_argument('--cuda-device-id', '-g', dest='cuda-device-id',
                    default='cuda',
                    help=('Specify GPU to use following CUDA semantics. '
                          'Sample values include "cuda"/"cuda:0"/"cuda:1"'))
    ap.add_argument('--training-data-dir', dest='training_data_dir', type=str)
    ap.add_argument('--validation-data-dir', dest='validation_data_dir', type=str)
    args = vars(ap.parse_args())

    with open(args['config-path']) as j:
        config = json.load(j)
    device = helper.get_cuda_device(args['cuda-device-id'])
    properties = torch.cuda.get_device_properties(device)
    logging.info(f"GPU Model: {properties.name}")
    logging.info(f"GPU Memory: {properties.total_memory / 1024**3:.2f} GB")
    logging.info(f"GPU CUDA semantics: {device}")

    if 'random_seeds' in config['model'] and args['model_id'] in config['model']['random_seeds']:
        set_seed(config['model']['random_seeds'][args['model_id']])
    else:
        logging.warning(
            f'model_id [{args["model_id"]}] does not have pre-defined random seed, '
            'will use unix epoch time as seed instead'
        )
        set_seed(int(time.time()))
    helper.init_transforms((
        config['model']['input_image_size']['height'],
        config['model']['input_image_size']['width']
    ))
    train(
        bool(args['load_parameters']), args['model_name'], args['model_id'],
        args['training_data_dir'], args['validation_data_dir'],
        float(args['dropout_rate']), args['learning_rate'], args['epochs'],
        int(args['batch_size'])
    )
    logging.info('Training completed')


if __name__ == '__main__':
    main()
