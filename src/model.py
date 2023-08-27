from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from typing import Dict, Any, Tuple, Optional


import argparse
import datetime as dt
import helper
import logging
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
import torchvision
import torch.cuda
import time


curr_dir = os.path.dirname(os.path.abspath(__file__))
device = helper.get_cuda_device()
config: Dict[str, Any]


class VGG16MinusMinus(nn.Module):
    dropout = 0.7

    def __init__(self, num_classes: int = 10) -> None:
        super(VGG16MinusMinus, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        # self.layer6 = nn.Sequential(
        #    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm2d(256),
        #    nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        # self.layer9 = nn.Sequential(
        #    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm2d(512),
        #    nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        # self.layer12 = nn.Sequential(
        #    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm2d(512),
        #    nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(int(224 / 32) * int(426 / 32) * 512, int(4096 / 6)),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(int(4096 / 6), int(4096 / 6)),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(int(4096 / 6), num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        # x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        # x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def get_data_loaders(data_path: str,
                     random_seed: int = 0) -> Tuple[DataLoader, DataLoader]:

    class TransformedSubset(torch.utils.data.Subset):

        def __init__(
            self, subset: torch.utils.data.Subset,
            transform: Optional[torchvision.transforms.Compose] = None
        ) -> None:
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self) -> int:
            return len(self.subset)

    dataset = ImageFolder(root=data_path, transform=None)
    val_split = 0.2
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed))
    batch_size = 20
    shuffle = True
    # Apply the respective transformations to each subset
    train_dataset = TransformedSubset(train_dataset, transform=helper.train_transforms)
    val_dataset = TransformedSubset(val_dataset, transform=helper.test_transforms)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size, shuffle=shuffle)
    return (train_loader, val_loader)


def write_metrics_to_csv(filename: str, metrics_dict: Dict[str, float]) -> None:

    csv_path = os.path.join(curr_dir, '..', 'diagnostics', filename)
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()
    # logging.info(f'Appending:\n{metrics_dict}\n--- to ---\n{df}')
    df = df.append(metrics_dict, ignore_index=True)
    df.to_csv(csv_path, index=False)


def evalute_model_classification(
    model: nn.Module, num_classes: int, data_loader: DataLoader,
    ds_name: str, sample_ratio: float
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
    with torch.no_grad():
        for batch_idx, (images, y_trues) in enumerate(data_loader):
            if random.randint(0, 99) > sample_ratio * 100:
                continue

            images, y_trues = images.to(device), y_trues.to(device)
            # forward pass
            y_preds = model(images)

            # compute the predicted labels
            _, predicted = torch.max(y_preds, 1)

            y_trues_total.extend(y_trues.tolist())
            y_preds_total.extend(predicted.tolist())
            # update the number of correct predictions, total number of
            # samples, and true positives, false positives, and false
            # negatives for each class
            num_correct += (predicted == y_trues).sum().item()
            total_samples += y_trues.size(0)
            for i in range(y_trues.size(0)):
                if predicted[i] == y_trues[i]:
                    true_positives[y_trues[i]] += 1
                else:
                    false_positives[predicted[i]] += 1
                    false_negatives[y_trues[i]] += 1

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

    logging.info(f'Metrics from dataset: {ds_name} '
                 f'({sample_ratio * 100}% of samples evaluted)')

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
    generate_curves(ds_name, 10)
    generate_curves(ds_name, 20)
    generate_curves(ds_name, 40)

    cm = metrics.confusion_matrix(y_trues_total, y_preds_total)
    logging.info(f'Confusion matrix (true-by-pred):\n{cm}')


def save_params(m: nn.Module) -> None:
    if os.path.exists(config['model']['parameters']):
        dst_path = config["model"]["parameters"] + ".bak"
        logging.warning(
            'Model saved from the previous training exists, '
            f'will move from [{config["model"]["parameters"]}] to [{dst_path}]'
        )
        shutil.move(config['model']['parameters'], dst_path)
    torch.save(m.state_dict(), config['model']['parameters'])
    logging.info('Model weights saved to '
                 f'[{config["model"]["parameters"]}]')


def save_ts_model(m: nn.Module) -> None:
    logging.info('Serializing model to Torch Script file')
    if os.path.exists(config['model']['torch_script_serialization']):
        dst_path = config["model"]["torch_script_serialization"] + ".bak"
        logging.warning(
            'Torch Script file saved from the previous training exists, '
            f'will move from [{config["model"]["torch_script_serialization"]}] '
            f'to [{dst_path}]'
        )
        shutil.move(config['model']['torch_script_serialization'], dst_path)
    m_ts = torch.jit.script(m)
    logging.info('Torch Script model created')
    m_ts.save(config['model']['torch_script_serialization'])
    logging.info(
        'Torch Script model saved to '
        f'{config["model"]["torch_script_serialization"]}'
    )


def save_transformed_samples(dataloader: DataLoader,
                             save_dir: str, num_samples: int) -> None:
    logging.info(f'Saving transformed samples to {save_dir}')
    from torchvision.utils import save_image
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    dataset_size = len(dataloader.dataset)
    for i in range(num_samples):
        image_dst_path = f"{save_dir}/sample_{i}.jpg"
        sample_idx = random.randint(0, dataset_size - 1)
        save_image(
            dataloader.dataset[sample_idx][0],
            # dataloader.dataset[sample_idx][1] is label
            image_dst_path
        )


def train(load_parameters: bool, lr: float = 0.001, epochs: int = 10) -> nn.Module:
    start_ts = time.time()

    logging.info(f'Training using {device}')

    num_classes = 2
    v16mm = VGG16MinusMinus(num_classes)
    v16mm.to(device)
    if load_parameters:
        logging.warning(
            'Loading existing model parameters to continue training')
        v16mm.load_state_dict(torch.load(config['model']['parameters']))
    logging.info(v16mm)
    total_params = sum(p.numel() for p in v16mm.parameters())
    logging.info(f"Number of parameters: {total_params:,}")

    # Define the dataset and data loader for the training set
    train_loader, val_loader = get_data_loaders(
        config["dataset"]["path"],
        config['dataset']['validation_split_seed']
    )
    save_transformed_samples(
        train_loader, config['diagnostics']['preview']['training_samples'], 50
    )
    save_transformed_samples(
        val_loader, config['diagnostics']['preview']['validation_samples'], 10
    )

    # Define the loss function, optimizer and learning rate scheduler
    loss_fn = nn.CrossEntropyLoss()

    # weight_decay is L2 regularization's lambda
    optimizer = optim.Adam(
        v16mm.parameters(),
        lr=(0.001 if lr is None else lr),
        weight_decay=3e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # Train the model
    for epoch in range(epochs):
        logging.info('\n========================================\n'
                     f'Epoch {epoch + 1} / {epochs} started, '
                     f'lr: {scheduler.get_last_lr()}'
                     '\n========================================')

        v16mm.train()   # switch to training mode
        for batch_idx, (images, y_trues) in enumerate(train_loader):
            # Every data instance is an input + label pair
            images, y_trues = images.to(device), y_trues.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()
            # Make predictions for this batch
            y_preds = v16mm(images)

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

        evalute_model_classification(v16mm, num_classes, train_loader,
                                     'training_eval-off', 0.1)
        # switch to evaluation mode
        v16mm.eval()
        evalute_model_classification(v16mm, num_classes, train_loader,
                                     'training_eval-on', 0.1)
        evalute_model_classification(v16mm, num_classes, val_loader,
                                     'validation', 0.5)
        eta = start_ts + (time.time() - start_ts) / ((epoch + 1) / epochs)
        logging.info(
            f'ETA: {dt.datetime.fromtimestamp(eta).astimezone().isoformat()}'
            f', estimated training duration: {(eta - start_ts)/3600:.1f} hrs'
        )
    return v16mm


def generate_curves(filename: str, mv_window: int = 1) -> None:
    csv_path = os.path.join(curr_dir, '..', 'diagnostics', f'{filename}.csv')
    img_path = os.path.join(curr_dir, '..', 'diagnostics',
                            f'{filename}_mv{mv_window}.png')
    if os.path.isfile(csv_path) is not True:
        raise FileNotFoundError(f'{csv_path} not found')
    df = pd.read_csv(csv_path)

    plt.clf()
    for col in df.columns:
        df[col] = df[col].rolling(window=mv_window).mean()
        plt.plot(df.index, df[col], label=col)

    # Customize chart
    plt.title('Line Chart')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.legend()
    plt.savefig(img_path)


def main() -> None:

    global config
    with open(os.path.join(curr_dir, '..', 'config.json')) as j:
        config = json.load(j)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    ap = argparse.ArgumentParser()
    ap.add_argument('--load-parameters', '-l', action='store_true',
                    help='load existing parameters for continue training')
    ap.add_argument('--learning-rate', '-lr', dest='learning_rate',
                    help='Specify a learning rate')
    ap.add_argument('--epochs', '-e', dest='epochs')
    args = vars(ap.parse_args())

    try:
        lr = float(args['learning_rate'])
    except Exception:
        lr = 0.001
    try:
        epochs = int(args['epochs'])
    except Exception:
        epochs = 10
    m = train(args['load_parameters'], lr, epochs)
    save_params(m)
    # save_ts_model(m)
    logging.info('Training completed')


if __name__ == '__main__':
    main()
