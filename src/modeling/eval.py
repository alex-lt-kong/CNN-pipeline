from typing import Any, Dict, List, Tuple
from PIL import Image
from torchvision.datasets import ImageFolder
from sklearn import metrics
from torch.utils.data import DataLoader

import argparse
import model
import numpy as np
import json
import helper
import os
import shutil
import torch

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable')
device = torch.device("cuda")


def evaluate(
    settings: Dict[str, Any], v16mms: List[model.VGG16MinusMinus],
    dataset: ImageFolder, misclassified_dir: str
) -> None:

    batch_size = 16
    misclassified_count = 0
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    # shuffle breaks the relationship of batch and file path.

    # Classify the image using the model
    torch.set_grad_enabled(False)
    # Loop through the images in batches
    num_correct = 0
    total_samples = 0
    true_positives = np.zeros(v16mms[0].num_classes)
    false_positives = np.zeros(v16mms[0].num_classes)
    false_negatives = np.zeros(v16mms[0].num_classes)

    y_trues_total = []
    y_preds_total = []
    misclassified_samples: List[Tuple[int, int]] = []
    for batch_idx, (images, y_trues) in enumerate(data_loader):

        print(f'Evaluating {batch_idx+1}/{len(data_loader)} batch of samples...')

        # Use your model to make predictions for the batch of images
        outputs: List[torch.Tensor] = []
        output = torch.zeros([len(y_trues), v16mms[0].num_classes], dtype=torch.float32)
        images, y_trues = images.to(device), y_trues.to(device)
        output = output.to(device)
        for i in range(len(v16mms)):
            outputs.append(v16mms[i](images))
            output += outputs[i]

        assert isinstance(output, torch.Tensor)
        y_preds = torch.argmax(output, dim=1)
        y_trues_total.extend(y_trues.tolist())
        y_preds_total.extend(y_preds.tolist())
        num_correct += (y_preds == y_trues).sum().item()
        total_samples += y_trues.size(0)
        # Loop through the predicted labels and check if they match the true labels
        for i, pred_label in enumerate(y_preds):
            if y_preds[i] == y_trues[i]:
                true_positives[y_trues[i]] += 1
            else:
                false_positives[y_preds[i]] += 1
                false_negatives[y_trues[i]] += 1

            y_true = y_trues[i].item()
            pred_label = pred_label.item()

            if pred_label == y_true:
                continue

            misclassified_count += 1
            misclassified_samples.append((batch_idx, i))
            print(
                f"[{misclassified_count}-th misclassified sample] {i}-th sample's y-true is "
                f"{y_true} but y-hat is {pred_label}", end=''
            )
            # If the predicted label is incorrect, save the misclassified image to the file system
            folder_name = str(y_true)
            output_dir = os.path.join(misclassified_dir, folder_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                dataset.samples[batch_idx * batch_size + i][0].split("/")[-1]
            )
            print(f', sample copied from {dataset.samples[batch_idx * batch_size + i][0]} to {output_path}')
            Image.open(dataset.samples[batch_idx * batch_size + i][0]).save(output_path)
            print(f'Raw results from {len(outputs)} models are:')
            for j in range(len(outputs)):
                print(outputs[j][i])
            print(f'and arithmetic average of raw results is:\n{output[i]}')

    torch.set_grad_enabled(True)
    print(f'All misclassified samples are:\n{misclassified_samples}')

    precision = np.zeros(v16mms[0].num_classes)
    recall = np.zeros(v16mms[0].num_classes)
    fscore = np.zeros(v16mms[0].num_classes)
    for i in range(v16mms[0].num_classes):
        precision[i] = true_positives[i] / (true_positives[i] + false_positives[i])
        recall[i] = true_positives[i] / (true_positives[i] + false_negatives[i])
        beta = 1  # set beta to 1 for f1-score
        fscore[i] = (1 + beta**2) * (precision[i] * recall[i]) / (beta**2 * precision[i] + recall[i])

    print('Class\tPrecision\tRecall\t\tF-Score')
    for i in range(v16mms[0].num_classes):
        print('{}    \t{:.2f}%\t\t{:.2f}%\t\t{:.2f}%'.format(
            i, precision[i] * 100, recall[i] * 100, fscore[i] * 100))

    print(f'{misclassified_count} misclassified samples found')
    cm = metrics.confusion_matrix(y_trues_total, y_preds_total)
    print(f'Confusion matrix (true-by-pred):\n{cm}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-path', '-c', dest='config-path', required=True,
                    help='Config file path')
    ap.add_argument('--model-ids', '-i', dest='model-ids', required=True)
    args = vars(ap.parse_args())
    model_ids = str(args['model-ids']).split(',')


    # settings: Dict[str, Any]
    with open(args['config-path']) as j:
        settings = json.load(j)
    target_img_size = (
        settings['model']['input_image_size']['height'],
        settings['model']['input_image_size']['width']
    )

    helper.init_transforms(target_img_size)
    v16mms = []
    for i in range(len(model_ids)):
        v16mms.append(model.VGG16MinusMinus(
            settings['model']['num_output_class'], target_img_size))
        v16mms[i].to(device)
        model_path = settings['model']['parameters'].replace(r'{id}', model_ids[i])
        print(f'Loading parameters from [{model_path}] to model [{model_ids[i]}]')
        v16mms[i].load_state_dict(torch.load(model_path))
        v16mms[i].eval()
        total_params = sum(p.numel() for p in v16mms[i].parameters())
        print(f"Number of parameters: {total_params:,}")

    misclassified_dir = os.path.join(settings['model']['diagnostics_dir'], 'misclassified')
    if os.path.exists(misclassified_dir):
        shutil.rmtree(misclassified_dir)
    print(f'Misclassified sample will be saved to: {misclassified_dir}')
    for dir in [
        os.path.join(settings['dataset']['training']),
        os.path.join(settings['dataset']['validation'])
    ]:
        print(f'\n\n====={dir}=====\n\n')
        dataset = ImageFolder(
            root=dir,
            transform=helper.dummy_transforms
        )
        evaluate(settings, v16mms, dataset, misclassified_dir)
        input()

if __name__ == '__main__':
    main()
