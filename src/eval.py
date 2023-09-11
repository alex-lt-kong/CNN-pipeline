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


ap = argparse.ArgumentParser()
ap.add_argument('--model-id', '-i', dest='model-id', required=True)
args = vars(ap.parse_args())
model_id = args['model-id']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)
v16mm = model.VGG16MinusMinus(2)
v16mm.to(device)

curr_dir = os.path.dirname(os.path.abspath(__file__))
# settings: Dict[str, Any]
with open(os.path.join(curr_dir, '..', 'config.json')) as j:
    settings = json.load(j)

v16mm.load_state_dict(torch.load(settings['model']['parameters'].replace(r'{idx}', model_id)))
misclassified_dir = settings['diagnostics']['misclassified'].replace(r'{idx}', model_id)
if os.path.exists(misclassified_dir):
    shutil.rmtree(misclassified_dir)
dataset = ImageFolder(root=settings['dataset']['validation'],
                      transform=helper.test_transforms)

batch_size = 16
misclassified_count = 0
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# shuffle breaks the relationship of batch and file path.

v16mm.eval()
# Classify the image using the model
torch.set_grad_enabled(False)
# Loop through the images in batches
num_correct = 0
total_samples = 0
true_positives = np.zeros(v16mm.num_classes)
false_positives = np.zeros(v16mm.num_classes)
false_negatives = np.zeros(v16mm.num_classes)

y_trues_total = []
y_preds_total = []
for batch_idx, (images, y_trues) in enumerate(data_loader):

    print(f'Evaluating {batch_idx+1}/{len(data_loader)} batch of samples...')

    images, y_trues = images.to(device), y_trues.to(device)
    # Use your model to make predictions for the batch of images
    output = v16mm(images)
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
        print(f"[{misclassified_count}-th misclassified sample] {i}-th sample's y-true is "
              f"{y_true} but y-hat is {pred_label}", end='')
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
torch.set_grad_enabled(True)

precision = np.zeros(v16mm.num_classes)
recall = np.zeros(v16mm.num_classes)
fscore = np.zeros(v16mm.num_classes)
for i in range(v16mm.num_classes):
    precision[i] = true_positives[i] / (true_positives[i] + false_positives[i])
    recall[i] = true_positives[i] / (true_positives[i] + false_negatives[i])
    beta = 1  # set beta to 1 for f1-score
    fscore[i] = (1 + beta**2) * (precision[i] * recall[i]) / (beta**2 * precision[i] + recall[i])

print('Class\tPrecision\tRecall\t\tF-Score')
for i in range(v16mm.num_classes):
    print('{}    \t{:.2f}%\t\t{:.2f}%\t\t{:.2f}%'.format(
        i, precision[i] * 100, recall[i] * 100, fscore[i] * 100))

print(f'{misclassified_count} misclassified samples found')
cm = metrics.confusion_matrix(y_trues_total, y_preds_total)
print(f'Confusion matrix (true-by-pred):\n{cm}')
