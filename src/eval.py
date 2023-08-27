from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import model
import helper
import os
import shutil
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)
v16mm = model.VGG16MinusMinus(2)
v16mm.to(device)

settings = helper.read_config_file()
v16mm.load_state_dict(torch.load(settings['model']['parameters']))
if os.path.exists(settings['diagnostics']['misclassified']):
    shutil.rmtree(settings['diagnostics']['misclassified'])
dataset = ImageFolder(root=settings['dataset']['path'],
                      transform=helper.test_transforms)

batch_size = 48
misclassified_count = 0
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# shuffle breaks the relationship of batch and file path.

v16mm.eval()
# Classify the image using the model
with torch.no_grad():
    # Loop through the images in batches
    for batch_idx, (images, y_trues) in enumerate(data_loader):

        images = images.to(device)
        # Use your model to make predictions for the batch of images
        output = v16mm(images)
        y_preds = torch.argmax(output, dim=1)

        print(f'Evaluating {batch_idx+1}/{len(data_loader)} batch of samples...')

        # Loop through the predicted labels and check if they match the true labels
        for i, pred_label in enumerate(y_preds):
            y_true = y_trues[i].item()
            pred_label = pred_label.item()

            if pred_label == y_true:
                continue

            misclassified_count += 1
            print(f"[{misclassified_count}-th misclassified sample] {i}-th sample's y-true is "
                  f"{y_true} but y-hat is {pred_label}", end='')
            # If the predicted label is incorrect, save the misclassified image to the file system
            folder_name = str(y_true)
            output_dir = os.path.join(
                settings['diagnostics']['misclassified'], folder_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                dataset.samples[batch_idx * batch_size + i][0].split("/")[-1]
            )
            print(f', sample copied from {dataset.samples[batch_idx * batch_size + i][0]} to {output_path}')
            Image.open(dataset.samples[batch_idx * batch_size + i][0]).save(output_path)
            # if ('20230520-163835_走廊_00016.jpg' in dataset.samples[batch_idx * batch_size + i][0]):
            # breakpoint()

print(f'{misclassified_count} misclassified samples found')
