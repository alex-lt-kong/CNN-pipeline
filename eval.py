from PIL import Image
from typing import List, Any, Dict
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

import argparse
import json
import model
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)
v16mm = model.VGG16MinusMinus(2)
v16mm.to(device)

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

settings = read_config_file()
v16mm.load_state_dict(torch.load(settings['model']['model']))

train_loader, val_loader = model.get_data_loaders(
        v16mm.transforms, settings["dataset"]["path"],
        settings['dataset']['validation_split_seed']
    )

# Define the batch size
batch_size = 16
misclassified_count = 0

# Classify the image using the model
with torch.no_grad():
  # Loop through the images in batches
    for batch_idx, (images, y_trues) in enumerate(train_loader):
        print(f'{batch_idx}/{len(train_loader)}')
        images = images.to(device)
        # Use your model to make predictions for the batch of images
        output = v16mm(images)
        _, predicted = torch.max(output, 1)
        
        # Loop through the predicted labels and check if they match the true labels
        for j, pred_label in enumerate(predicted):
            y_true = y_trues[j].item()
            pred_label = pred_label.item()
            if pred_label != y_true:
                misclassified_count += 1
                print(f"[{misclassified_count}] {j}-th sample's y-true is "
                      f"{y_true} but y-hat is {pred_label}", end='')
                # If the predicted label is incorrect, save the misclassified image to the file system
                folder_name = str(y_true)
                output_dir = os.path.join("/tmp/misclassified", folder_name)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, train_loader.data.samples[batch_idx * batch_size + j][0].split("/")[-1])
                print(f', sample saved to {output_path}')
                Image.open(train_loader.data.samples[batch_idx * batch_size + j][0]).save(output_path)
