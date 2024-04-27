from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.cuda


class VGG16MinusMinus(nn.Module):
    dropout = 0.0
    num_classes = -1

    def __init__(
            self, num_classes: int, target_image_size: Tuple[int, int],
            fc_layer_neuron_count: int = 4096
        ) -> None:
        self.num_classes = num_classes
        self.target_image_size = target_image_size

        super(VGG16MinusMinus, self).__init__()
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU())
        # self.layer4 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm2d(128),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer6 = nn.Sequential(
        #    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm2d(256),
        #    nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer8 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU())
        # self.layer9 = nn.Sequential(
        #    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm2d(512),
        #    nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer11 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU())
        self.layer12 = nn.Sequential(
             nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer13 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(
                int(self.target_image_size[0] / 64) *
                int(self.target_image_size[1] / 64) * 128,
                fc_layer_neuron_count
            ),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(fc_layer_neuron_count, fc_layer_neuron_count),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(fc_layer_neuron_count, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.layer5(x)
        # x = self.layer6(x)
        x = self.layer7(x)
        # x = self.layer8(x)
        # x = self.layer9(x)
        x = self.layer10(x)
        # x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def vggnet(config: Dict[str, Any]):
    return VGG16MinusMinus(
        config['model']['num_classes'],
        (
            config['model']['input_image_size']['height'],
            config['model']['input_image_size']['width']
        ),
        int(config['model']['fully_connected_layer_neuron_count'])
    )
