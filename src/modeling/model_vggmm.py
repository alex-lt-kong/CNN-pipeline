from typing import Any, Dict, Tuple

import torch
import torch.nn as nn



class VGG16MinusMinus(nn.Module):
    dropout = 0.75
    num_classes = -1

    def __init__(self, num_classes: int, target_image_size: Tuple[int, int], dropout_rate: float) -> None:
        self.num_classes = num_classes
        self.dropout = dropout_rate
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
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(
                int(self.target_image_size[0] / 64) *
                int(self.target_image_size[1] / 64) * 256,
                int(4096 / 41)
            ),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(int(4096 / 41), int(4096 / 41)),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(int(4096 / 41), num_classes))

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


class VGG16ThreeMinuses(nn.Module):
    dropout = 0.75
    num_classes = -1

    def __init__(
        self, num_classes: int, target_image_size: Tuple[int, int], dropout_rate: float
    ) -> None:
        self.num_classes = num_classes
        self.dropout = dropout_rate
        self.target_image_size = target_image_size

        super(VGG16ThreeMinuses, self).__init__()

        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer10 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer12 = nn.Sequential(
             nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer13 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(
                int(self.target_image_size[0] / 64) *
                int(self.target_image_size[1] / 64) * 256,
                50
            ),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(50, 50),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(50, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer5(x)
        x = self.layer7(x)
        x = self.layer10(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class VGG16FourMinuses(nn.Module):
    dropout = 0.75
    num_classes = -1

    def __init__(
        self, num_classes: int, target_image_size: Tuple[int, int], dropout_rate: float
    ) -> None:
        self.num_classes = num_classes
        self.dropout = dropout_rate
        self.target_image_size = target_image_size

        super(VGG16FourMinuses, self).__init__()

        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer10 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
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
                50
            ),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(50, 50),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(50, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer5(x)
        x = self.layer7(x)
        x = self.layer10(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class VGG16FiveMinuses(nn.Module):
    dropout = 0.5
    num_classes = -1

    def __init__(
        self, num_classes: int, target_image_size: Tuple[int, int], dropout_rate: float
    ) -> None:
        self.num_classes = num_classes
        self.dropout = dropout_rate
        self.target_image_size = target_image_size

        super(VGG16FiveMinuses, self).__init__()

        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer10 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer12 = nn.Sequential(
             nn.Conv2d(36, 48, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(48),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer13 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(
                int(self.target_image_size[0] / 64) *
                int(self.target_image_size[1] / 64) * 64,
                32
            ),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(32, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer5(x)
        x = self.layer7(x)
        x = self.layer10(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class VGG16SixMinuses(nn.Module):
    dropout = 0.5
    num_classes = -1

    def __init__(
        self, num_classes: int, target_image_size: Tuple[int, int], dropout_rate: float
    ) -> None:
        self.num_classes = num_classes
        self.dropout = dropout_rate
        self.target_image_size = target_image_size

        super(VGG16SixMinuses, self).__init__()

        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(12, 18, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer10 = nn.Sequential(
            nn.Conv2d(18, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer12 = nn.Sequential(
             nn.Conv2d(24, 30, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(30),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer13 = nn.Sequential(
            nn.Conv2d(30, 36, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(
                int(self.target_image_size[0] / 64) *
                int(self.target_image_size[1] / 64) * 36,
                24
            ),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(24, 24),
            nn.Dropout(self.dropout),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(24, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer5(x)
        x = self.layer7(x)
        x = self.layer10(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def vggmm(configs: Dict[str, Any], dropout_rate: float):    
    return VGG16MinusMinus(
        configs['model']['num_classes'],
        (configs['model']['input_image_size']['width'], configs['model']['input_image_size']['height']),
        dropout_rate
    )


def vgg3m(configs: Dict[str, Any], dropout_rate: float):    
    return VGG16ThreeMinuses(
        configs['model']['num_classes'],
        (configs['model']['input_image_size']['width'], configs['model']['input_image_size']['height']),
        dropout_rate
    )


def vgg4m(configs: Dict[str, Any], dropout_rate: float):    
    return VGG16FourMinuses(
        configs['model']['num_classes'],
        (configs['model']['input_image_size']['width'], configs['model']['input_image_size']['height']),
        dropout_rate
    )


def vgg5m(configs: Dict[str, Any], dropout_rate: float):    
    return VGG16FiveMinuses(
        configs['model']['num_classes'],
        (configs['model']['input_image_size']['width'], configs['model']['input_image_size']['height']),
        dropout_rate
    )
    

def vgg6m(configs: Dict[str, Any], dropout_rate: float):    
    return VGG16SixMinuses(
        configs['model']['num_classes'],
        (configs['model']['input_image_size']['width'], configs['model']['input_image_size']['height']),
        dropout_rate
    )
