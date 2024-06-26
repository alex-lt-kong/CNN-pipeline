from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

# Original VGG16 implementation from here: https://blog.paperspace.com/vgg-from-scratch-pytorch/


class VGG16TwoMinuses(nn.Module):
    dropout = 0.75
    num_classes = -1

    def __init__(
        self, num_classes: int, target_image_size: Tuple[int, int],
        dropout_rate: float
    ) -> None:
        self.num_classes = num_classes
        self.dropout = dropout_rate
        self.target_image_size = target_image_size

        super(VGG16TwoMinuses, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
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
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 50),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(50, 50),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(50, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer5(x)
        x = self.layer7(x)
        x = self.layer10(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
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
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 50),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(50, 50),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(50, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer5(x)
        x = self.layer7(x)
        x = self.layer10(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
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
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 32),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer5(x)
        x = self.layer7(x)
        x = self.layer10(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
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
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(36 * 7 * 7, 24),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(24, 24),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(24, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer5(x)
        x = self.layer7(x)
        x = self.layer10(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16SevenMinuses(nn.Module):
    dropout = 0.5
    num_classes = -1

    def __init__(
        self, num_classes: int, target_image_size: Tuple[int, int], dropout_rate: float
    ) -> None:
        self.num_classes = num_classes
        self.dropout = dropout_rate
        self.target_image_size = target_image_size

        super(VGG16SevenMinuses, self).__init__()

        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer10 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer12 = nn.Sequential(
             nn.Conv2d(12, 18, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(18),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer13 = nn.Sequential(
            nn.Conv2d(18, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(24 * 7 * 7, 18),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(18, 18),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(18, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer5(x)
        x = self.layer10(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16EightMinuses(nn.Module):
    dropout = 0.5
    num_classes = -1

    def __init__(
        self, num_classes: int, target_image_size: Tuple[int, int], dropout_rate: float
    ) -> None:
        self.num_classes = num_classes
        self.dropout = dropout_rate
        self.target_image_size = target_image_size

        super(VGG16EightMinuses, self).__init__()

        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer10 = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer12 = nn.Sequential(
             nn.Conv2d(8, 10, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(10),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer13 = nn.Sequential(
            nn.Conv2d(10, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(12 * 7 * 7, 10),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(10, 10),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(10, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer5(x)
        x = self.layer10(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg2m(configs: Dict[str, Any], dropout_rate: float):    
    return VGG16TwoMinuses(
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


def vgg7m(configs: Dict[str, Any], dropout_rate: float):    
    return VGG16SevenMinuses(
        configs['model']['num_classes'],
        (configs['model']['input_image_size']['width'], configs['model']['input_image_size']['height']),
        dropout_rate
    )


def vgg8m(configs: Dict[str, Any], dropout_rate: float):    
    return VGG16EightMinuses(
        configs['model']['num_classes'],
        (configs['model']['input_image_size']['width'], configs['model']['input_image_size']['height']),
        dropout_rate
    )
