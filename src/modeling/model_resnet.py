from typing import Any, Dict
from torchvision.models.resnet import resnet18 as _resnet18
from torchvision.models.resnet import resnet34 as _resnet34
from torchvision.models.resnet import resnet50 as _resnet50

def resnet18(configs: Dict[str, Any], dropout_rate: float):    
    return _resnet18(progress=False, num_classes=configs['model']['num_classes'])

def resnet34(configs: Dict[str, Any], dropout_rate: float):    
    return _resnet34(progress=False, num_classes=configs['model']['num_classes'])

def resnet50(configs: Dict[str, Any], dropout_rate: float):    
    return _resnet50(progress=False, num_classes=configs['model']['num_classes'])

