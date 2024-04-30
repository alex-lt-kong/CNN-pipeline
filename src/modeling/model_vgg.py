from typing import Any, Dict
from torchvision.models.vgg import vgg11 as _vgg11
from torchvision.models.vgg import vgg16 as _vgg16
from torchvision.models.vgg import vgg19 as _vgg19


def vgg11(config: Dict[str, Any], dropout_rate: float):
    return _vgg11(dropout=dropout_rate)

def vgg16(config: Dict[str, Any], dropout_rate: float):
    return _vgg16(dropout=dropout_rate)

def vgg19(config: Dict[str, Any], dropout_rate: float):
    return _vgg19(dropout=dropout_rate)