from typing import Any, Dict
from torchvision.models.resnet import _resnet, BasicBlock

def resnet10(configs: Dict[str, Any]):    
    return _resnet(BasicBlock, [1, 1, 1, 1], num_classes=configs['model']['num_classes'])

