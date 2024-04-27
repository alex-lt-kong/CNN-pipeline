from typing import Any, Dict
from torchvision.models import mobilenetv3

def mobilenet_v3_small(config: Dict[str, Any]):    
    return mobilenetv3.mobilenet_v3_small(
        num_classes=config['model']['num_classes'], dropout=0.5
    )
