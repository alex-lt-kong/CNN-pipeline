from typing import Any, Dict
from torchvision.models import squeezenet1_1 as _squeezenet1_1

def squeezenet1_1(config: Dict[str, Any], dropout_rate: float):    
    return _squeezenet1_1(
        num_classes=config['model']['num_classes'], dropout=dropout_rate
    )
