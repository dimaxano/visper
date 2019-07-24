import gin
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

def add_external_configurables():
    #pass
    from .models.utils import get_class_weight
    gin.external_configurable(get_class_weight)

    gin.external_configurable(ReduceLROnPlateau)

    gin.external_configurable(Adam, blacklist=["params", "lr", "weight_decay"])


