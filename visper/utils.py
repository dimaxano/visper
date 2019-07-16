import gin

def add_external_configurables():
    #pass
    from .models.utils import get_class_weight
    gin.external_configurable(get_class_weight)

