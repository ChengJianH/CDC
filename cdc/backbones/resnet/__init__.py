

from .resnet import resnet18 as default_resnet18
from .resnet import resnet34 as default_resnet34
from .resnet import resnet50 as default_resnet50

def resnet18(method, *args, **kwargs):
    return default_resnet18(*args, **kwargs)

def resnet34(method, *args, **kwargs):
    return default_resnet34(*args, **kwargs)

def resnet50(method, *args, **kwargs):
    return default_resnet50(*args, **kwargs)


__all__ = ["resnet18", "resnet34", "resnet50"]
