from collections import OrderedDict

from torch import nn

from modeling import registry
from .resnet import *
from .ProHG import *

def build_backbone(cfg):
    assert cfg.BACKBONE.BODY in registry.BACKBONES, \
        "cfg.BACKBONE.BODY: {} are not registered in registry".format(
                cfg.BACKBONE.BODY)
    return registry.BACKBONES[cfg.BACKBONE.BODY](cfg)
