#!/bin/env python3

import torch
from torchstat import stat
import torchvision.models as models
from torchsummary import summary
from ptflops import get_model_complexity_info

def get_param(net, feature_map):
  stat(net, feature_map)
  
feature_map = (3, 224, 224)

shuffleNet = models.ShuffleNetV2()
print(models.ShuffleNetV2())
get_param(shuffleNet, feature_map)
