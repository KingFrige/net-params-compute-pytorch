#!/bin/env python3

import torch
from torchstat import stat
import torchvision.models as models
from torchsummary import summary
from ptflops import get_model_complexity_info

def get_param(net, feature_map):
  stat(net, feature_map)
  
  macs, params = get_model_complexity_info(net, feature_map, as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
  
  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  summary(net.to(device), feature_map)
  
  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  total = sum([param.nelement() for param in net.parameters()])
  print("Number of parameter: %.2fM" % (total / 1e6))
  
  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
  
  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

feature_map = (3, 224, 224)

resnet50 = models.resnet50()
print(models.resnet50())
get_param(resnet50, feature_map)
