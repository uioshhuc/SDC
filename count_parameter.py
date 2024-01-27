import models
import torch
from torch.autograd import Variable
import numpy as np
from thop import profile
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print('using {} device'.format(device))

model_names = [
        
        # 'imagenet_ResNet18_Lt1_Lt2',
        # 'imagenet_ResNet18_ang1_ang2',
        # 'imagenet_ResNet34_Lt1_Lt2',
        # 'imagenet_ResNet34_ang1_ang2',
                   ]

for model_name in model_names:
    models.f = 1
    net = getattr(models, model_name)().to(device)
    dummy = torch.randn(1, 3, 224, 224).to(device) #cifar10: 1, 3, 32, 32 #ImageNet: 1, 3, 224, 224
    flops, params = profile(net, (dummy,))
    print(model_name+': Parameters: (new) %.2fM' % (
          params/1000000))
    print(model_name+': FLOPs: (new) %.2fM' % (
         flops/1000000))
