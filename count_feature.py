import os

import numpy
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import torchvision.transforms as transforms
import torchvision as tvm
from torch.utils.data import DataLoader
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
import models_test

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        out1, out = self.submodule(x)
        outputs["out1"] = out1

        return outputs


def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def getData():  
    #cifar10
    # transform = transforms.Compose([
    #     transforms.RandomCrop(32, 4),
    #     # transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),  
    #     transforms.RandomHorizontalFlip(),  
    #     transforms.ToTensor(),  
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],  
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # transform_val = transforms.Compose([
    #     transforms.ToTensor(),  
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],  
    #                          std=[0.229, 0.224, 0.225])
    # ])

    #ImageNet
    transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])
        
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])

    ])
    trainset = tv.datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)  
    testset = tv.datasets.CIFAR10(root='../data', train=False, transform=transform_val, download=True)  

    train_loader = DataLoader(trainset, batch_size=8, shuffle=True)  
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)  
    return train_loader, test_loader, trainset
def get_feature(net, model_name, load_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader, trainset = getData()
    if load_path != '':
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        best_prec1 = 0#checkpoint['best_prec1']
        net.load_state_dict(checkpoint['state_dict'])
        best_epoch = checkpoint['best_epoch']
    print('testing {}, total epoch: {}, best epoch: {}, best acc: {}'.format(model_name, start_epoch, best_epoch,
                                                                                    best_prec1))
    dst = './features/'+model_name  
    make_dirs(dst)
    net.to(device)
    net.eval()  
    f_sum = None
    therd_size = 256  
    with torch.no_grad():
        for inputs_cpu, labels_cpu in test_loader:
            img = inputs_cpu.to(device)
            exact_list = None

            myexactor = FeatureExtractor(net, exact_list)
            outs = myexactor(img)
            if f_sum == None:
                f_sum = {k:torch.zeros((o.shape[2], o.shape[3]), device=device) for k, o in outs.items()}
                #length = {k:0 for k, o in outs.items()}
            for k, v in outs.items():
                features = v[0]
                features = torch.sum(features, dim=0)
                features = features / features.max()
                f_sum[k] += features
                
        for k, feature_img in f_sum.items():
            dst_file = os.path.join(dst, str(k) + '.png')
            feature_img = feature_img / feature_img.max() * 230
            feature_img = feature_img.data.cpu().numpy().astype(np.uint8)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst, str(k) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)
            cv2.imwrite(dst_file, feature_img)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print('using {} device'.format(device)) 
    model_names = [
    
        # 'imagenet_ResNet18_Lt',
        # 'imagenet_ResNet18_ang',
                   ]
    load_paths = ['./save_weights/weights_imagenet/modelBest_' + name +'.pth' for name in model_names]
    for model_name, load_path in zip(model_names, load_paths):
        net = getattr(models_test, model_name)().to(device)
        get_feature(net, model_name, load_path)
