import torch
import models
import os
import cv2
import numpy as np

device = torch.device("cpu")

model_names = [
               'imagenet_ResNet18',
               'imagenet_ResNet34',
              ]
load_paths = ['./save_weights/weights_imagenet/modelBest_' + name +'.pth' for name in model_names]

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
therd_size = 256
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def visual_kernel(s):
    feature_img = k1.data.cpu().numpy()
    dst_file = os.path.join(dst, str(k1.shape[0])+'x'+ str(k1.shape[1])+ s + '.png')
    feature_img = normalization(feature_img)
    feature_img = (1-feature_img) * 255
    feature_img = cv2.applyColorMap(feature_img.astype(np.uint8), cv2.COLORMAP_AUTUMN)
    if feature_img.shape[0] < therd_size:
        tmp_file = os.path.join(dst, str(k1.shape[0])+'x'+ str(k1.shape[1]) + s + str(therd_size) + '.png')
        tmp_img = feature_img.copy()
        tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(tmp_file, tmp_img)
    cv2.imwrite(dst_file, feature_img)

for model_name, load_path in zip(model_names, load_paths):
    models.f = 1
    dst = './kernel_values/' + model_name
    make_dirs(dst)
    net = getattr(models, model_name)().to(device)
    if load_path != '':
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['state_dict'],)
        best_epoch = checkpoint['best_epoch']
    print('{}, best epoch: {}'.format(model_name, best_epoch))
    ksum = []
    num = []

    for name, para in net.named_parameters():
        if 'conv' not in name:
            continue
        k = para.data
        size = k.shape
        k = torch.abs(k.view(-1, size[2], size[3]))
        k = torch.sum(k, dim=0)
        inter = k.view(-1)
        maxvalue, inds = torch.max(inter, dim=0)
        k = k / maxvalue
        F = False
        if name[:5] == 'layer' and k.shape[0] == 3:

            j = int(name[5])
            if ksum.__len__() < j:
                ksum.append(torch.zeros((size[2], size[3])))
                num.append(0)

            ksum[j - 1] += k
            num[j - 1] += 1

    for i, (k1, n1) in enumerate(zip(ksum, num)):
        k1 /= n1
        print(model_name, 'shape——padding', k1.shape,': ')#, k1)
        print(k1)
        k = np.sort(k1, None)
        if k1.shape[0] != 3:
            continue
        print('layer', i + 1, ':max:', k1.max(), 'min:', k1.min(), 'avg:', k1.mean(), '\n', 'max_index:', torch.argmax(k1)+1, "n1-n2:", k[-1]-k[-2])#, 'std:', k1.std())
    print('---------------------------------------')

