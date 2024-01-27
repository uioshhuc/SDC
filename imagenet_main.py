import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import os
import datetime
import numpy as np
import models
import distributed_utils as utils

load_path = '' 
save_dir = './save_weights/weights_imagenet'
BATCH_SIZE = 128  
LR = 0.05  # learning rate
EPOCH = 120 
warmup = True
warmup_epoch = 1

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def getData():  
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
    trainset = torchvision.datasets.ImageFolder(
        root='C:/Users\data\ImageNet_ILSVRC2012/train',
        transform=transform)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    valset = torchvision.datasets.ImageFolder(
        root='C:/Users\data\ImageNet_ILSVRC2012/val',
        transform=transform_val)

    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, val_loader, trainset


def train(net, model_name, load_path):
    trainset_loader, valset_loader, trainset = getData()  
    net.to(device)

    start_epoch = 0
    best_top1 = 0
    best_top5 = 0
    best_epoch = 0
    print_per = 50

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().to(device)  
    optimizer = torch.optim.SGD(net.parameters(), LR,
                                momentum=0.9, weight_decay=1e-4)#
    if warmup is True:  
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000*warmup_epoch, warmup_epoch*(len(trainset_loader) - 1))
        lr_scheduler_2 = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[30, 60, 90])

    if load_path != '':
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        best_top5 = checkpoint['best_top5']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_epoch = checkpoint['best_epoch']
    print('traing {}, starting from epoch: {}, best epoch: {}, best top1: {}, best top5: {}'.format(model_name, start_epoch, best_epoch, best_top1, best_top5))
    # Train the model
    for epoch in range(start_epoch, EPOCH):
        sum_loss = 0.
        total = 0.
        top1_train = 0.
        top5_train = 0.
        net.train()
        begin_time = datetime.datetime.now()
        for step, (inputs_cpu, labels_cpu) in enumerate(trainset_loader):

            inputs = inputs_cpu.to(device)
            labels = labels_cpu.to(device)
            output = net(inputs)
            loss = criterion(output, labels)  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            if lr_scheduler_2 is not None and start_epoch < warmup_epoch and epoch < warmup_epoch:
                lr_scheduler_2.step()

            _, predicted = torch.max(output, 1)  
            _, predicted_top5 = torch.sort(output, 1, descending=True)
            top5_train += (predicted_top5[:, :5] == labels.unsqueeze(-1).repeat(1, 5)).sum()
            sum_loss += loss.item()
            total += labels.size(0)
            top1_train += (predicted == labels).sum()
            end_time = datetime.datetime.now()
            if step % print_per == 0:
                print("epoch %d | step%d/%d, lr:%f, time:%s, %.2fms/item, loss = %.4f, the accuracy now is: top1 acc=%.3f%%, top5 acc=%.3f%% " % (
                    epoch, step, len(trainset_loader), optimizer.param_groups[0]["lr"],
                    str((end_time - begin_time) * (len(trainset_loader))).split('.')[0],
                    (end_time.timestamp() - begin_time.timestamp()) * 1000 / BATCH_SIZE,
                    sum_loss / (step + 1), 100. * top1_train.cpu().numpy() / total, 100. * top5_train.cpu().numpy() / total))
            begin_time = datetime.datetime.now()
        lr_scheduler.step()

        # Test
        if epoch > int(EPOCH/2):
            top1, top5 = test(net, valset_loader)
            is_best1 = top1 > best_top1
            best_top1 = max(top1, best_top1)

            if is_best1:
                best_top5 = top5
                best_epoch = epoch
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'best_top1': best_top1,
                    'best_top5': best_top5,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'best_epoch': best_epoch
                }, os.path.join(save_dir, 'modelBest_{}.pth'.format(model_name)))
            print("___________________________________________________")
            print("epoch %d : val accuracy: top1=%.4f %%, top5=%.4f %%" % (epoch, 100 * top1, 100 * top5))
            print('best_epoch: %d acc: top1=%.4f %%, top5=%.4f %%' % (best_epoch, 100 * best_top1, 100 * best_top5))
            print("---------------------------------------------------")

        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_top1': best_top1,
            'best_top5': best_top5,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_epoch': best_epoch
        }, os.path.join(save_dir, 'modelLatest_{}.pth'.format(model_name)))



    print('Finished Training ' + model_name)
    with open('save_weights/weights_imagenet/testout_imagenet.txt', 'a') as f :
        f.write(model_name + ': last top1:{},  last top5:{}, best_epoch: {}, best top1 acc: {}, best top5 acc: {}\n'.format(top1, top5, best_epoch, best_top1, best_top5) )
    return


def test(net, valdata):
    correct, total, top5 = 0., 0., 0.
    net.eval()  
    with torch.no_grad():
        for inputs_cpu, labels_cpu in valdata:
            inputs = inputs_cpu.to(device)
            labels = labels_cpu.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            _, predicted_top5 = torch.sort(outputs, 1, descending=True)
            top5 += (predicted_top5[:, :5] == labels.unsqueeze(-1).repeat(1, 5)).sum()
            total += labels.size(0)
            correct += (predicted == labels).sum()
    return float(correct.cpu().numpy()) / total, float(top5.cpu().numpy()) / total

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print('using {} device'.format(device))    
    model_names = [#"imagenet_ResNet18",
                   #"imagenet_ResNet34_Lt",
                   ]
    load_paths = ['save_weights/weights_imagenet/modelLatest_imagenet_ResNet34_Lt.pth']
    load_paths += ['' for i in range(model_names.__len__()-1)]
    for model_name, load_path in zip(model_names, load_paths):
        models.f = 1
        net = getattr(models, model_name)().to(device)
        train(net, model_name, load_path)

