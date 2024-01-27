import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import os
import datetime
import numpy as np
import models
import distributed_utils as utils

load_path = ''  
save_dir = 'save_weights/weights_cifar'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
BATCH_SIZE = 64  
LR = 0.1  # learning rate
EPOCH = 200  
warmup = True
warmup_epoch = 10

def getData():  
    transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])
    ])
    trainset = tv.datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)  
    testset = tv.datasets.CIFAR10(root='../data', train=False, transform=transform_val, download=True)  

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)  
    return train_loader, test_loader, trainset


def train(net, model_name, load_path):
    trainset_loader, testset_loader, trainset = getData()  
    net.to(device)

    start_epoch = 0
    best_prec1 = 0
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
                                                        milestones=[80, 140])

    if load_path != '':
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_epoch = checkpoint['best_epoch']
    print('traing {}, starting from epoch: {}, best epoch: {}, best acc: {}'.format(model_name, start_epoch, best_epoch, best_prec1))
    # Train the model
    for epoch in range(start_epoch, EPOCH):
        sum_loss = 0.
        total = 0.
        accuracy = 0.
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
            sum_loss += loss.item()
            total += labels.size(0)
            accuracy += (predicted == labels).sum()
            end_time = datetime.datetime.now()
            if step % print_per == 0:
                
                print("epoch %d | step%d/%d, lr:%f, time:%s, %.2fms/item, loss = %.4f, the accuracy now is %.3f %%." % (
                    epoch, step, len(trainset_loader), optimizer.param_groups[0]["lr"],
                    str((end_time - begin_time) * (len(trainset_loader))).split('.')[0],
                    (end_time.timestamp() - begin_time.timestamp()) * 1000 / BATCH_SIZE,
                    sum_loss / (step + 1), 100. * accuracy.cpu().numpy() / total))
            begin_time = datetime.datetime.now()
        lr_scheduler.step()
        
        # Test
        if epoch > int(EPOCH/2):
            acc = test(net, testset_loader)
            is_best = acc > best_prec1
            best_prec1 = max(acc, best_prec1)

            if is_best:
                best_epoch = epoch
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'best_epoch': best_epoch
                }, os.path.join(save_dir, 'modelBest_{}.pth'.format(model_name)))
            print("___________________________________________________")
            print("epoch %d : training accuracy = %.4f %%" % (epoch, 100 * acc))
            print('best_epoch: {} acc: {}'.format(best_epoch, best_prec1))
            print("---------------------------------------------------")

        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_epoch': best_epoch
        }, os.path.join(save_dir, 'modelLatest_{}.pth'.format(model_name)))



    print('Finished Training ' + model_name)
    with open(save_dir + '/testout_cifar.txt', 'a') as f :
        f.write(model_name + ': last acc:{}, best_epoch: {}, best acc: {}\n'.format(acc, best_epoch, best_prec1) )
    return


def test(net, testdata):
    correct, total = .0, .0
    net.eval()
    with torch.no_grad():
        for inputs_cpu, labels_cpu in testdata:
            inputs = inputs_cpu.to(device)
            labels = labels_cpu.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    return float(correct.cpu().numpy()) / total

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print('using {} device'.format(device))
    model_names = [
                   # 'resnet20_Lt1_Lt2',
                   # 'resnet20_ang1_ang2',
                    # 'resnet20_sdc1x1s_Lt',
                    # 'resnet20_sdc1x1s_ang',
                   ]
    load_paths = ['']
    load_paths += ['' for i in range(model_names.__len__()-1)]
    for model_name, load_path in zip(model_names, load_paths):
        net = getattr(models, model_name)().to(device)
        train(net, model_name, load_path)

