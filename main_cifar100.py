'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--data_path', type=str, help='Path for cifar100')
    parser.add_argument('--log_path', type=str, help='Path for logging')
    args = parser.parse_args()
    return args


def prepare_log(log_path):
    if log_path is None:
        return sys.stdout
    else:
        return open(log_path, 'a')


def preapre_data(data_path):
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm')

    return trainloader, transform_train, testloader, transform_test, classes


def prepare_model():
    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')

    if os.path.exists('./checkpoint/ckpt.t7'):
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net = checkpoint['net']
        start_epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        print("Restoring from checkpoint of epoch {} with acc {}".format(start_epoch, acc))
    else:
        # net = VGG('VGG19')
        net = ResNet18(num_classes=100)
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        start_epoch = -1
        acc = 0.
        print("Traing from scratch")

    return net, start_epoch + 1, acc


def train(net, criterion, optimizer, trainloader, epoch, use_cuda, logf=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        lrs = ':'.join(["{:.6}".format(group['lr']) for group in optimizer.param_groups])

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %s'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, lrs))

    if logf is not None:
        logf.write("Train: Loss {:.6f} | ACC {:.6f} (%d/%d) | LR {s}\n".format(test_loss/(batch_idx+1), 100.*correct/total), correct, total, lrs)


def test(net, criterion, optimizer, testloader, epoch, use_cuda, logf=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

    print('Saving..')
    state = {
        'net': net.module if use_cuda else net,
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7')

    if logf is not None:
        logf.write("Test: Loss {:.6f} | ACC {:.6f}\n".format(test_loss/(batch_idx+1), 100.*correct/total))


def main():

    args = parse_args()

    logf = prepare_log(args.log_path)

    use_cuda = torch.cuda.is_available()

    # prepare data
    trainloader, transform_train, testloader, transform_test, classes = preapre_data(args.data_path)

    # prepare model
    net, start_epoch, _ = prepare_model()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 250], gamma=0.1)

    for epoch in range(start_epoch, 350):
        lr_scheduler.step(epoch=epoch)
        train(net, criterion, optimizer, trainloader, epoch, use_cuda)
        test(net, criterion, optimizer, testloader, epoch, use_cuda)

    logf.close()


if __name__ == '__main__':
    main()
