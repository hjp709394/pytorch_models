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
import logging

from models import *
from utils import progress_bar
from torch.autograd import Variable


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--data_path', type=str, help='Path for cifar100')
    parser.add_argument('--log_path', type=str, help='Path for logging')
    parser.add_argument('--dataset', type=str, help='cifar10 | cifar100')
    parser.add_argument('--model_name', type=str, default='ResNet18', help='Model structure name')
    args = parser.parse_args()
    return args


def prepare_log(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if log_path == None:
        ch = logging.StreamHandler()
    else:
        ch = logging.FileHandler(log_path)
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s - %(name)s - %(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def preapre_data(data_path, dataset, logger=None):
    if dataset != 'cifar10' and dataset != 'cifar100':
        print("Invalid dataset: {}".format(dataset))
        sys.exit(1)

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

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    classes_cifar100 = (
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

    if dataset == 'cifar10':
        classes = classes_cifar10
    elif dataset == 'cifar100':
        classes = classes_cifar100

    return trainloader, transform_train, testloader, transform_test, classes


def prepare_model(num_classes=-1, model_name='ResNet18', logger=None):
    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')

    message = []
    if os.path.exists('./checkpoint/ckpt.t7'):
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net = checkpoint['net']
        start_epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        message.append("Restoring from checkpoint of epoch {} with acc {}".format(start_epoch, acc))
    else:
        if model_name == 'VGG19':
            net = VGG('VGG19')
        elif model_name == 'ResNet18':
            net = ResNet18(num_classes=num_classes)
        elif model_name == 'ExpNetV1':
            net = ExpNetV1(num_classes=num_classes)
        elif model_name == 'PreActResNet18':
            net = PreActResNet18()
        elif model_name == 'GoogLeNet':
            net = GoogLeNet()
        elif model_name == 'DenseNet121':
            net = DenseNet121()
        elif model_name == 'ResNeXt29_2x64d':
            net = ResNeXt29_2x64d()
        elif model_name == 'MobileNet':
            net = MobileNet()
        elif model_name == 'MobileNetV2':
            net = MobileNetV2()
        elif model_name == 'DPN92':
            net = DPN92()
        elif model_name == 'ShuffleNetG2':
            net = ShuffleNetG2()
        elif model_name == 'SENet18':
            net = SENet18()
        start_epoch = -1
        acc = 0.
        message.append("Training from scratch")

    for mes in message:
        if logger is not None:
            logger.info(mes)
        else:
            print(mes)

    return net, start_epoch + 1, acc


def train(net, criterion, optimizer, trainloader, epoch, use_cuda, logger=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    logging_template = 'Loss: {:.3f} | Acc: {:.3f}% ({}/{}) | LR: {}'
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

        progress_bar(batch_idx, len(trainloader), logging_template.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total, lrs))

    if logger is not None:
        logger.info("Training epoch {:5} | ".format(epoch) + logging_template.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total, lrs))


def test(net, criterion, optimizer, testloader, epoch, use_cuda, logger=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    logging_template = 'Loss: {:.3f} | Acc: {:.3f}% ({}/{})'
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

        progress_bar(batch_idx, len(testloader), logging_template.format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))

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

    if logger is not None:
        logger.info("Testing  epoch {:5} | ".format(epoch) + logging_template.format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))


def main():

    args = parse_args()

    logger = prepare_log(args.log_path)

    use_cuda = torch.cuda.is_available()

    # prepare data
    trainloader, transform_train, testloader, transform_test, classes = preapre_data(args.data_path, args.dataset)

    # prepare model
    net, start_epoch, _ = prepare_model(num_classes=len(classes), model_name=args.model_name, logger=logger)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 250], gamma=0.1)

    for epoch in range(start_epoch, 350):
        lr_scheduler.step(epoch=epoch)
        train(net, criterion, optimizer, trainloader, epoch, use_cuda, logger)
        test(net, criterion, optimizer, testloader, epoch, use_cuda, logger)


if __name__ == '__main__':
    main()
