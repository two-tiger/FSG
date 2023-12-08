import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from utils.train_tools import progress_bar, is_int, train, test

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNetReduce(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNetReduce, self).__init__()
        self.in_channels = 16  # 注意，开始的通道数减少了，适应更浅的网络
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# official
def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

# unofficial
def resnet20(num_classes):
    return ResNetReduce(BasicBlock, [2, 2, 2], num_classes=num_classes)

def resnet32(num_classes):
    return ResNetReduce(BasicBlock, [5, 5, 5], num_classes=num_classes)

def resnet56(num_classes):
    return ResNetReduce(BasicBlock, [9, 9, 9], num_classes=num_classes)

parser = argparse.ArgumentParser(description='Full precision of ResNet train')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--retrain', '-r', default=None, help='Retrain from a pre-trained model')
parser.add_argument('--model', '-m', type=str, default='ResNet18', help='Model arch')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--optimizer', '-o', type=str, default='SGD', help='Optimizer')
parser.add_argument('--adjust', '-ad', default='adaptive', type=str, help='Training strategy')
parser.add_argument('--epoch', default=200, type=int, help='Max train epoch')
parser.add_argument('--pretrain', '-pre', type=str, default=None, help='Test with pretrain')
args = parser.parse_args()

# 数据预处理
if args.dataset == 'CIFAR10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    num_classes = 10

elif args.dataset == 'CIFAR100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./datasets/CIFAR100', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./datasets/CIFAR100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    num_classes = 100


use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
model_name = args.model
dataset_name = args.dataset

save_root = './full_precision/%s-%s' %(model_name, dataset_name)
if not os.path.exists(save_root):
    os.makedirs(save_root)


# 实例化模型
if model_name == 'ResNet18':
    net = resnet18(num_classes=num_classes)
elif model_name == 'ResNet34':
    net = resnet34(num_classes=num_classes)
elif model_name == 'ResNet50':
    net = resnet50(num_classes=num_classes)
elif model_name == 'ResNet101':
    net = resnet101(num_classes=num_classes)
elif model_name == 'ResNet152':
    net = resnet152(num_classes=num_classes)
elif model_name == 'ResNet20':
    net = resnet20(num_classes=num_classes)
elif model_name == 'ResNet32':
    net = resnet32(num_classes=num_classes)
elif model_name == 'ResNet56':
    net = resnet56(num_classes=num_classes)
else:
    raise NotImplementedError

if args.pretrain is not None:
    pretrain = True
    # pretrain_path = 'Results/ResNet56-CIFAR100/ResNet56-CIFAR100-pretrain.pth'
    pretrain_path = args.pretrain
    net.load_state_dict(torch.load(pretrain_path, map_location=device), strict=False)
else:
    pretrain = False
    
if args.retrain is not None:
    # Load pretrain ckpt.
    print('==> Retrain from pre-trained model %s' % args.retrain)
    ckpt = torch.load('%s/checkpoint/%s_ckpt.t7' % (save_root, model_name))
    net = ckpt['net']
    start_epoch = ckpt['epoch']
    best_test_acc = ckpt['acc']
else:
    start_epoch = 0
    best_test_acc = 0

if use_cuda:
    net.cuda()
    cudnn.benchmark = True

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
if args.optimizer in ['Adam', 'adam']:
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if not pretrain:
    # Begin Training
    ascent_count = 0
    min_train_loss = 1e9
    max_training_epoch = args.epoch

    for epoch in range(start_epoch, start_epoch + max_training_epoch):

        print('Epoch: [%3d]' % epoch)
        train_loss, train_acc = train(net, trainloader, optimizer, criterion, device)
        test_loss, test_acc = test(net, testloader, criterion, device)

        # Save checkpoint.
        if test_acc > best_test_acc:
            print('Saving...')
            state = {
                'net': net.modules if use_cuda else net,
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('%s/checkpoint' % (save_root)):
                os.mkdir('%s/checkpoint' % (save_root))
            torch.save(state, '%s/checkpoint/%s_%s_ckpt.t7' % (save_root, model_name, args.optimizer))
            best_test_acc = test_acc
            torch.save(net.state_dict(), '%s/%s-%s-%s-pretrain.pth' % (save_root, model_name, dataset_name, args.optimizer))

        if args.adjust == 'adaptive':

            if train_loss < min_train_loss:
                min_train_loss = train_loss
                ascent_count = 0
            else:
                ascent_count += 1

            print('Current Loss: %.3f [%.3f], ascent count: %d' % (train_loss, min_train_loss, ascent_count))

            if ascent_count >= 3:
                optimizer.param_groups[0]['lr'] *= 0.1
                ascent_count = 0
                if (optimizer.param_groups[0]['lr']) < (args.lr * 1e-3):
                    print('Learning rate has decreased by three orders of magnitude!')
                    break

else:
    test_loss, test_acc = test(net, testloader, criterion, device)
    print('Test Loss: ', test_loss, 'Test ACC: ', test_acc)