'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from meta_utils.meta_quantized_module import MetaQuantConv, MetaQuantLinear

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def conv3x3(in_planes, out_planes, stride=1, bitW=1, alpha=0.9):
    " 3x3 convolution with padding "
    return MetaQuantConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, bitW=bitW, alpha=alpha)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bitW = 1,
                 layer_idx=None, block_idx=None, layer_name_list = None, alpha=0.9):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, bitW=bitW, alpha=alpha)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bitW=bitW, alpha=alpha)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.layer_idx = layer_idx
        self.block_idx = block_idx
        # self.layer_name_list = layer_name_list
        layer_name_list.append(['layer%d.%d.conv1' %(self.layer_idx, self.block_idx),
                                ['layer%d' %self.layer_idx, self.block_idx, 'conv1']])
        layer_name_list.append(['layer%d.%d.conv2' % (self.layer_idx, self.block_idx),
                                ['layer%d' %self.layer_idx, self.block_idx, 'conv2']])

        assert (self.layer_idx is not None and self.block_idx is not None)

    def forward(self, x, quantized_type = None, meta_grad_dict = dict(), slow_grad_dict = None, lr=1e-3):
        residual = x

        if 'layer%d.%d.conv1' %(self.layer_idx, self.block_idx) in meta_grad_dict and slow_grad_dict is None:
            out = self.conv1(x = x, quantized_type = quantized_type,
                             meta_grad = meta_grad_dict['layer%d.%d.conv1' %(self.layer_idx, self.block_idx)],
                             slow_grad = None,
                             lr = lr)
        elif 'layer%d.%d.conv1' %(self.layer_idx, self.block_idx) in meta_grad_dict and 'layer%d.%d.conv1' %(self.layer_idx, self.block_idx) in slow_grad_dict:
            out = self.conv1(x = x, quantized_type = quantized_type,
                             meta_grad = meta_grad_dict['layer%d.%d.conv1' %(self.layer_idx, self.block_idx)],
                             slow_grad = slow_grad_dict['layer%d.%d.conv1' %(self.layer_idx, self.block_idx)],
                             lr = lr)
        else:
            out = self.conv1(x, quantized_type)
        out = self.bn1(out)
        out = self.relu(out)

        if 'layer%d.%d.conv2' %(self.layer_idx, self.block_idx) in meta_grad_dict and slow_grad_dict is None:
            out = self.conv2(x = out, quantized_type = quantized_type,
                             meta_grad = meta_grad_dict['layer%d.%d.conv2' %(self.layer_idx, self.block_idx)],
                             slow_grad = None,
                             lr = lr)
        elif 'layer%d.%d.conv2' %(self.layer_idx, self.block_idx) in meta_grad_dict and 'layer%d.%d.conv2' %(self.layer_idx, self.block_idx) in slow_grad_dict:
            out = self.conv2(x = out, quantized_type = quantized_type,
                             meta_grad = meta_grad_dict['layer%d.%d.conv2' %(self.layer_idx, self.block_idx)],
                             slow_grad = slow_grad_dict['layer%d.%d.conv2' %(self.layer_idx, self.block_idx)],
                             lr = lr)
        else:
            out = self.conv2(out, quantized_type)
        out = self.bn2(out)

        if self.downsample is not None:
            # residual = self.downsample(x)
            for module in self.downsample:
                if isinstance(module, MetaQuantConv):
                    if 'layer%d.%d.downsample.0' %(self.layer_idx, self.block_idx) in meta_grad_dict and slow_grad_dict is None:
                        residual = module(x = residual, quantized_type = quantized_type,
                             meta_grad = meta_grad_dict['layer%d.%d.downsample.0' %(self.layer_idx, self.block_idx)],
                             slow_grad = None,
                             lr = lr)
                    elif 'layer%d.%d.downsample.0' %(self.layer_idx, self.block_idx) in meta_grad_dict and 'layer%d.%d.downsample.0' %(self.layer_idx, self.block_idx) in slow_grad_dict:
                        residual = module(x = residual, quantized_type = quantized_type,
                             meta_grad = meta_grad_dict['layer%d.%d.downsample.0' %(self.layer_idx, self.block_idx)],
                             slow_grad = slow_grad_dict['layer%d.%d.downsample.0' %(self.layer_idx, self.block_idx)],
                             lr = lr)
                    else:
                        residual = module(residual, quantized_type)
                else:
                    residual = module(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bitW=1, layer_idx=None, block_idx=None, layer_name_list=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, bitW=bitW)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, bitW=bitW)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = MetaQuantConv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        self.layer_idx = layer_idx
        self.block_idx = block_idx
        layer_name_list.append(['layer%d.%d.conv1' %(self.layer_idx, self.block_idx),
                                ['layer%d' %self.layer_idx, self.block_idx, 'conv1']])
        layer_name_list.append(['layer%d.%d.conv2' % (self.layer_idx, self.block_idx),
                                ['layer%d' %self.layer_idx, self.block_idx, 'conv2']])
        layer_name_list.append(['layer%d.%d.conv3' % (self.layer_idx, self.block_idx),
                                ['layer%d' %self.layer_idx, self.block_idx, 'conv3']])
        
        assert (self.layer_idx is not None and self.block_idx is not None)

    def forward(self, x, quantized_type=None, meta_grad_dict=dict(), lr=1e-3):
        residual = x
        
        if 'layer%d.%d.conv1' %(self.layer_idx, self.block_idx) in meta_grad_dict:
            out = self.conv1(x=x, quantized_type=quantized_type, meta_grad=meta_grad_dict['layer%d.%d.conv1' %(self.layer_idx, self.block_idx)], lr=lr)
        else:
            out = self.conv1(x, quantized_type)
        out = self.bn1(out)
        out = self.relu(out)

        if 'layer%d.%d.conv2' %(self.layer_idx, self.block_idx) in meta_grad_dict:
            out = self.conv2(x=x, quantized_type=quantized_type, meta_grad=meta_grad_dict['layer%d.%d.conv2' %(self.layer_idx, self.block_idx)], lr=lr)
        else:
            out = self.conv2(x, quantized_type)
        out = self.bn2(out)
        out = self.relu(out)

        if 'layer%d.%d.conv3' %(self.layer_idx, self.block_idx) in meta_grad_dict:
            out = self.conv3(x=x, quantized_type=quantized_type, meta_grad=meta_grad_dict['layer%d.%d.conv3' %(self.layer_idx, self.block_idx)], lr=lr)
        else:
            out = self.conv3(x, quantized_type)
        out = self.bn3(out)

        if self.downsample is not None:
            # residual = self.downsample(x)
            for module in self.downsample:
                if isinstance(module, MetaQuantConv):
                    if 'layer%d.%d.downsample.0' %(self.layer_idx, self.block_idx) in meta_grad_dict:
                        residual = module(x = residual, quantized_type=quantized_type, meta_grad=meta_grad_dict['layer%d.%d.downsample.0' %(self.layer_idx, self.block_idx)], lr=lr)
                    else:
                        residual = module(residual, quantized_type)
                else:
                    residual = module(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, first_stride=1, num_classes=10, bitW=1, alpha=0.9):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.bitW = bitW
        self.layer_name_list = [['conv1', ['conv1']], ['fc', ['fc']]]
        # self.layer_name_list = []

        self.conv1 = MetaQuantConv(3, 16, kernel_size=3, stride=first_stride, padding=1, bias=False, bitW=bitW, alpha=alpha)
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=first_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], layer_idx=1, alpha=alpha)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, layer_idx=2, alpha=alpha)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, layer_idx=3, alpha=alpha)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc = MetaQuantLinear(64 * block.expansion, num_classes, bitW=bitW, alpha=alpha)
        # self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, layer_idx=0, alpha=0.9):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MetaQuantConv(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False, bitW=self.bitW, alpha=alpha),
                nn.BatchNorm2d(planes * block.expansion)
            )
            # self.layer_name_list.append('layer%d.0.downsample.0' %layer_idx)
            self.layer_name_list.append(['layer%d.0.downsample.0' %layer_idx,
                                         ['layer%d' % layer_idx, 0, 'downsample', 0]])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, layer_idx=layer_idx,
                            block_idx=0, bitW=self.bitW, layer_name_list=self.layer_name_list, alpha=alpha))
        self.inplanes = planes * block.expansion
        for blk_idx in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                layer_idx=layer_idx, block_idx=blk_idx, bitW=self.bitW,
                                layer_name_list=self.layer_name_list, alpha=alpha))

        return nn.Sequential(*layers)


    def forward(self, x, quantized_type = None, meta_grad_dict = dict(), slow_grad_dict = None, lr=1e-3):

        if 'conv1' in meta_grad_dict and slow_grad_dict is None:
            x = self.conv1(x = x, quantized_type = quantized_type, meta_grad = meta_grad_dict['conv1'], slow_grad = None, lr = lr)
        elif 'conv1' in meta_grad_dict and 'conv1' in slow_grad_dict:
            x = self.conv1(x = x, quantized_type = quantized_type, meta_grad = meta_grad_dict['conv1'], slow_grad = slow_grad_dict['conv1'], lr = lr)
        else:
            x = self.conv1(x, quantized_type)
        # x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                    x = block(x, quantized_type, meta_grad_dict, slow_grad_dict, lr)
                    

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)

        if 'fc' in meta_grad_dict and slow_grad_dict is None:
            x = self.fc(x = x, quantized_type = quantized_type, meta_grad = meta_grad_dict['fc'], slow_grad = None, lr = lr)
        elif 'fc' in meta_grad_dict and 'fc' in slow_grad_dict:
            x = self.fc(x = x, quantized_type = quantized_type, meta_grad = meta_grad_dict['fc'], slow_grad = slow_grad_dict['fc'], lr = lr)
        else:
            x = self.fc(x, quantized_type)
        # x = self.fc(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, first_stride=1, bitW=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.bitW = bitW
        # self.layer_name_list = [['conv1', ['conv1']], ['fc', ['fc']]]
        self.layer_name_list = []
        
        # self.conv1 = MetaQuantConv(3, 64, kernel_size=7, stride=2, padding=3, bias=False, bitW=bitW)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], layer_idx=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, layer_idx=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, layer_idx=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, layer_idx=4)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.relu2 = nn.ReLU(inplace=True)
        # self.fc = MetaQuantLinear(512 * block.expansion, num_classes, bitW=bitW)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1, layer_idx=0):
        downsample = None
        # if stride != 1 or self.in_channels != out_channels * block.expansion:
        #     downsample = nn.Sequential(
        #         MetaQuantConv(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False, bitW=self.bitW),
        #         nn.BatchNorm2d(out_channels * block.expansion),
        #     )
        #     self.layer_name_list.append(['layer%d.0.downsample.0' %layer_idx,
        #                                  ['layer%d' % layer_idx, 0, 'downsample', 0]])
        if stride != 1 or self.in_channels != out_channels * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, layer_idx=layer_idx, block_idx=0, bitW=self.bitW,  layer_name_list=self.layer_name_list))
        self.in_channels = out_channels * block.expansion
        for blk_idx in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, layer_idx=layer_idx, block_idx=blk_idx, bitW=self.bitW, layer_name_list=self.layer_name_list))

        return nn.Sequential(*layers)

    def forward(self, x, quantized_type=None, meta_grad_dict=dict(), slow_grad_dict = None, lr=1e-3):
        
        # if 'conv1' in meta_grad_dict and slow_grad_dict is None:
        #     x = self.conv1(x=x, quantized_type=quantized_type, meta_grad=meta_grad_dict['conv1'], slow_grad = None, lr=lr)
        # elif 'conv1' in meta_grad_dict and 'conv1' in slow_grad_dict:
        #     x = self.conv1(x = x, quantized_type = quantized_type, meta_grad = meta_grad_dict['conv1'], slow_grad = slow_grad_dict['conv1'], lr = lr)
        # else:
        #     x = self.conv1(x, quantized_type)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x, quantized_type, meta_grad_dict, slow_grad_dict, lr)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        
        # if 'fc' in meta_grad_dict and slow_grad_dict is None:
        #     x = self.fc(x = x, quantized_type = quantized_type, meta_grad = meta_grad_dict['fc'], slow_grad = None, lr = lr)
        # elif 'fc' in meta_grad_dict and 'fc' in slow_grad_dict:
        #     x = self.fc(x = x, quantized_type = quantized_type, meta_grad = meta_grad_dict['fc'], slow_grad = slow_grad_dict['fc'], lr = lr)
        # else:
        #     x = self.fc(x, quantized_type)
        
        x = self.fc(x)
        x = self.logsoftmax(x)

        return x


def get_kernel_shape(net):

    try:
        net = net.module
    except:
        pass

    for layer_info in net.layer_name_list:
        layer_name = layer_info[0]
        layer_idx = layer_info[1]
        # Parser layer information
        if len(layer_idx) == 1 or (not isinstance(layer_idx, list)):
            layer = getattr(net, layer_name)
        elif len(layer_idx) == 3:
            layer = getattr(getattr(net, layer_idx[0])[layer_idx[1]], layer_idx[2])
        elif len(layer_idx) == 4:
            layer = getattr(getattr(net, layer_idx[0])[layer_idx[1]], layer_idx[2])[layer_idx[3]]
        else:
            print(layer_idx)
            raise NotImplementedError
        # Access to layer weight
        weight = layer.weight
        print('%s: %s' %(layer_name, weight.shape))


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet20_stl(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], first_stride = 3, **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model

def resnet32_stl(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], first_stride = 3, **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet56_stl(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model

# official
def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    # net = preact_resnet110_cifar()
    net = resnet20_cifar(bitW=2).cuda()
    inputs = torch.rand([1, 3, 32, 32]).cuda()
    outputs = net(inputs)
    # print(net)
    # print(outputs.size())
    # get_kernel_shape(net)

