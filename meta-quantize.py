"""
A simplification version of meta-quantize for multiple experiments
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import pickle
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from utils.dataset import get_dataloader
from meta_utils.meta_network import *
from meta_utils.SGD import SGD
from meta_utils.adam import Adam
from meta_utils.helpers import *
from meta_utils.meta_quantized_module import *
from utils.recorder import Recorder
from utils.miscellaneous import AverageMeter, accuracy, progress_bar
from utils.miscellaneous import get_layer
from utils.quantize import test

##################
# Import Network #
##################
from models_CIFAR.quantized_meta_resnet import *
# from models_ImageNet.quantized_meta_resnet import resnet18, resnet34, resnet50

import argparse
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string') # 不是有效的布尔字符串
    return s == 'True'

parser = argparse.ArgumentParser(description='Meta Quantization')
parser.add_argument('--model', '-m', type=str, default='ResNet20', help='Model Arch')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--optimizer', '-o', type=str, default='Adam', help='Optimizer Method')
parser.add_argument('--quantize', '-q', type=str, default='dorefa', help='Quantization Method')
parser.add_argument('--exp_spec', '-e', type=str, default='', help='Experiment Specification')
parser.add_argument('--init_lr', '-lr', type=float, default=1e-2, help='Initial Learning rate')
parser.add_argument('--bitW', '-bw', type=int, default=1, help='Quantization Bit')
parser.add_argument('--meta_type', '-meta', type=str, default='MultiFC', help='Type of Meta Network')
parser.add_argument('--hidden_size', '-hidden', type=int, default=100,
                    help='Hidden size of meta network')
parser.add_argument('--num_fc', '-nfc', type=int, default=3,
                    help='Number of layer of FC in MultiFC')
parser.add_argument('--num_lstm', '-nlstm', type=int, default=2,
                    help='Number of layer of LSTM in MultiLSTMFC')
parser.add_argument('--n_epoch', '-n', type=int, default=100,
                    help='Maximum training epochs')
parser.add_argument('--fix_meta', '-fix', type=boolean_string, default='False',
                    help='Whether to fix meta')
parser.add_argument('--fix_meta_epoch', '-n_fix', type=int, default=0,
                    help='When to fix meta')
parser.add_argument('--random', '-r', type=str, default=None,
                    help='Whether to use random layer')
parser.add_argument('--meta_nonlinear', '-nonlinear', type=str, default=None,
                    help='Nonlinear used in meta network')
parser.add_argument('--lr_adjust', '-ad', type=str,
                    default='30', help='LR adjusting method')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size')
parser.add_argument('--weight_decay', '-decay', type=float, default=0,
                    help='Weight decay for training meta quantizer')
parser.add_argument('--use_lora', action='store_true', default=False)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
parser.add_argument('--break_continue', action='store_true', default=False)
args = parser.parse_args()

# ------------------------------------------
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
model_name = args.model # ResNet32
dataset_name = args.dataset
meta_method = args.meta_type # ['LSTM', 'FC', 'simple', 'MultiFC']
MAX_EPOCH = args.n_epoch
optimizer_type = args.optimizer # ['SGD', 'SGD-M', 'adam'] adam
hidden_size = args.hidden_size
num_lstm = args.num_lstm
num_fc = args.num_fc
random_type = args.random
lr_adjust = args.lr_adjust
batch_size = args.batch_size
bitW = args.bitW
quantized_type = args.quantize
save_root = './Results/%s-%s' % (model_name, dataset_name)
checkpoint_dir = './checkpoint/%s-%s' % (model_name, dataset_name)
break_continue = args.break_continue
use_lora = args.use_lora
# save_root = './full_precision/%s-%s' % (model_name, dataset_name)
# ------------------------------------------
print(args)
# input('Take a look')

import utils.global_var as gVar # 全局参数
gVar.meta_count = 0

###################
# Initial Network #
###################
if model_name == 'ResNet20':
    net = resnet20_cifar(bitW=bitW)
elif model_name == 'ResNet32':
    net = resnet32_cifar(bitW=bitW)
elif model_name == 'ResNet56':
    net = resnet56_cifar(num_classes=100, bitW=bitW)
elif model_name == 'ResNet44':
    net = resnet44_cifar(bitW=bitW)
elif model_name == 'ResNet18':
    if args.dataset == 'CIFAR10':
        net = resnet18(bitW=bitW, num_classes=10)
    elif args.dataset == 'ImageNet':
        net = resnet18(bitW=bitW, num_classes=1000)
    print(net)
else:
    raise NotImplementedError

localtime = time.strftime("%m-%d-%H:%M:%S", time.localtime())

if model_name in ['ResNet20', 'ResNet32', 'ResNet56', 'ResNet44'] and not break_continue:
    pretrain_path = '%s/%s-%s-pretrain.pth' % (save_root, model_name, dataset_name)
    net.load_state_dict(torch.load(pretrain_path, map_location=device), strict=False)
elif model_name in ['ResNet18'] and not break_continue:
    if args.dataset == 'ImageNet':
        pretrain_path = '/root/bqqi/fscil/MetaQuant/Results/ResNet18-ImageNet/resnet18-5c106cde.pth'
        net.load_state_dict(torch.load(pretrain_path, map_location='cpu'), strict=False)
    elif args.dataset == 'CIFAR10':
        # pretrain_path = '/root/bqqi/fscil/MetaQuant/Results/ResNet18-CIFAR10/model9509.pkl'
        # with open(pretrain_path, 'rb') as f:
        #     checkpoint = pickle.load(f)
        # net.load_state_dict(checkpoint, strict=False)
        pretrain_path = '/root/bqqi/fscil/MetaQuant/Results/ResNet18-CIFAR10/ResNet18-CIFAR10-pretrain.pth'
        net.load_state_dict(torch.load(pretrain_path, map_location='cpu'), strict=False)
        
    # pretrain_path = '%s/%s.t7' % (save_root, model_name)
    # pretrain_path = '/root/bqqi/fscil/MetaQuant/Results/ResNet18-ImageNet/resnet18-5c106cde.pth'
    # pretrain_path = '/root/bqqi/fscil/MetaQuant/Results/ResNet18-CIFAR10/ResNet18-CIFAR10-pretrain.pth'
    

# Get layer name list
layer_name_list = net.layer_name_list
assert (len(layer_name_list) == gVar.meta_count)
print('Layer name list completed.')

if use_lora:
    for param in net.parameters():
        param.requires_grad = False
    for layer_info in layer_name_list:
        layer_name = layer_info[0]
        layer_idx = layer_info[1] # ['layer2', 6, 'conv2']
        
        if len(layer_idx) == 3:
            layer = getattr(net, layer_idx[0])
            block = layer[layer_idx[1]]
            sublayer = getattr(block, layer_idx[2])
            setattr(block, layer_idx[2], MetaQuantConvWithLoRA.from_object(sublayer.in_channels, sublayer.out_channels, sublayer.kernel_size, sublayer.stride, sublayer.padding, sublayer.dilation, sublayer.groups, False, bitW, rank=8, alpha_lora=16, in_obj=sublayer))
        elif len(layer_idx) == 4:
            layer = getattr(net, layer_idx[0])
            block = layer[layer_idx[1]]
            sublayers = getattr(block, layer_idx[2])
            sublayer = sublayers[layer_idx[3]]
            sublayers[layer_idx[3]] = MetaQuantConvWithLoRA.from_object(sublayer.in_channels, sublayer.out_channels, sublayer.kernel_size, sublayer.stride, sublayer.padding, sublayer.dilation, sublayer.groups, False, bitW, rank=8, alpha_lora=16, in_obj=sublayer)
        elif len(layer_idx) == 1:
            sublayer = getattr(net, layer_idx[0])
        
            if 'fc' in layer_idx[0]:
                setattr(net, layer_idx[0], MetaQuantLinearWithLoRA.from_object(sublayer.in_features, sublayer.out_features, bitW=bitW, rank=8, alpha_lora=16, in_obj=sublayer))
            else:
                setattr(net, layer_idx[0], MetaQuantConvWithLoRA.from_object(sublayer.in_channels, sublayer.out_channels, sublayer.kernel_size, sublayer.stride, sublayer.padding, sublayer.dilation, sublayer.groups, False, bitW, rank=8, alpha_lora=16, in_obj=sublayer))
    

if use_cuda:
    # net = nn.DataParallel(net).cuda()
    net.cuda()

################
# Load Dataset #
################
train_loader = get_dataloader(dataset_name, 'train', batch_size)
test_loader = get_dataloader(dataset_name, 'test', 100)

##########################
# Construct Meta-Network #
##########################
if meta_method in ['LSTMFC-Grad', 'LSTMFC', 'LSTMFC-merge','LSTMFC-momentum']:
    meta_net = MetaLSTMFC(hidden_size=hidden_size)
    SummaryPath = '%s/runs-Quant/Meta-%s-Nonlinear-%s-' \
                  'hidden-size-%d-nlstm-1-%s-%s-%dbits-lr-%s-batchsize-%s' \
                  % (save_root, meta_method, args.meta_nonlinear, hidden_size,
                     quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH)
elif meta_method in ['FC-Grad']:
    meta_net = MetaFC(hidden_size=hidden_size, use_nonlinear=args.meta_nonlinear)
    SummaryPath = '%s/runs-Quant/Meta-%s-Nonlinear-%s-' \
                  'hidden-size-%d-%s-%s-%dbits-lr-%s-batchsize-%s' \
                  % (save_root, meta_method, args.meta_nonlinear, hidden_size,
                     quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH)
elif meta_method == 'MultiFC':
    meta_net = MetaDesignedMultiFC(hidden_size=hidden_size,
                                   num_layers = args.num_fc,
                                   use_nonlinear=args.meta_nonlinear)
    SummaryPath = '%s/runs-Quant/Meta-%s-Nonlinear-%s-' \
                  'hidden-size-%d-nfc-%d-%s-%s-%dbits-lr-%s' \
                  % (save_root, meta_method, args.meta_nonlinear, hidden_size, num_fc,
                     quantized_type, optimizer_type, bitW, lr_adjust)
elif meta_method == 'MultiFC-simple':
    meta_net = MetaMultiFC(hidden_size=hidden_size,
                                   use_nonlinear=args.meta_nonlinear)
    SummaryPath = '%s/runs-Quant/Meta-%s-Nonlinear-%s-' \
                  'hidden-size-%d-nfc-%d-%s-%s-%dbits-lr-%s' \
                  % (save_root, meta_method, args.meta_nonlinear, hidden_size, num_fc,
                     quantized_type, optimizer_type, bitW, lr_adjust)
elif meta_method == 'MetaCNN':
    meta_net = MetaCNN()
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust)
elif meta_method == 'MetaTransformer':
    meta_net = MetaTransformer(d_model=1, nhead=1, num_layers=4)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust)
elif meta_method in ['MetaMultiFCBN']:
    meta_net = MetaMultiFCBN(hidden_size=hidden_size, use_nonlinear=args.meta_nonlinear)
    SummaryPath = '%s/runs-Quant/Meta-%s-Nonlinear-%s-' \
                  'hidden-size-%d-%s-%s-%dbits-lr-%s' \
                  % (save_root, meta_method, args.meta_nonlinear, hidden_size,
                     quantized_type, optimizer_type, bitW, lr_adjust)
elif meta_method == 'MetaSimple':
    meta_net = MetaSimple()
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust)
elif meta_method == 'MetaLSTMLoRA':
    meta_net = MetaLSTMLoRA(hidden_size=hidden_size)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH)
elif meta_method == 'MetaMamba':
    meta_net = MetaMamba(d_model=1, d_state=16, d_conv=4, expand=100)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH, localtime)
elif meta_method == 'MetaMambaHistory':
    meta_net = MetaMambaHistory(d_model=1, d_state=16, d_conv=8, expand=100)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH, localtime)
elif meta_method == 'MetaMambaFusion':
    meta_net = MetaMambaHistory(d_model=1, d_state=32, d_conv=4, expand=32)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH, localtime)
elif meta_method == 'MetaS4':
    meta_net = MetaS4(d_model=1, d_state=16)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH, localtime)
elif meta_method == 'MetaS4History':
    meta_net = S4ModelHand(d_input=1, d_model=100, d_output=1, n_layers=1, d_state=16)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH, localtime)
elif meta_method == 'MetaS5History':
    meta_net = MetaS5Block(d_input=1, dim=8, state_dim=8, bidir=False)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH, localtime)
elif meta_method == 'MetaS5Fusion':
    meta_net = MetaS5Block(d_input=1, dim=1, state_dim=32, bidir=False)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH, localtime)
elif meta_method == 'MetaFastAndSlow':
    slow_meta_net = MetaMambaHistory(num_layers=len(layer_name_list), d_model=1, d_state=16, d_conv=8, expand=100)
    fast_meta_net = MetaMultiFC(hidden_size=hidden_size, use_nonlinear=args.meta_nonlinear)
    # slow_meta_net = nn.DataParallel(slow_meta_net)
    # fast_meta_net = nn.DataParallel(fast_meta_net)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH, localtime)
elif meta_method == 'MetaFastAndLSTM':
    slow_meta_net = MetaLSTMFC(hidden_size=hidden_size)
    fast_meta_net = MetaMultiFC(hidden_size=hidden_size, use_nonlinear=args.meta_nonlinear)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH, localtime)
elif meta_method == 'MetaMambaAndFC':
    fast_meta_net = MetaMambaHistory(d_model=1, d_state=16, d_conv=8, expand=100)
    slow_meta_net = MetaMultiFC(hidden_size=hidden_size, use_nonlinear=args.meta_nonlinear)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH, localtime)
elif meta_method == 'MetaDualGrad':
    meta_net = MetaDualGrad(d_model=1, d_state=16, d_conv=8, expand=100, hidden_size=hidden_size, use_nonlinear=args.meta_nonlinear)
    SummaryPath = '%s/runs-Quant/%s-%s-%s-%dbits-lr-%s-batchsize-%s-%s' \
                  % (save_root, meta_method, quantized_type, optimizer_type, bitW, lr_adjust, MAX_EPOCH, localtime)
else:
    raise NotImplementedError

if meta_method in ['MetaFastAndSlow', 'MetaFastAndLSTM', 'MetaMambaAndFC']:
    print(slow_meta_net)
    print(fast_meta_net)
    if use_cuda:
        slow_meta_net.cuda()
        fast_meta_net.cuda()
    slow_meta_optimizer = optim.Adam(slow_meta_net.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    fast_meta_optimizer = optim.Adam(fast_meta_net.parameters(), lr=1e-3, weight_decay=args.weight_decay)
else:
    print(meta_net)
    if use_cuda:
        meta_net.cuda()
    meta_optimizer = optim.Adam(meta_net.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    
meta_hidden_state_dict = dict() # Dictionary to store hidden states for all layers for memory-based meta network
meta_grad_dict = dict() # Dictionary to store meta net output: gradient for origin network's weight / bias
momentum_dict = dict()
history_grad = dict()
conv_state_dict = dict()
ssm_state_dict = dict()
s4_state_dict = dict()
slow_meta_grad_dict = dict()
fast_meta_grad_dict = dict()

##################
# Begin Training #
##################
# meta_opt_flag = True # When it is false, stop updating meta optimizer

# Optimizer for original network, just for zeroing gradient and get refined gradient
if optimizer_type == 'SGD-M':
    optimizee = SGD(net.parameters(), lr=args.init_lr,
                    momentum=0.9, weight_decay=5e-4)
elif optimizer_type == 'SGD':
    optimizee = SGD(net.parameters(), lr=args.init_lr)
elif optimizer_type in ['adam', 'Adam']:
    optimizee = Adam(net.parameters(), lr=args.init_lr) # ,weight_decay=5e-4
else:
    raise NotImplementedError

start_epoch = 0
if break_continue:
    # net continue learning
    net_checkpoint_path = '%s/net_checkpoint.pth' % (checkpoint_dir)
    # 判断该文件是否存在，如果存在则torch.load()读取
    if os.path.exists(net_checkpoint_path):
        net_checkpoint = torch.load(net_checkpoint_path, map_location=device)
        start_epoch = net_checkpoint['epoch']
        net.load_state_dict(net_checkpoint['model_state_dict'])
        optimizee.load_state_dict(net_checkpoint['optimizer_state_dict'])
        for layer_info in layer_name_list:
            layer_name = layer_info[0]
            layer_idx = layer_info[1]
            layer = get_layer(net, layer_idx)
            layer.quantized_grads = net_checkpoint[layer_name]
            layer.pre_quantized_weight = net_checkpoint[layer_name]
    slow_checkpoint = '%s/slow_checkpoint.pth' % (checkpoint_dir)
    if os.path.exists(slow_checkpoint):
        slow_checkpoint = torch.load(slow_checkpoint, map_location=device)
        slow_meta_net.load_state_dict(slow_checkpoint['model_state_dict'])
        slow_meta_optimizer.load_state_dict(slow_checkpoint['optimizer_state_dict'])
    fast_checkpoint = '%s/fast_checkpoint.pth' % (checkpoint_dir)
    if os.path.exists(fast_checkpoint):
        fast_checkpoint = torch.load(fast_checkpoint, map_location=device)
        fast_meta_net.load_state_dict(fast_checkpoint['model_state_dict'])
        fast_meta_optimizer.load_state_dict(fast_checkpoint['optimizer_state_dict'])

####################
# Initial Recorder #
####################
if args.exp_spec != '':
    SummaryPath += ('-' + args.exp_spec)

print('Save to %s' %SummaryPath)

if os.path.exists(SummaryPath):
    print('Record exist, remove')
    # input()
    shutil.rmtree(SummaryPath)
    os.makedirs(SummaryPath)
else:
    os.makedirs(SummaryPath)

recorder = Recorder(SummaryPath=SummaryPath, dataset_name=dataset_name)

##################
# Begin Training #
##################
meta_grad_dict = dict()
for epoch in range(start_epoch, MAX_EPOCH):

    if recorder.stop: break

    print('\nEpoch: %d, lr: %e' % (epoch, optimizee.param_groups[0]['lr']))

    net.train()
    end = time.time()

    recorder.reset_performance()
    
    train_loader = tqdm(train_loader, total=len(train_loader))

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if meta_method in ['MetaFastAndSlow', 'MetaFastAndLSTM', 'MetaMambaAndFC']:
            fast_meta_optimizer.zero_grad()
            slow_meta_optimizer.zero_grad()
        else:
            meta_optimizer.zero_grad() # 元优化器

        # Ignore the first meta gradient generation due to the lack of natural gradient
        if batch_idx == 0 and epoch == 0:
            pass
        else:
            if meta_method in ['MetaMamba', 'MetaS4']:
                meta_grad_dict, history_grad, conv_state_dict, ssm_state_dict, s4_state_dict = metassm_gradient_generation(meta_net, net, meta_method, history_grad, conv_state_dict, ssm_state_dict, s4_state_dict, False)
            elif meta_method in ['MetaFastAndSlow', 'MetaFastAndLSTM', 'MetaMambaAndFC']:
                if use_lora:
                    fast_meta_grad_dict, slow_meta_grad_dict, history_grad, meta_hidden_state_dict = meta_gradient_lora_generation(fast_meta_net, slow_meta_net, net, meta_method, history_grad, False, meta_hidden_state_dict)
                else:
                    fast_meta_grad_dict, slow_meta_grad_dict, history_grad, meta_hidden_state_dict = meta_fast_slow_gradient_generation(fast_meta_net, slow_meta_net, net, meta_method, history_grad, False, meta_hidden_state_dict)
            elif meta_method in ['MetaDualGrad']:
                fast_meta_grad_dict, slow_meta_grad_dict, history_grad, meta_hidden_state_dict = dual_gradient_generation(meta_net, net, meta_method, history_grad, False)
            else:
                meta_grad_dict, meta_hidden_state_dict, momentum_dict, history_grad = \
                    meta_gradient_generation(
                            meta_net, net, meta_method, meta_hidden_state_dict, False, momentum_dict, history_grad
                    )
            # meta_grad_dict_tosave = {key:value[1].detach().cpu() for key,value in meta_grad_dict.items()}
        # Conduct inference with meta gradient, which is incorporated into the computational graph
        if meta_method in ['MetaFastAndSlow', 'MetaFastAndLSTM', 'MetaMambaAndFC', 'MetaDualGrad']:
            outputs = net(
                inputs, quantized_type=quantized_type, meta_grad_dict=fast_meta_grad_dict, slow_grad_dict=slow_meta_grad_dict, lr=optimizee.param_groups[0]['lr']
            )
        else:
            outputs = net(
                inputs, quantized_type=quantized_type, meta_grad_dict=meta_grad_dict, slow_grad_dict=None, lr=optimizee.param_groups[0]['lr']
            )
        
        # Clear gradient, which is stored in layer.weight.grad
        optimizee.zero_grad()

        # Backpropagation to attain natural gradient, which is stored in layer.pre_quantized_grads
        losses = nn.CrossEntropyLoss()(outputs, targets)
        losses.backward()

        # for name, param in meta_net.named_parameters():
        #     print(param.device)
        #     print(name)
        
        if meta_method in ['MetaFastAndSlow', 'MetaFastAndLSTM', 'MetaMambaAndFC']:
            fast_meta_optimizer.step()
            slow_meta_optimizer.step()
        else:
            meta_optimizer.step()

        # Assign meta gradient for actual gradients used in update_parameters
        if meta_method in ['MetaFastAndSlow', 'MetaFastAndLSTM', 'MetaMambaAndFC', 'MetaDualGrad']:
            if use_lora:
                if len(fast_meta_grad_dict) != 0:
                    for layer_info in net.layer_name_list:
                        layer_name = layer_info[0]
                        layer_idx = layer_info[1]
                        layer = get_layer(net, layer_idx)
                        try:
                            layer.weight.grad.data = (
                                layer.delta_w
                            )
                            layer.A.grad.data = (fast_meta_grad_dict[layer_name][1][0] * layer.calibration_A)
                            layer.B.grad.data = (fast_meta_grad_dict[layer_name][1][1] * layer.calibration_B)
                        except:
                            pass

                    # Get refine gradients for actual parameters update
                    optimizee.get_refine_gradient()

                    # Actual parameters update using the refined gradient from meta gradient
                    update_parameters(net, lr=optimizee.param_groups[0]['lr'])
            else:
                if len(fast_meta_grad_dict) != 0:
                    for layer_info in net.layer_name_list:
                        layer_name = layer_info[0]
                        layer_idx = layer_info[1]
                        layer = get_layer(net, layer_idx)
                        layer.weight.grad.data = (
                            layer.calibration * fast_meta_grad_dict[layer_name][1].data
                        )

                    # Get refine gradients for actual parameters update
                    optimizee.get_refine_gradient()

                    # Actual parameters update using the refined gradient from meta gradient
                    update_parameters(net, lr=optimizee.param_groups[0]['lr'])
        else:
            if len(meta_grad_dict) != 0:
                for layer_info in net.layer_name_list:
                    layer_name = layer_info[0]
                    layer_idx = layer_info[1]
                    layer = get_layer(net, layer_idx)
                    layer.weight.grad.data = (
                        layer.calibration * meta_grad_dict[layer_name][1].data
                    )

                # Get refine gradients for actual parameters update
                optimizee.get_refine_gradient()

                # Actual parameters update using the refined gradient from meta gradient
                update_parameters(net, lr=optimizee.param_groups[0]['lr'])

        recorder.update(loss=losses.data.item(), acc=accuracy(outputs.data, targets.data, (1,5)),
                        batch_size=outputs.shape[0], cur_lr=optimizee.param_groups[0]['lr'], end=end)

        # recorder.print_training_result(batch_idx, len(train_loader))
        end = time.time()
        
    net_checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizee.state_dict(),
    }
    for layer_info in layer_name_list:
        layer_name = layer_info[0]
        layer_idx = layer_info[1]
        layer = get_layer(net, layer_idx)
        net_checkpoint[layer_name+'quantized_grads'] = layer.quantized_grads
        net_checkpoint[layer_name+'pre_quantized_weight'] = layer.pre_quantized_weight
        net_checkpoint[layer_name+'bias_grad'] = layer.bias_grad
    torch.save(net_checkpoint, '%s/net_checkpoint.pth' % checkpoint_dir)
    slow_checkpoint = {
        'epoch': epoch,
        'model_state_dict': slow_meta_net.state_dict(),
        'optimizer_state_dict': slow_meta_optimizer.state_dict(),
    }
    torch.save(slow_checkpoint, '%s/slow_checkpoint.pth' % checkpoint_dir)
    fast_checkpoint = {
        'epoch': epoch,
        'model_state_dict': fast_meta_net.state_dict(),
        'optimizer_state_dict': fast_meta_optimizer.state_dict(),
    }
    torch.save(fast_checkpoint, '%s/fast_checkpoint.pth' % checkpoint_dir)
    # if epoch % 5 == 0:
    #     draw_weight_distribution(net, epoch)
    test_acc = test(net, quantized_type=quantized_type, test_loader=test_loader,
                    dataset_name=dataset_name, n_batches_used=None)
    bta_epoch = recorder.get_best_test_acc()
    recorder.update(loss=None, acc=test_acc, batch_size=0, end=None, is_train=False)

    # Adjust learning rate
    recorder.adjust_lr(optimizer=optimizee, adjust_type=lr_adjust, epoch=epoch)

best_test_acc = recorder.get_best_test_acc()
if type(best_test_acc) == tuple:
    print('Best test top 1 acc: %.3f, top 5 acc: %.3f' % (best_test_acc[0], best_test_acc[1]))
else:
    print('Best test acc: %.3f' %best_test_acc)
recorder.close()