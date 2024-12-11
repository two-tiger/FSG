"""
Some helper function for meta inference
"""

import torch
from utils.miscellaneous import get_layer
import numpy as np
import pandas as pd
import joypy
import matplotlib.pyplot as plt


def draw_weight_distribution(net, epoch):
    
    layer_name_list = net.layer_name_list # 这里把主网络的层名字都拿出来了
    meta_weight_dict = {}

    for idx, layer_info in enumerate(layer_name_list):

        layer_name = layer_info[0]
        layer_idx = layer_info[1]
        
        layer = get_layer(net, layer_idx)
        meta_weight = layer.meta_weight.flatten().cpu().detach().numpy()
        # meta_weight = layer.pre_quantized_weight.flatten().cpu().detach().numpy()
        meta_weight_dict[layer_name] = meta_weight
        
    df_meta_weight = pd.DataFrame.from_dict(meta_weight_dict, orient='index').transpose()
    fig, axes = joypy.joyplot(df_meta_weight, figsize=(10,20), grid='y', linewidth=1, title='Weight distribution chart')
    # plt.savefig('/root/bqqi/fscil/MetaQuant/visualization/weight_distribution/epoch%s-weight.pdf' % (str(epoch)))
    plt.savefig('/root/bqqi/fscil/MetaQuant/visualization/without_decay/epoch%s-weight.pdf' % (str(epoch)))


def meta_gradient_generation(meta_net, net, meta_method, meta_hidden_state_dict=None, fix_meta=False, momentum_dict=None, history_grad=None):

    meta_grad_dict = dict()
    new_meta_hidden_state_dict = dict()
    new_momentum_dict = dict()
    layer_name_list = net.layer_name_list # 这里把主网络的层名字都拿出来了

    for idx, layer_info in enumerate(layer_name_list):

        layer_name = layer_info[0]
        layer_idx = layer_info[1]

        layer = get_layer(net, layer_idx)

        grad = layer.quantized_grads.data # 这里拿到这一层的梯度 (16,3,3,3)
        pre_quantized_weight = layer.pre_quantized_weight.data # 这里拿到这一层的权重大小 (16,3,3,3)
        bias = layer.bias # 这里拿到这一层的偏置

        if bias is not None:
            bias_grad = bias.grad.data.clone()
        else:
            bias_grad = None

        if meta_method == 'FC-Grad':
            meta_input = grad.data.view(-1, 1)

            if fix_meta:
                with torch.no_grad():
                    meta_grad = meta_net(meta_input)
            else:
                meta_grad = meta_net(meta_input)

        elif meta_method == 'simple':
            meta_grad = grad.data

        elif meta_method in ['MultiFC','MultiFC-simple','MetaCNN','MetaTransformer','MetaMultiFCBN','MetaSimple']:

            flatten_grad = grad.data.view(-1, 1)
            flatten_weight = pre_quantized_weight.data.view(-1, 1)

            if fix_meta:
                with torch.no_grad():
                    meta_output = meta_net(flatten_weight)
            else:
                meta_output = meta_net(flatten_weight)

            meta_grad = flatten_grad * meta_output

        elif meta_method in ['LSTMFC']: # Meta LSTM Weight

            flatten_grad = grad.data.view(1, -1, 1) # (1,432,1)
            flatten_weight = pre_quantized_weight.data.view(1, -1, 1) # (1,432,1)

            if meta_hidden_state_dict is not None and layer_name in meta_hidden_state_dict:
                meta_hidden_state = meta_hidden_state_dict[layer_name]
            else:
                meta_hidden_state = None

            if fix_meta:
                with torch.no_grad():
                    meta_output, hidden = meta_net(flatten_weight, meta_hidden_state)
                    # meta_output, hidden = meta_net(flatten_grad, meta_hidden_state)
            else:
                meta_output, hidden = meta_net(flatten_weight, meta_hidden_state)
                # meta_output, hidden = meta_net(flatten_grad, meta_hidden_state)

            new_meta_hidden_state_dict[layer_name] = tuple(h.detach() for h in hidden)

            meta_grad = flatten_grad * meta_output
            # meta_grad = meta_output
            
        elif meta_method in ['LSTMFC-Grad']: # Meta LSTM Grad

            flatten_grad = grad.data.view(1, -1, 1) # (1,432,1)
            flatten_weight = pre_quantized_weight.data.view(1, -1, 1) # (1,432,1)

            if meta_hidden_state_dict is not None and layer_name in meta_hidden_state_dict:
                meta_hidden_state = meta_hidden_state_dict[layer_name]
            else:
                meta_hidden_state = None

            if fix_meta:
                with torch.no_grad():
                    # meta_output, hidden = meta_net(flatten_weight, meta_hidden_state)
                    meta_output, hidden = meta_net(flatten_grad, meta_hidden_state)
            else:
                # meta_output, hidden = meta_net(flatten_weight, meta_hidden_state)
                meta_output, hidden = meta_net(flatten_grad, meta_hidden_state)

            new_meta_hidden_state_dict[layer_name] = tuple(h.detach() for h in hidden)

            meta_grad = flatten_grad * meta_output
            # meta_grad = meta_output
            
        elif meta_method in ['LSTMFC-merge']: # Meta LSTM weight + grad

            flatten_grad = grad.data.view(1, -1, 1) # (1,432,1)
            flatten_weight = pre_quantized_weight.data.view(1, -1, 1) # (1,432,1)
            merge_input = torch.cat((flatten_weight, flatten_grad), dim=0)

            if meta_hidden_state_dict is not None and layer_name in meta_hidden_state_dict:
                meta_hidden_state = meta_hidden_state_dict[layer_name]
            else:
                meta_hidden_state = None

            if fix_meta:
                with torch.no_grad():
                    meta_output, hidden = meta_net(merge_input, meta_hidden_state)
            else:
                meta_output, hidden = meta_net(merge_input, meta_hidden_state)

            new_meta_hidden_state_dict[layer_name] = tuple(h.detach() for h in hidden)

            meta_grad = flatten_grad * meta_output
            # meta_grad = meta_output

        elif meta_method == 'LSTMFC-momentum':
            
            flatten_grad = grad.data.view(1, -1, 1) # (1,432,1)
            flatten_weight = pre_quantized_weight.data.view(1, -1, 1) # (1,432,1)
            
            if momentum_dict is not None and layer_name in momentum_dict:
                momentum = momentum_dict[layer_name]
            else:
                momentum = flatten_grad
                
            new_momentum = 0.9 * momentum + (1 - 0.9) * flatten_grad
            new_momentum_dict[layer_name] = new_momentum

            if meta_hidden_state_dict is not None and layer_name in meta_hidden_state_dict:
                meta_hidden_state = meta_hidden_state_dict[layer_name]
            else:
                meta_hidden_state = None

            if fix_meta:
                with torch.no_grad():
                    meta_output, hidden = meta_net(momentum, meta_hidden_state)
            else:
                meta_output, hidden = meta_net(momentum, meta_hidden_state)

            new_meta_hidden_state_dict[layer_name] = tuple(h.detach() for h in hidden)

            meta_grad = flatten_grad * meta_output
            # meta_grad = meta_output
            
        elif meta_method in ['MetaMambaHistory', 'MetaS4History', 'MetaS5History']:
            
            grad_in = grad.data.view(1, -1, 1) # (16, 3, 3, 3) ()
            weight_in = pre_quantized_weight.data.view(1, -1, 16)
            
            b,l,d = grad_in.shape
            
            if history_grad is not None and layer_name in history_grad:
                his_grad = history_grad[layer_name]
                # new_grad = 0.3 * his_grad + (1 - 0.3) * grad_in
                if his_grad.shape[1]/l == 5:
                    his_grad = torch.cat((his_grad[:,-l*4:,:], grad_in), 1)
                else:
                    his_grad = torch.cat((his_grad, grad_in), 1)
            else:
                his_grad = grad_in
                
            history_grad[layer_name] = his_grad

            if fix_meta:
                with torch.no_grad():
                    meta_output = meta_net(his_grad)
            else:
                meta_output = meta_net(his_grad)

            meta_output = meta_output[:, -grad_in.shape[1]:, :] #.unsqueeze(1)
            # meta_grad = grad_in * meta_output
            meta_grad = meta_output
            
        elif meta_method in ['MetaMambaFusion', 'MetaS4Fusion', 'MetaS5Fusion']:
            
            grad_in = grad.data.view(1, -1, 1) # (16, 3, 3, 3) ()
            weight_in = pre_quantized_weight.data.view(1, -1, 16)
        
            b,l,d = grad_in.shape
            
            if momentum_dict is not None and layer_name in momentum_dict:
                momentum = momentum_dict[layer_name]
            else:
                momentum = grad_in
                
            new_momentum = 0.3 * momentum + (1 - 0.3) * grad_in
            new_momentum_dict[layer_name] = new_momentum
            
            if history_grad is not None and layer_name in history_grad:
                his_grad = history_grad[layer_name]
                # new_grad = 0.3 * his_grad + (1 - 0.3) * grad_in
                if his_grad.shape[1]/l == 5:
                    his_grad = torch.cat((his_grad[:,-l*4:,:], new_momentum), 1)
                else:
                    his_grad = torch.cat((his_grad, new_momentum), 1)
            else:
                his_grad = grad_in
                
            history_grad[layer_name] = his_grad

            if fix_meta:
                with torch.no_grad():
                    meta_output = meta_net(his_grad)
            else:
                meta_output = meta_net(his_grad)

            meta_output = meta_output[:, -grad_in.shape[1]:, :] #.unsqueeze(1)
            # meta_grad = grad_in * meta_output
            meta_grad = meta_output
            
        
        else:
            raise NotImplementedError

        # Reshape the flattened meta gradient into the original shape
        meta_grad = meta_grad.reshape(grad.shape)

        if bias is not None:
            meta_grad_dict[layer_name] = (layer_idx, meta_grad, bias_grad.data)
        else:
            meta_grad_dict[layer_name] = (layer_idx, meta_grad, None)

        # Assigned pre_quantized_grads with meta grad for weights update
        # layer.pre_quantized_grads = meta_grad.data.clone()

    return meta_grad_dict, new_meta_hidden_state_dict, new_momentum_dict, history_grad


def metassm_gradient_generation(meta_net, net, meta_method, history_grad=None, conv_state_dict=None, ssm_state_dict=None, s4_state_dict=None, fix_meta=False):

    meta_grad_dict = dict()
    new_conv_state_dict = dict()
    new_ssm_state_dict = dict()
    new_s4_state_dict = dict()
    layer_name_list = net.layer_name_list # 这里把主网络的层名字都拿出来了

    for idx, layer_info in enumerate(layer_name_list):

        layer_name = layer_info[0]
        layer_idx = layer_info[1]

        layer = get_layer(net, layer_idx)

        grad = layer.quantized_grads.data # 这里拿到这一层的梯度 (16,3,3,3)
        pre_quantized_weight = layer.pre_quantized_weight.data # 这里拿到这一层的权重大小 (16,3,3,3)
        bias = layer.bias # 这里拿到这一层的偏置

        if bias is not None:
            bias_grad = bias.grad.data.clone()
        else:
            bias_grad = None

        if meta_method == "MetaMamba":
            grad_in = grad.data.view(-1, 1, 1)
            weight_in = pre_quantized_weight.data.view(1, -1, 16)
            
            b,l,d = grad_in.shape
            
            # if history_grad is not None and layer_name in history_grad:
            #     his_grad = history_grad[layer_name]
            #     if his_grad.shape[1] == 2:
            #         his_grad = torch.cat((his_grad[:,1:,:], grad_in), 1)
            #     else:
            #         his_grad = torch.cat((his_grad, grad_in), 0)
            # else:
            #     his_grad = grad_in
                
            # history_grad[layer_name] = his_grad
            
            if conv_state_dict is not None and layer_name in conv_state_dict:
                conv_state = conv_state_dict[layer_name]
                conv_state = torch.stack(conv_state)
            else:
                conv_state = torch.zeros_like(grad_in).repeat(1, 100, 1).cuda()
                

            if ssm_state_dict is not None and layer_name in ssm_state_dict:
                ssm_state = ssm_state_dict[layer_name]
                ssm_state = torch.stack(ssm_state)
            else:
                ssm_state = torch.zeros((grad_in.shape[0], 100, 16)).cuda() # expand * d_model, d_state

            if fix_meta:
                with torch.no_grad():
                    meta_output, conv_state, ssm_state = meta_net(grad_in, conv_state, ssm_state)
            else:
                meta_output, conv_state, ssm_state = meta_net(grad_in, conv_state, ssm_state)

            new_conv_state_dict[layer_name] = tuple(h.detach() for h in conv_state)
            
            new_ssm_state_dict[layer_name] = tuple(h.detach() for h in ssm_state)
            
            # meta_output = meta_output[:, -1, :]
            # meta_grad = grad_in * meta_output
            meta_grad = meta_output
            
        elif meta_method == 'MetaS4':
            grad_in = grad.data.view(-1, 1)
            
            b,h = grad_in.shape
            
            # if history_grad is not None and layer_name in history_grad:
            #     his_grad = history_grad[layer_name]
            #     if his_grad.shape[1] == 2:
            #         his_grad = torch.cat((his_grad[:,1:,:], grad_in), 1)
            #     else:
            #         his_grad = torch.cat((his_grad, grad_in), 0)
            # else:
            #     his_grad = grad_in
                
            # history_grad[layer_name] = his_grad
            
            if s4_state_dict is not None and layer_name in s4_state_dict:
                s4_state = s4_state_dict[layer_name]
                s4_state = torch.stack(s4_state)
            else:
                # s4_state = torch.zeros_like(grad_in).repeat(1, 1, 16).cuda()
                s4_state = torch.zeros((1, 1, 16)).cuda()

            if fix_meta:
                with torch.no_grad():
                    meta_output, s4_state = meta_net(grad_in, s4_state)
            else:
                meta_output, s4_state = meta_net(grad_in, s4_state)

            new_s4_state_dict[layer_name] = tuple(h.detach() for h in s4_state)
            # new_s4_state_dict[layer_name] = s4_state
            
            # meta_output = meta_output[:, -1, :]
            # meta_grad = grad_in * meta_output
            meta_grad = meta_output
            

        # Reshape the flattened meta gradient into the original shape
        meta_grad = meta_grad.reshape(grad.shape)

        if bias is not None:
            meta_grad_dict[layer_name] = (layer_idx, meta_grad, bias_grad.data)
        else:
            meta_grad_dict[layer_name] = (layer_idx, meta_grad, None)

        # Assigned pre_quantized_grads with meta grad for weights update
        layer.pre_quantized_grads = meta_grad.data.clone()

    return meta_grad_dict, history_grad, new_conv_state_dict, new_ssm_state_dict, new_s4_state_dict


def meta_fast_slow_gradient_generation(fast_meta_net, slow_meta_net, net, meta_method, history_grad=None, fix_meta=False, meta_hidden_state_dict=None, length=5):
    
    '''
    类似momentum这种具有历史信息的梯度被认为是slow grad使用SSM、LSTM建模；当前的梯度直接用FC进行建模
    '''

    fast_meta_grad_dict = dict()
    slow_meta_grad_dict = dict()
    new_meta_hidden_state_dict = dict()

    layer_name_list = net.layer_name_list # 这里把主网络的层名字都拿出来了

    for idx, layer_info in enumerate(layer_name_list):

        layer_name = layer_info[0] # 'layer2.6.conv2'
        layer_idx = layer_info[1] # ['layer2', 6, 'conv2']

        layer = get_layer(net, layer_idx)

        grad = layer.quantized_grads.data # 这里拿到这一层的梯度 (16,3,3,3)
        pre_quantized_weight = layer.pre_quantized_weight.data # 这里拿到这一层的权重大小 (16,3,3,3)
        bias = layer.bias # 这里拿到这一层的偏置

        if bias is not None:
            bias_grad = bias.grad.data.clone()
        else:
            bias_grad = None
            
        if meta_method in ['MetaFastAndSlow']:

            flatten_grad = grad.data.view(-1, 1)
            flatten_weight = pre_quantized_weight.data.view(-1, 1)
            
            # fast meta net
            if fix_meta:
                with torch.no_grad():
                    fast_meta_output = fast_meta_net(flatten_weight)
            else:
                fast_meta_output = fast_meta_net(flatten_weight)
            
            fast_meta_grad = flatten_grad * fast_meta_output
            
            # slow meta net

            grad_in = grad.data.view(1, -1, 1)
            weight_in = pre_quantized_weight.data.view(1, -1, 1)
            
            # padding_size = 200 - grad_in.shape[1] % 200
            # padding_weight = torch.cat([grad_in, torch.zeros(1, padding_size, 1).cuda()], dim=1)
            # grad_in = padding_weight.view(1, -1, 200) # 1, l, 200
            
            b,l,d = grad_in.shape
            
            if history_grad is not None and layer_name in history_grad:
                his_grad = history_grad[layer_name]
                # new_grad = 0.3 * his_grad + (1 - 0.3) * grad_in
                if length == 1:
                    his_grad = grad_in
                elif his_grad.shape[1]/l == length:
                    l_tmp = int(l*(length-1))
                    his_grad = torch.cat((his_grad[:,-l_tmp:,:], grad_in), 1)
                else:
                    his_grad = torch.cat((his_grad, grad_in), 1)
            else:
                his_grad = grad_in
                
            history_grad[layer_name] = his_grad
            if fix_meta:
                with torch.no_grad():
                    slow_meta_output = slow_meta_net(his_grad, idx)
            else:
                slow_meta_output = slow_meta_net(his_grad, idx)

            # slow_meta_output = slow_meta_output[:, -grad_in.shape[1]:, :].reshape(1, -1, 1)[:, :-padding_size, :]
            slow_meta_output = slow_meta_output[:, -grad_in.shape[1]:, :].view(1, -1, 1)
            # meta_grad = grad_in * slow_meta_output
            slow_meta_grad = slow_meta_output
            
        elif meta_method in ['MetaFastAndLSTM']:
            
            flatten_grad = grad.data.view(1, -1, 1) # (1,432,1)
            flatten_weight = pre_quantized_weight.data.view(1, -1, 1) # (1,432,1)

            if meta_hidden_state_dict is not None and layer_name in meta_hidden_state_dict:
                meta_hidden_state = meta_hidden_state_dict[layer_name]
            else:
                meta_hidden_state = None

            if fix_meta:
                with torch.no_grad():
                    slow_meta_output, hidden = slow_meta_net(flatten_weight, meta_hidden_state)
                    # meta_output, hidden = meta_net(flatten_grad, meta_hidden_state)
            else:
                # print(flatten_weight.shape)
                slow_meta_output, hidden = slow_meta_net(flatten_weight, meta_hidden_state)
                # meta_output, hidden = meta_net(flatten_grad, meta_hidden_state)

            new_meta_hidden_state_dict[layer_name] = tuple(h.detach() for h in hidden)

            slow_meta_grad = flatten_grad * slow_meta_output
            
            flatten_grad = grad.data.view(-1, 1)
            flatten_weight = pre_quantized_weight.data.view(-1, 1)
            
            # fast meta net
            if fix_meta:
                with torch.no_grad():
                    fast_meta_output = fast_meta_net(flatten_weight)
            else:
                fast_meta_output = fast_meta_net(flatten_weight)

            fast_meta_grad = flatten_grad * fast_meta_output
            
        
        elif meta_method in ['MetaMambaAndFC']:

            flatten_grad = grad.data.view(-1, 1)
            flatten_weight = pre_quantized_weight.data.view(-1, 1)
            
            # multi FC as slow meta net
            if fix_meta:
                with torch.no_grad():
                    slow_meta_output = slow_meta_net(flatten_weight)
            else:
                slow_meta_output = slow_meta_net(flatten_weight)

            slow_meta_grad = flatten_grad * slow_meta_output
            
            # Mamba as fast meta net

            grad_in = grad.data.view(1, -1, 1)
            weight_in = pre_quantized_weight.data.view(1, -1, 1)
            
            b,l,d = grad_in.shape
            
            if history_grad is not None and layer_name in history_grad:
                his_grad = history_grad[layer_name]
                # new_grad = 0.3 * his_grad + (1 - 0.3) * grad_in
                if his_grad.shape[1]/l == 5:
                    his_grad = torch.cat((his_grad[:,-l*4:,:], grad_in), 1)
                else:
                    his_grad = torch.cat((his_grad, grad_in), 1)
            else:
                his_grad = grad_in
                
            history_grad[layer_name] = his_grad

            if fix_meta:
                with torch.no_grad():
                    fast_meta_output = fast_meta_net(his_grad)
            else:
                fast_meta_output = fast_meta_net(his_grad)

            fast_meta_output = fast_meta_output[:, -grad_in.shape[1]:, :]
            # meta_grad = grad_in * slow_meta_output
            fast_meta_grad = fast_meta_output
            
        
        else:
            raise NotImplementedError

        # Reshape the flattened meta gradient into the original shape
        fast_meta_grad = fast_meta_grad.reshape(grad.shape)
        slow_meta_grad = slow_meta_grad.reshape(grad.shape)

        if bias is not None:
            fast_meta_grad_dict[layer_name] = (layer_idx, fast_meta_grad, bias_grad.data)
            slow_meta_grad_dict[layer_name] = (layer_idx, slow_meta_grad, bias_grad.data)
        else:
            fast_meta_grad_dict[layer_name] = (layer_idx, fast_meta_grad, None)
            slow_meta_grad_dict[layer_name] = (layer_idx, slow_meta_grad, None)

        # Assigned pre_quantized_grads with meta grad for weights update
        # layer.pre_quantized_grads = meta_grad.data.clone()


    return fast_meta_grad_dict, slow_meta_grad_dict, history_grad, new_meta_hidden_state_dict

def meta_gradient_lora_generation(fast_meta_net, slow_meta_net, net, meta_method, history_grad=None, fix_meta=False, meta_hidden_state_dict=None):
    fast_meta_grad_dict = dict()
    slow_meta_grad_dict = dict()
    new_meta_hidden_state_dict = dict()

    layer_name_list = net.layer_name_list # 这里把主网络的层名字都拿出来了

    for idx, layer_info in enumerate(layer_name_list):

        layer_name = layer_info[0] # 'layer2.6.conv2'
        layer_idx = layer_info[1] # ['layer2', 6, 'conv2']

        layer = get_layer(net, layer_idx)
        
        # grad_A = layer.A_grad.data
        # grad_B = layer.B_grad.data
        weight_A = layer.A.data
        weight_B = layer.B.data
        try:
            grad_A = layer.A.grad.data
            grad_B = layer.B.grad.data
        except:
            grad_A = torch.zeros_like(weight_A)
            grad_B = torch.zeros_like(weight_B)
        # weight_A = layer.meta_A.data
        # weight_B = layer.meta_B.data

        # grad = layer.quantized_grads.data # 这里拿到这一层的梯度 (16,3,3,3)
        # pre_quantized_weight = layer.pre_quantized_weight.data # 这里拿到这一层的权重大小 (16,3,3,3)
        bias = layer.bias # 这里拿到这一层的偏置

        if bias is not None:
            try:
                bias_grad = bias.grad.data.clone()
            except:
                bias_grad = torch.zeros_like(bias).cuda()
        else:
            bias_grad = None
            
        if meta_method in ['MetaFastAndSlow']:

            flatten_grad_A = grad_A.data.view(-1, 1)
            flatten_grad_B = grad_B.data.view(-1, 1)
            a, b = flatten_grad_A.shape[0], flatten_grad_B.shape[0]
            con_grad = torch.cat((flatten_grad_A, flatten_grad_B), dim=0)
            flatten_weight_A = weight_A.data.view(-1, 1)
            flatten_weight_B = weight_B.data.view(-1, 1)
            con_weight = torch.cat((flatten_weight_A, flatten_weight_B), dim=0)
            
            # fast meta net
            if fix_meta:
                with torch.no_grad():
                    fast_meta_output = fast_meta_net(con_weight)
            else:
                fast_meta_output = fast_meta_net(con_weight)
            
            fast_meta_grad = con_grad * fast_meta_output
            fast_mata_grad_list = [fast_meta_grad[:a,:].reshape(grad_A.shape), fast_meta_grad[-b:,:].reshape(grad_B.shape)]
            
            # slow meta net
            
            # padding_size = 200 - grad_in.shape[1] % 200
            # padding_weight = torch.cat([grad_in, torch.zeros(1, padding_size, 1).cuda()], dim=1)
            # grad_in = padding_weight.view(1, -1, 200) # 1, l, 200
            
            grad_A_in = grad_A.data.view(1, -1, 1)
            grad_B_in = grad_B.data.view(1, -1, 1)
            la, lb = grad_A_in.shape[1], grad_B_in.shape[1]
            con_grad_in = torch.cat((grad_A_in, grad_B_in), dim=1)
            b,l,d = con_grad_in.shape
            
            if history_grad is not None and layer_name in history_grad:
                his_grad = history_grad[layer_name]
                # new_grad = 0.3 * his_grad + (1 - 0.3) * grad_in
                if his_grad.shape[1]/l == 5:
                    his_grad = torch.cat((his_grad[:,-l*4:,:], con_grad_in), 1)
                else:
                    his_grad = torch.cat((his_grad, con_grad_in), 1)
            else:
                his_grad = con_grad_in
                
            history_grad[layer_name] = his_grad
            
            if fix_meta:
                with torch.no_grad():
                    slow_meta_output = slow_meta_net(his_grad, idx)
            else:
                slow_meta_output = slow_meta_net(his_grad, idx)

            slow_meta_output = slow_meta_output[:, -l:, :].view(-1, 1)
            # meta_grad = grad_in * slow_meta_output
            slow_meta_grad = con_grad_in * slow_meta_output
            slow_mata_grad_list = [slow_meta_grad[:,:la,:].reshape(grad_A.shape), slow_meta_grad[:,-lb:,:].reshape(grad_B.shape)]
            
        else:
            raise NotImplementedError

        # Reshape the flattened meta gradient into the original shape
        # fast_meta_grad = fast_meta_grad.reshape(grad.shape)
        # slow_meta_grad = slow_meta_grad.reshape(grad.shape)

        if bias is not None:
            fast_meta_grad_dict[layer_name] = (layer_idx, fast_mata_grad_list, bias_grad.data)
            slow_meta_grad_dict[layer_name] = (layer_idx, slow_mata_grad_list, bias_grad.data)
        else:
            fast_meta_grad_dict[layer_name] = (layer_idx, fast_mata_grad_list, None)
            slow_meta_grad_dict[layer_name] = (layer_idx, slow_mata_grad_list, None)

        # Assigned pre_quantized_grads with meta grad for weights update
        # layer.pre_quantized_grads = meta_grad.data.clone()


    return fast_meta_grad_dict, slow_meta_grad_dict, history_grad, new_meta_hidden_state_dict

def dual_gradient_generation(meta_net, net, meta_method, history_grad=None, fix_meta=False):
    fast_meta_grad_dict = dict()
    slow_meta_grad_dict = dict()
    new_meta_hidden_state_dict = dict()

    layer_name_list = net.layer_name_list # 这里把主网络的层名字都拿出来了

    for idx, layer_info in enumerate(layer_name_list):

        layer_name = layer_info[0]
        layer_idx = layer_info[1]

        layer = get_layer(net, layer_idx)

        grad = layer.quantized_grads.data # 这里拿到这一层的梯度 (16,3,3,3)
        pre_quantized_weight = layer.pre_quantized_weight.data # 这里拿到这一层的权重大小 (16,3,3,3)
        bias = layer.bias # 这里拿到这一层的偏置

        if bias is not None:
            bias_grad = bias.grad.data.clone()
        else:
            bias_grad = None
            
        if meta_method in ['MetaDualGrad']:
            
            flatten_grad = grad.data.view(-1, 1)
            flatten_weight = pre_quantized_weight.data.view(-1, 1)
            grad_in = grad.data.view(1, -1, 1)
            weight_in = pre_quantized_weight.data.view(1, -1, 1)
            
            b,l,d = grad_in.shape
            
            if history_grad is not None and layer_name in history_grad:
                his_grad = history_grad[layer_name]
                # new_grad = 0.3 * his_grad + (1 - 0.3) * grad_in
                if his_grad.shape[1]/l == 5:
                    his_grad = torch.cat((his_grad[:,-l*4:,:], grad_in), 1)
                else:
                    his_grad = torch.cat((his_grad, grad_in), 1)
            else:
                his_grad = grad_in
                
            history_grad[layer_name] = his_grad
            
            if fix_meta:
                with torch.no_grad():
                    slow_grad, fast_grad = meta_net(flatten_weight, his_grad)
            else:
                slow_grad, fast_grad = meta_net(flatten_weight, his_grad)
                    
            slow_meta_grad = slow_grad[:, -grad_in.shape[1]:, :]
            fast_meta_grad = flatten_grad * fast_grad

        else:
            raise NotImplementedError

        # Reshape the flattened meta gradient into the original shape
        fast_meta_grad = fast_meta_grad.reshape(grad.shape)
        slow_meta_grad = slow_meta_grad.reshape(grad.shape)

        if bias is not None:
            fast_meta_grad_dict[layer_name] = (layer_idx, fast_meta_grad, bias_grad.data)
            slow_meta_grad_dict[layer_name] = (layer_idx, slow_meta_grad, bias_grad.data)
        else:
            fast_meta_grad_dict[layer_name] = (layer_idx, fast_meta_grad, None)
            slow_meta_grad_dict[layer_name] = (layer_idx, slow_meta_grad, None)
            
    return fast_meta_grad_dict, slow_meta_grad_dict, history_grad, new_meta_hidden_state_dict

def update_parameters(net, lr):
    for param in net.parameters():
        # if torch.sum(torch.abs(param.grad.data)) == 0:
        #     print('[Warning] Gradient is 0, missing assigned?')
        try:
            param.data.add_(-lr * param.grad.data)
        except:
            pass