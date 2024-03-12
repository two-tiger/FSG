"""
Some helper function for meta inference
"""

import torch
from utils.miscellaneous import get_layer
import numpy as np


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
            
        elif meta_method == 'MetaMamba':
            
            grad_in = grad.data.view(1, -1, 1)
            weight_in = pre_quantized_weight.data.view(1, -1, 16)
            
            l = grad_in.shape[1]
            
            if history_grad is None:
                his_grad = grad_in
            else:
                his_grad = torch.cat((history_grad, grad_in), 1)

            if fix_meta:
                with torch.no_grad():
                    meta_output = meta_net(his_grad)
            else:
                meta_output = meta_net(his_grad)

            meta_output = meta_output[:, -l, :]
            meta_grad = grad_in * meta_output
            
        
        else:
            raise NotImplementedError

        # Reshape the flattened meta gradient into the original shape
        meta_grad = meta_grad.reshape(grad.shape)

        if bias is not None:
            meta_grad_dict[layer_name] = (layer_idx, meta_grad, bias_grad.data)
        else:
            meta_grad_dict[layer_name] = (layer_idx, meta_grad, None)

        # Assigned pre_quantized_grads with meta grad for weights update
        layer.pre_quantized_grads = meta_grad.data.clone()

    return meta_grad_dict, new_meta_hidden_state_dict, new_momentum_dict, his_grad


def update_parameters(net, lr):
    for param in net.parameters():
        # if torch.sum(torch.abs(param.grad.data)) == 0:
        #     print('[Warning] Gradient is 0, missing assigned?')
        param.data.add_(-lr * param.grad.data)