r'''
HyperGNN_block:

    w_v2e:      ('same','')

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from net.neck.sub_hgnn.generate_hyperedges import *
import math
from net.neck.DHG.hgnnp_conv import HGNNPConv
from net.neck.DHG.hypergraph import Hypergraph

class HyperGNN_batch(nn.Module):
    def __init__(self, in_channels=256, 
                        out_channels=1024, 
                        bias=True,
                        use_bn=False,
                        drop_rate=0,
                        is_last=False
                        ):
        super().__init__()
        self.hgnn = HGNNPConv(in_channels,
                              out_channels,
                              bias=bias,
                              use_bn=use_bn,
                              drop_rate=drop_rate,
                              is_last=is_last)
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward(self, X, hg, 
                    _v2e_weight=None, 
                    _e2v_weight=None,
                    aggr='sum',
                    group_name='three'
                    ):
        r"""
        X: N,T*V, C_in
        """
        N, TV, C = X.shape
        futures = []
        for i in range(N):
            x_b = X[i]
            hg_b = hg 
            futures.append(torch.jit.fork(self.hgnn, x_b, hg_b, 
                                                    _v2e_weight, 
                                                    _e2v_weight,
                                                    aggr=aggr,
                                                    group_name=group_name))
        outputs = [torch.jit.wait(fut) for fut in futures]
        outputs = torch.stack(outputs, dim=0).contiguous() 
        return outputs


class HyperGNN_block(nn.Module):
    def __init__(self, in_channels=256, 
                        out_channels=1024, 
                        window_size=5,
                        stride=1,
                        bias=True,
                        use_bn=False,
                        drop_rate=0,
                        is_last=False,
                        v2e_kernel=3,
                        v2e_stride=1,
                        padding=(0,0),
                        e2v_kernel=3,
                        e2v_stride=1,
                        e2v_kernel_kind = 'one',
                        e2v_kernel_center = 0,
                        is_mask = False 
                        ):
        super().__init__()
        self.hgnn_batch = HyperGNN_batch(in_channels,
                        out_channels,
                        bias=bias,
                        use_bn=use_bn,
                        drop_rate=drop_rate,
                        is_last=is_last)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size
        self.stride = stride
        self.all_hypergraph = None
        self.is_mask = is_mask
        self.init_hypergraph_weights2D(out_channels, in_channels, v2e_kernel, e2v_kernel, e2v_kernel_kind, e2v_kernel_center)
    '''
    def get_tensor_shape(self, 
                        shape_0 = 1,
                        shape_1 = 1, 
                        shape_2 = 0,
                        center=0,
                        init_way='kaiming'):
        """
        """
        if shape_2 != 0:
            shape = [shape_0, shape_1, shape_2]
        else:
            shape = [shape_0, shape_1]
        device = next(self.parameters()).device
        if init_way == 'kaiming':
            tensor = torch.empty(shape, device=device)
            init.kaiming_uniform_(tensor, a=math.sqrt(5))
        elif init_way == 'normal':
            tensor = torch.empty(shape, device=device)
            init.normal_(tensor, mean=0.0, std=0.02)
        elif init_way == 'ones':
            tensor = torch.ones(shape, device=device)
        elif init_way == 'zeros_one':
            tensor = torch.zeros(shape, device=device)
            tensor[center][0] = 1.0
        return tensor  
    '''
    def get_tensor_shape(self, 
                     shape_0 = 1,
                     shape_1 = 1, 
                     shape_2 = 0,
                     center=0,
                     init_way='kaiming'):
        device = next(self.parameters()).device

        if shape_2 != 0:
            if init_way == 'kaiming':
                temp_weight = torch.empty([shape_1, shape_2, shape_0, 1], device=device)
                init.kaiming_uniform_(temp_weight, a=math.sqrt(5))
                tensor = temp_weight.squeeze(-1).permute(2, 0, 1)  # -> [shape_0, shape_1, shape_2]
            elif init_way == 'normal':
                tensor = torch.empty([shape_0, shape_1, shape_2], device=device)
                init.normal_(tensor, mean=0.0, std=0.02)
            elif init_way == 'ones':
                tensor = torch.ones([shape_0, shape_1, shape_2], device=device)
            elif init_way == 'zeros_one':
                tensor = torch.zeros([shape_0, shape_1, shape_2], device=device)
                tensor[center][0][0] = 1.0
        else:
            if init_way == 'kaiming':
                tensor = torch.empty([shape_0, shape_1], device=device)
                init.kaiming_uniform_(tensor, a=math.sqrt(5))
            elif init_way == 'normal':
                tensor = torch.empty([shape_0, shape_1], device=device)
                init.normal_(tensor, mean=0.0, std=0.02)
            elif init_way == 'ones':
                tensor = torch.ones([shape_0, shape_1], device=device)
            elif init_way == 'zeros_one':
                tensor = torch.zeros([shape_0, shape_1], device=device)
                tensor[center][0] = 1.0

        return tensor


    def init_hypergraph_weights2D(self, out_channels,
                                        in_channels, 
                                        v2e_kernel,
                                        e2v_kernel,
                                        e2v_kernel_kind, 
                                        e2v_kernel_center
                                       ):
        device = next(self.parameters()).device
        # weights
        
        temp_weight = self.get_tensor_shape(v2e_kernel, out_channels, in_channels, init_way='kaiming')
        self.v2e_weights = nn.Parameter(temp_weight, requires_grad=True) 

        # e2v_weights
        if e2v_kernel_kind == 'one':
            temp_weight = self.get_tensor_shape(e2v_kernel, 1, center = e2v_kernel_center, init_way='zeros_one').view(-1).repeat(18)
            self.e2v_weights = nn.Parameter(temp_weight, requires_grad=False)  
        elif e2v_kernel_kind == 'same':
            temp_weight = self.get_tensor_shape(e2v_kernel, init_way='kaiming').view(-1).repeat(18)
            self.e2v_weights = nn.Parameter(temp_weight, requires_grad=False)
        
    def init_hypergraph(self, new_T, V, is_mask=False):
        w_half = self.window_size // 2
        self.all_hypergraph = Hypergraph(num_v=new_T * V)
        for t in range(0 + w_half, new_T - w_half, self.stride):
            center_frame = t
            self.generate_hypergraph(center_frame, 
                                        new_T, 
                                        self.window_size, 
                                        V,
                                        is_mask=is_mask)
    def generate_hypergraph(self,
                            center_frame, 
                            num_frames, 
                            w_frames, 
                            num_joints,
                            is_mask=False):
        group_name = {3: 'three', 5: 'five'}

        hyperedges_list = generate_multi_hyperedges_by_kernel(
                                        center_frame=center_frame, 
                                        num_frames=num_frames, 
                                        num_joints=num_joints,
                                        w_frames=w_frames,
                                        is_mask=is_mask)
        hedges_length = len(hyperedges_list[0])
        self.all_hypergraph.add_hyperedges(e_list=hyperedges_list,
                                w_list_v2e=None,
                                w_list_e2v=None,  
                                e_weight=None,
                                merge_op='mean',
                                group_name='three')
        return 
    def forward(self, x, _V=1): 
        # x shape: (N, T, C, V)
        # HGNN input:  x: N, Cin


        N, TV, C = x.shape 
        if self.all_hypergraph is None:
            self.init_hypergraph(TV // _V, _V, is_mask = self.is_mask)
        
        
        '''
        print(" ")
        print("v2e_weight:", _v2e_weight[:2, :1, :1], _v2e_weight[120:122, :1, :1])
        print("v2e_weight:", _v2e_weight[12:14, :1, :1], _v2e_weight[132:134, :1, :1])
        print("e2v_weight:", _e2v_weight[:12], _e2v_weight[120:132])
        print("e2v_weight:", _e2v_weight[12:19], _e2v_weight[132:141])
        '''
        v2e_weights = self.v2e_weights
        #print("v2e_weights:", v2e_weights.shape)
        new_T = TV // _V - 2 * (self.window_size // 2)
        e2v_weights = self.e2v_weights.repeat(new_T)
        x = self.hgnn_batch(x, self.all_hypergraph, 
                            v2e_weights,
                            e2v_weights,
                            aggr='mean',
                            group_name='three'
                            )  # (V, out_dim)
        return x  # or apply temporal pooling