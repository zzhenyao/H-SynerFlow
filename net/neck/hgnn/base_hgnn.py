import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from net.neck.DHG.hgnnp_conv import HGNNPConv
from net.neck.DHG.hypergraph import Hypergraph
from net.neck.sub_hgnn.hgnn_block import HyperGNN_block
from net.neck.sub_hgnn.hgnn_layer import HyperGNN_layer
from net.neck.sub_hgnn.generate_hyperedges import *
from net.net_utils.net_utils import get_activation

import math

class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Unit2Dpy(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 activation='relu',
                 padding='same',
                 use_bias=False,
                 use_bn=True,
                 use_ac=True):
        super(Unit2Dpy, self).__init__()
        if padding.lower() == 'same':
            padding_val = kernel_size[0] // 2, kernel_size[1] // 2
        else:
            padding_val = (0, 0)

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_val,
            bias=use_bias
        )
        self.use_bn = use_bn
        self.use_ac = use_ac
        if self.use_bn:
            #self.bn = nn.BatchNorm2d(out_channels)
            self.bn = nn.BatchNorm1d(out_channels)
        if self.use_ac:
            self.activation = get_activation(activation)

    def forward(self, x):
        
        x = self.conv(x)
        N, C, T, V = x.shape
        if self.use_bn:
            x = x.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
            x = x.view(N * V, C, T)
            x = self.bn(x)
            x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous() 
        if self.use_ac:
            x = self.activation(x)
        return x


class downsample2D(nn.Module):
    def __init__(self, 
                kernel_size=(1, 1), 
                stride=(1, 1), 
                padding=0,
                downsample_type='avg'):
        super(downsample2D, self).__init__()
        p1 = (kernel_size[0] - 1) // 2
        p2 = (kernel_size[1] - 1) // 2  
        if downsample_type == 'avg':
            self.downsample = nn.AvgPool2d(kernel_size=kernel_size,
                                        stride=stride,
                                        padding=(p1, p2))   
        elif downsample_type == 'max':
            self.downsample = nn.MaxPool2d(kernel_size=kernel_size,
                                        stride=stride,
                                        padding=(p1, p2))   
        else:
            raise ValueError("Unsupported downsample type: {}".format(downsample_type)) 
   
        
    def forward(self, x):
        return self.downsample(x)

class HyperGNN_branch(nn.Module):
    def __init__(self, in_channels, 
                        out_channels, 
                        window_size=3, 
                        stride=1,
                        bias=True,
                        use_bn=False,
                        use_ac=False,
                        drop_rate=0,
                        is_last=False,
                        v2e_kernel=3,
                        e2v_kernel=3,
                        e2v_kernel_kind = 'one',
                        e2v_kernel_center = 3 // 2,
                        activation='relu',
                        is_mask=False):
        super(HyperGNN_branch, self).__init__()
        self.branch_layer = HyperGNN_layer(in_channels=in_channels, 
                                    out_channels=out_channels,
                                    window_size=window_size,
                                    stride=stride,
                                    bias=bias,
                                    use_bn=False,
                                    drop_rate=drop_rate,
                                    is_last=is_last,
                                    v2e_kernel=v2e_kernel,
                                    e2v_kernel=e2v_kernel,
                                    e2v_kernel_kind = e2v_kernel_kind,
                                    e2v_kernel_center = e2v_kernel_center,
                                    is_mask=is_mask
                                    )
        self.use_bn = use_bn
        self.use_ac = use_ac
        if self.use_bn:
            #self.bn = nn.BatchNorm2d(out_channels)
            self.bn = nn.BatchNorm1d(out_channels)
        if self.use_ac:
            self.activation = get_activation(activation)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        x = self.branch_layer(x)  # (N, T, V, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)
        N, C, T, V = x.shape
        if self.use_bn:
            x = x.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
            x = x.view(N * V, C, T)
            x = self.bn(x)
            x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous() 
        if self.use_ac:
            x = self.activation(x)
        return x  # (N, C, T, V)


class SimpleInceptionHGNN(nn.Module):
    def __init__(self, in_channels, 
                        out_channels,
                        out1_1=16,
                        out2_1=16,
                        out2_2=16,
                        out3_1=16,
                        out3_2=16,
                        out4=16,
                        is_residual=False,
                        branch4_type='avg',
                        is_last=False):
        super(SimpleInceptionHGNN, self).__init__()
        if is_residual:
            use_bn = True
            use_ac = False
            '''
            self.res_project = Unit2Dpy(in_channels, 
                                out_channels, 
                                kernel_size=(1, 1),
                                use_bn=use_bn,    
                                use_ac=use_ac)   
            '''
            #self.bn = nn.BatchNorm2d(out_channels)
            self.bn = nn.BatchNorm1d(out_channels)
            if is_last:
                self.activation = lambda x: x
            else:
                self.activation = get_activation('relu')
        else:
            use_bn = True
            use_ac = True
        self.is_residual = is_residual
        
        self.branch1 = Unit2Dpy(in_channels, 
                                out1_1, 
                                kernel_size=(1, 1),
                                use_bn=use_bn,    
                                use_ac=use_ac)

        self.branch2 = nn.Sequential(
            Unit2Dpy(in_channels, 
                                out2_1, 
                                kernel_size=(1, 1),
                                use_bn=True,    
                                use_ac=True),
            HyperGNN_branch(in_channels=out2_1, 
                                    out_channels= out2_2,
                                    window_size=3,
                                    stride=1,
                                    bias=True,
                                    use_bn=use_bn,
                                    use_ac=use_ac,
                                    drop_rate=0,
                                    is_last=False,
                                    v2e_kernel=7,
                                    e2v_kernel=3,
                                    e2v_kernel_kind = 'one',
                                    e2v_kernel_center = 3 // 2
                                    )
        )
        
        self.branch3 = nn.Sequential(
            Unit2Dpy(in_channels, 
                                out3_1, 
                                kernel_size=(1, 1),
                                use_bn=True,    
                                use_ac=True),
            HyperGNN_branch(in_channels=out3_1, 
                                    out_channels= out3_2,
                                    window_size=5,
                                    stride=1,
                                    bias=True,
                                    use_bn=use_bn,
                                    use_ac=use_ac,
                                    drop_rate=0,
                                    is_last=False,
                                    v2e_kernel=25,
                                    e2v_kernel=5,
                                    e2v_kernel_kind = 'one',
                                    e2v_kernel_center = 3 // 2
                                    )
        )
        
        #self.branch4_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0))
        if branch4_type == 'avg':
            self.branch4_pool = nn.AvgPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0))
        elif branch4_type == 'max':
            self.branch4_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0))
        else:
            raise ValueError("Unsupported branch4_type: {}".format(branch4_type))
        self.branch4_conv = Unit2Dpy(in_channels, 
                                    out4, 
                                    kernel_size=(1, 1),
                                    use_bn=use_bn,
                                    use_ac=use_ac)
        
    def forward(self, x):
        r"""
            input:
                (N, C_in, T, V)
            output:    
                (N, C_out, T, V)
        """
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4_conv(self.branch4_pool(x))
        out = torch.cat([b1, b2, b3, b4], dim=1)
        if self.is_residual:
            
            N, C, T, V = out.shape
            out = x + out
            out = out.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
            out = out.view(N * V, C, T)
            out = self.bn(out)
            out = out.view(N, V, C, T).permute(0, 2, 3, 1).contiguous() 
            
            out = self.activation(out)
        return out





