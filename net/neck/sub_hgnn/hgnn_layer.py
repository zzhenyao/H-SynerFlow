r'''
HyperGNN_layer:
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from .hgnn_block import HyperGNN_block
from net.neck.sub_hgnn.generate_hyperedges import *
import math
from net.neck.DHG.hgnnp_conv import HGNNPConv
from net.neck.DHG.hypergraph import Hypergraph
from net.net_utils import DEBUG_PROFILE
from net.net_utils.net_utils import start_timer, end_timer
class HyperGNN_layer(nn.Module):
    r"""
    HyperGNN_layer:

    """
    def __init__(self, in_channels=256, 
                        out_channels=1024, 
                        window_size=3,
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
                        is_mask=False
                        ):
        r"""
        HyperGNN_layer:
        Args:
        Returns:
            None
        """

        super().__init__()
        self.hgnn_block = HyperGNN_block(in_channels, out_channels,
                                window_size=window_size,
                              drop_rate=drop_rate,
                              is_last=is_last,
                              bias=bias,
                              use_bn=use_bn,
                              v2e_kernel=v2e_kernel,
                              v2e_stride=v2e_stride,
                              padding=padding,
                              e2v_kernel=e2v_kernel,
                              e2v_stride=e2v_stride,
                              e2v_kernel_kind=e2v_kernel_kind,
                              e2v_kernel_center=e2v_kernel_center,
                              is_mask=is_mask
                              )
                              
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size
        self.stride = stride

    def forward(self, x): 
        r"""
            input:
                (N, T, V, C)
            output:    
                (N, T, V, out_dim)
        """
        if DEBUG_PROFILE:
            start_timer("HyperGNN_layer forward reshape")
        N, T, V, C = x.shape
        w_half = self.window_size // 2
        pad_size = w_half
        zero_padding_l = torch.zeros((N, pad_size, V, C), device=x.device, dtype=x.dtype) 
        zero_padding_r = torch.zeros((N, pad_size, V, C), device=x.device, dtype=x.dtype)
        x = torch.cat([zero_padding_l, x[:, :, :, :],zero_padding_r], dim=1)
        N, new_T, V, C = x.shape
        #print(f"HyperGNN_layer forward x.shape: {x.shape}, T: {T}, V: {V}, C: {C}")
        x = x.reshape(N, new_T * V , C)
        if DEBUG_PROFILE:
            end_timer("HyperGNN_layer forward reshape")
            start_timer("HyperGNN_layer forward hgnn_block")
        x = self.hgnn_block(x, _V=V)
        if DEBUG_PROFILE:
            end_timer("HyperGNN_layer forward hgnn_block")
        #x = self.hgnn_layer2(x, frame_length=T)
        #x = self.hgnn_layer3(x, frame_length=T)
        #x = self.hgnn_layer4(x)
        #x = self.hgnn_layer5(x)
        # x: N,T, C, V
        #print(f"HyperGNN_layer forward x.shape after hgnn_block: {x.shape}")
        x = x.contiguous().reshape(N, T, V, self.out_channels)  # (N, T, V, out_dim) 
        
        return x  # or apply temporal pooling