import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import torch.nn.init as init

from net.backbone.stgcn_utils.graph import *
from net.backbone.stgcn_utils.base_stgcn import *


class STGCN(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, 
                        graph_args,
                        edge_importance_weighting, 
                        weights,
                        ignore_weights,
                        activation = 'relu',
                        **kwargs):
        super(STGCN, self).__init__()



        self.graph = Graph(**graph_args)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))


        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn_networks = nn.ModuleList((
            graphconv_block(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            graphconv_block(64, 64, kernel_size, 1, **kwargs),
            graphconv_block(64, 64, kernel_size, 1, **kwargs),
            graphconv_block(64, 64, kernel_size, 1, **kwargs),
            graphconv_block(64, 128, kernel_size, 2, **kwargs),
            graphconv_block(128, 128, kernel_size, 1, **kwargs),
            graphconv_block(128, 128, kernel_size, 1, **kwargs),
            graphconv_block(128, 256, kernel_size, 2, **kwargs),
            graphconv_block(256, 256, kernel_size, 1, **kwargs),
            graphconv_block(256, 256, kernel_size, 1, **kwargs),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])

        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        '''
        for name, param in self.named_parameters():
            if 'model_lstm' not in name:
                param.requires_grad = False
        '''


    def forward(self, x):
        r"""

        Args:
            x: N, C, T, V, M

        Returns:
            feats:n,m,c,t,v
        """

        N, C, T, V, M = x.size()

        # N, M, V, C, T
        x = x.permute(0, 4, 3, 1, 2).contiguous()

        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)


        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        _, c, t, v = x.size()

        x = x.contiguous().view(N, M, c, t, v)
        return x

