import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from net.neck.hgnn.base_hgnn import SimpleInceptionHGNN, downsample2D
from net.net_utils.net_utils import get_activation
class InceptionFeatureExtractorHGNN(nn.Module):
    def __init__(self, in_channels, mid_channels=256, out_channels=512):
        super(InceptionFeatureExtractorHGNN, self).__init__()
        self.layer1 = SimpleInceptionHGNN(64, 128,
                                        out1_1=32,
                                        out2_1=48,  out2_2=64,
                                        out3_1=8,   out3_2=16,
                                        out4=16,
                                        is_residual=False)
        self.layer1_downsample = downsample2D(kernel_size=(3, 1), 
                                                    stride=(1, 1), 
                                                    padding=0,
                                                    downsample_type='avg')
        self.layer2 = SimpleInceptionHGNN(128, 240,
                                        out1_1=64,
                                        out2_1=64,  out2_2=96,
                                        out3_1=16,   out3_2=48,
                                        out4=32,
                                        is_residual=False)
        self.layer2_downsample = downsample2D(kernel_size=(3, 1), 
                                                    stride=(2, 1), 
                                                    padding=0,
                                                    downsample_type='avg')
        self.layer3 = SimpleInceptionHGNN(240, 256,
                                        out1_1=96,
                                        out2_1=48,  out2_2=104,
                                        out3_1=8,   out3_2=24,
                                        out4=32,
                                        is_residual=False)
        self.layer3_downsample = downsample2D(kernel_size=(3, 1), 
                                                    stride=(2, 1), 
                                                    padding=0,
                                                    downsample_type='avg')
        self.layer4 = SimpleInceptionHGNN(256, 256,
                                        out1_1=80,
                                        out2_1=56,  out2_2=112,
                                        out3_1=12,   out3_2=32,
                                        out4=32,
                                        is_residual=True)
        self.layer4_downsample = downsample2D(kernel_size=(3, 1), 
                                                    stride=(2, 1), 
                                                    padding=0,
                                                    downsample_type='avg')
        
        self.layer5 = SimpleInceptionHGNN(256, 256,
                                        out1_1=64,
                                        out2_1=64,  out2_2=128,
                                        out3_1=12,   out3_2=32,
                                        out4=32,
                                        is_residual=True,
                                        is_last=False)   
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        r"""
            input:
                (N, C_in, T, V)
            output:    
                (N, C_out, T, V)
        """
        x = self.layer1(x)
        x = self.layer1_downsample(x)
        x = self.layer2(x)
        x = self.layer2_downsample(x)
        x = self.layer3(x)
        x = self.layer3_downsample(x)
        x = self.layer4(x)
        x = self.layer4_downsample(x)
        x = self.layer5(x)
        #x = self.avg_pool(x)
        return x

class Neck(nn.Module):
    def __init__(self, 
                in_channels=256, 
                mid_channels=256,
                out_channels=1024,  
                window_size=3, 
                stride=1,
                dropout=0,
                activation='relu',
                **kwargs):
        super().__init__()
        self.ihgnn = InceptionFeatureExtractorHGNN(in_channels=in_channels, 
                                    mid_channels=mid_channels,
                                    out_channels= out_channels
                                    )
        self.activation = get_activation(activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size
        self.stride = stride
        self.proj = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True)
        )
        
     
    def forward(self, x): 
        # x shape: (N, T, C, V)
        # HGNN input:  x: N, Cin
        N, T, C, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, T, V, C)
        x = self.proj(x)
        
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)

        #x = x.permute(0, 2, 1, 3).contiguous()  # (N, C, T, V)
        x = self.ihgnn(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        x = x.mean(dim=1)
        return x  # or apply temporal pooling