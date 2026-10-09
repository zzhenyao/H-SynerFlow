r"""


input:
    N, num_clips, M, C, T, V
output:
    N, num_clips, C, V
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import torch.nn.init as init
import random

from net.net_utils.net_utils import get_activation
class FeatureAdapter(nn.Module):
    def __init__(self, dropout=0,
                        activation='relu',
                        **kwargs):
        super(FeatureAdapter, self).__init__()
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, phase = 'train'):
        r"""
        Args:
            x: N,num_clips, M,C,T,V

        Returns:
            x: N, num_clips, C, V
        """
        N,num_clips,M,C,T,V = x.size()

        clips = x.mean(dim=2)
        clips = clips.mean(dim=3)
        clips = self.dropout(clips)
        return clips
    
