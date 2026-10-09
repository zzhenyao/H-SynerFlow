r"""
target model for clip-based action quality assessment
with target loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import torch.nn.init as init
from torchlight.io import import_class
from feeder import tools


class Model(nn.Module):
    r"""

    """

    def __init__(self, 
                    model_args, 
                    backbone_args, 
                    featureadapter_args, 
                    neck_args, 
                    head_args):
        super(Model, self).__init__()
        self.activation = model_args['activation']
        self.clip_num = model_args['clip_num']
        self.overlap_ratio = model_args['overlap_ratio']
        self.random_choose = model_args.get('random_choose', False)
        self.random_move = model_args.get('random_move', False)
        self.random_dropout = model_args.get('random_dropout', False)
        self.random_shift = model_args.get('random_shift', False)
        backbone_class = import_class(backbone_args['type'])
        self.backbone = backbone_class(**backbone_args['args'])
        
        featureadapter_class = import_class(featureadapter_args['type'])
        self.feature_adapter = featureadapter_class(**featureadapter_args['args'])

        neck_class = import_class(neck_args['type'])
        self.neck = neck_class(**neck_args['args'])
        
        head_class = import_class(head_args['type'])
        self.head = head_class(**head_args['args'])
        self.init_weights()
        feature_adapter_params = sum(p.numel() for p in self.feature_adapter.parameters())
        neck_params = sum(p.numel() for p in self.neck.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total_params = feature_adapter_params + neck_params + head_params
        print(f"Total parameters: {total_params/1e6:.3f}M")
        print(f"Feature Adapter parameters: {feature_adapter_params/1e6:.3f}M")
        print(f"Neck parameters: {neck_params/1e6:.3f}M")
        print(f"Head parameters: {head_params/1e6:.3f}M")

    def init_weights(self):
        for module in list(self.feature_adapter.modules()) \
                            + list(self.neck.modules()) \
                            + list(self.head.modules()):
            if isinstance(module, nn.Linear):
                if self.activation == 'relu':
                    init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                elif self.activation == 'leaky_relu':
                    init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                elif self.activation == 'elu':
                    init.kaiming_normal_(module.weight, nonlinearity='elu')
                else:
                    init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm1d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

            elif isinstance(module, nn.LayerNorm):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.constant_(param, 0)
    def forward(self, x, phase='train'):
        r"""

        Args:
            x: N, C, T, V, M

        Returns:
            socre
        """
        N,C,T,V,M = x.size()

        total_frames = T
        num_clips = self.clip_num
        overlap_ratio = self.overlap_ratio

        clip_length = int(total_frames / (1 + (num_clips - 1) * (1 - overlap_ratio)))
        step_size = int(clip_length * (1 - overlap_ratio))
        start_idx = [i * step_size for i in range(num_clips)]

        clips = torch.stack([x[:, :, start:start + clip_length] for start in start_idx], dim=1).contiguous() 
        
        clips = clips.view(N*num_clips, C, clip_length, V, M)
        #  N*num_clips, C, clip_length, V, M -> N*num_clips, M, C, clip_length, V
        
        if phase == 'train':
            if self.random_move:
                clips = tools.random_move_batch_vectorized(clips, sample_ratio=0.5)
            if self.random_dropout:
                clips = tools.random_dropout_batch_shared_mask(clips,
                                                                dropout_ratio=0.4, 
                                                                joint_mask_ratio=0.1,
                                                                sample_ratio=0.5)
        x = self.backbone(clips)  
        _, M, C_out, T_out, V_out = x.size()
        x = x.contiguous().view(N, num_clips, M, C_out, T_out, V_out)  
        x = self.feature_adapter(x, phase = phase)  
        x = self.neck(x)
        x, target_loss = self.head(x, phase = phase)
        
        return x, target_loss


    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
