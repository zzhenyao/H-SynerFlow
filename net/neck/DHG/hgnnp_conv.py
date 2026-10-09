import torch
import torch.nn as nn

from .hypergraph import Hypergraph
class HGNNPConv(nn.Module):
    

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        
        '''
        # num_e = 280
        device = next(self.parameters()).device
        weight = torch.ones(280 * 3, device=device)
        self.body_e_weights = nn.Parameter(weight, requires_grad=True)
        self.register_parameter('body_e_weights', self.body_e_weights) 
        '''
        

    def forward(self, X: torch.Tensor, 
                        hg: Hypergraph, 
                        _v2e_weight = None, 
                        _e2v_weight = None,
                        aggr='mean',
                        group_name='three'
                        ) -> torch.Tensor:
        r"""The forward function.
        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        
        
        X = hg.v2v_of_group(
            group_name=group_name,
            X=X, 
            v2e_weight=_v2e_weight,
            e2v_weight=_e2v_weight,
            aggr=aggr).contiguous() 
        '''
        X_res = self.theta(X)
        X_res = hg.v2v_of_group_res(
                group_name=group_name,
                X=X_res, 
                aggr=aggr).contiguous() 
        '''

        #X = X_1 + X_res
        #X = X_res
        #print('v2e_weight:', v2e_weight[:10])
        #print('e2v_weight:', e2v_weight[:10])
        
        
        return X
