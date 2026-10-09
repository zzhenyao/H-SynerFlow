import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import torch.nn.init as init
from net.net_utils.net_utils import get_activation

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0,
        proj_drop=0,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        # ---------------------------------------------

        self.attn_drop = nn.Dropout(attn_drop)
        self.LN_q = nn.LayerNorm(dim)
        self.LN_k = nn.LayerNorm(dim)
        self.LN_v = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, latents, context, temp = 1):
        B, N_latents, C = latents.shape
        _ , N_context, _ = context.shape

        q = self.to_q(self.LN_q(latents))
        k = self.to_k(self.LN_k(context))
        #v = self.to_v(self.LN_v(context))
        #q = self.to_q(latents)
        #k = self.to_k(context)
        v = self.to_v(context)
        # (B, N, C) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.reshape(B, N_latents, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N_context, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N_context, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # (B, H, N_lat, D_h) @ (B, H, D_h, N_ctx) -> (B, H, N_lat, N_ctx)
        attn_score = (q @ k.transpose(-2, -1)) * self.scale / temp
        attn = attn_score.softmax(dim=-1)
        attn_maps = attn
        attn = self.attn_drop(attn)

        # (B, H, N_lat, N_ctx) @ (B, H, N_ctx, D_h) -> (B, H, N_lat, D_h)
        x = (attn @ v)
        
        # (B, H, N_lat, D_h) -> (B, N_lat, H, D_h) -> (B, N_lat, C)
        x = x.transpose(1, 2).reshape(B, N_latents, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_score, attn_maps



class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attention = Attention(dim, heads)

    def forward(self, latents, context, temp = 1):
        out, attn, attn_maps = self.attention(latents, context, temp = temp)
        return out, attn, attn_maps




class CosineTargetAggregator(nn.Module):
    def __init__(self, in_dim, target_dim=256, num_targets=5, t_min_dis=0.5):
        super().__init__()
        self.num_targets = num_targets
        self.target_dim = target_dim
        self.activation = nn.ReLU()
        self.t_min_dis = t_min_dis
        self.targets = nn.Parameter(self._init_diverse_targets(num_targets, target_dim))
        
        
        self.CrossAttention = CrossAttention(dim = target_dim, 
                                            heads=2)
        self.SelfAttention = CrossAttention(dim = target_dim,
                                            heads=1)
        self.CrossAttention_A = CrossAttention(dim = target_dim, 
                                            heads=2)
       
        self.weight_mlp = nn.Sequential(
            nn.Linear(256, 64),
            self.activation,
            nn.Linear(64, 1),
        )
        self.LN = nn.LayerNorm(target_dim)
    def freeze_targets(self):
        self.targets.requires_grad = False
    
    def forward(self, x, phase='train'):
        # x: (B, V, C)
        B, V, C = x.shape
        x_cos, att_score_1, attn_maps_1 = self.CrossAttention(self.targets.unsqueeze(0).expand(B, -1, -1), x)  # (B, K, C)
        x_cos_1 = x_cos
        targets_cos, att_score_self, attn_maps_self = self.SelfAttention(x_cos, x_cos, temp = 1)  # (B, K, C)
        weight = self.weight_mlp(targets_cos).squeeze(-1)  # (B, K,)
        weight_temperature = 1
        weight = F.softmax(weight / weight_temperature, dim=-1)
        x_cos_A, att_score_2, attn_maps_2 = self.CrossAttention_A(targets_cos, x)
        
        x_cos = x_cos_A
        total_loss_targets = self.compute_diversity_protection_loss_v3(self.targets, min_distance = self.t_min_dis)
        total_loss_attn_self = self._compute_inter_target_jsd_loss_batched(attn_maps_self, min_jsd_margin=0.00001, k = 1)
        total_loss_attn_entropy_self = self._compute_peak_suppression_loss_batched(attn_maps_self, max_peak_threshold=0.99, k=1)
      
        total_loss = total_loss_targets + total_loss_attn_self + total_loss_attn_entropy_self
        return x_cos, total_loss, weight  # (B, K, C), (B, K), scalar
    def _compute_peak_suppression_loss_batched(self, attn_maps, max_peak_threshold=0.9, k=1.0, eps=1e-9, max_clip=1e3):
        if attn_maps.ndim == 3:
            attn_maps = attn_maps.unsqueeze(1)
        
        max_peak = attn_maps.max(dim=-1).values  # (B,H,K)
        
        violation_mask = max_peak > max_peak_threshold
        if not violation_mask.any():
            return torch.tensor(0.0, device=attn_maps.device, dtype=attn_maps.dtype)
        
        peak_violators = max_peak[violation_mask]
        normalized_violation = (peak_violators - max_peak_threshold) / (1.0 - max_peak_threshold + eps)
        normalized_violation = torch.clamp(normalized_violation, max=1-eps)
        
        explosive_penalty = (1.0 / (1.0 - normalized_violation + eps) - 1.0) ** 2
        explosive_penalty = k * torch.clamp(explosive_penalty, max=max_clip)
        
        penalties = torch.zeros_like(max_peak)
        penalties[violation_mask] = explosive_penalty
        
        return penalties.mean()
    def _compute_inter_target_jsd_loss_batched(self, attn_maps, min_jsd_margin=0.1, eps=1e-9, k=1.0, max_clip=1e3):
        if attn_maps.ndim == 3:
            attn_maps = attn_maps.unsqueeze(1)  # -> (B, 1, K, N)
        elif attn_maps.ndim != 4:
            raise ValueError(f"Input attn_maps must be 3D or 4D, but got {attn_maps.shape}")

        B, H, K, N = attn_maps.shape
        if K <= 1:
            return torch.tensor(0.0, device=attn_maps.device)

        indices = torch.combinations(torch.arange(K, device=attn_maps.device))
        idx_p, idx_q = indices[:, 0], indices[:, 1]

        p = attn_maps[:, :, idx_p, :]
        q = attn_maps[:, :, idx_q, :]

        m = 0.5 * (p + q)
        log_m = torch.log(m + eps)
        kl_p_m = F.kl_div(log_m, p, reduction='none').sum(dim=-1)
        kl_q_m = F.kl_div(log_m, q, reduction='none').sum(dim=-1)
        jsd = 0.5 * (kl_p_m + kl_q_m)

        jsd_normalized = jsd / torch.log(torch.tensor(2.0, device=attn_maps.device))

        penalties = torch.zeros_like(jsd_normalized)
        violation_mask = jsd_normalized < min_jsd_margin

        if violation_mask.any():
            jsd_violators = jsd_normalized[violation_mask].clamp_min(eps)
            explosive_penalty = k * ((min_jsd_margin / jsd_violators) - 1.0) ** 2
            explosive_penalty = torch.clamp(explosive_penalty, max=max_clip)
            penalties[violation_mask] = explosive_penalty

        return penalties.mean()


    def compute_diversity_protection_loss_v3(self, targets, min_distance=0.5, eps=1e-6):
        def compute_single_loss(T):
            T = F.normalize(T, dim=-1)
            sim_matrix = torch.mm(T, T.t())
            K = T.size(0)
            mask = torch.triu(torch.ones(K, K, device=T.device), diagonal=1)
            sim_pairs = sim_matrix[mask.bool()]
            distance_pairs = 1 - sim_pairs

            penalties = torch.zeros_like(distance_pairs)
            mask = distance_pairs < min_distance
            
            if mask.any():
                source_values = (min_distance / (distance_pairs[mask] + eps)) - 1
                
                penalties[mask] = source_values.to(penalties.dtype)
                
            return penalties.mean()

        if targets.ndim == 2:
            return compute_single_loss(targets)
        elif targets.ndim == 3:
            B = targets.size(0)
            loss_list = [compute_single_loss(targets[b]) for b in range(B)]
            return torch.stack(loss_list).mean()
        else:
            raise ValueError(f"Unsupported targets shape: {targets.shape}")
   
    
    def _init_diverse_targets(self, num_targets, target_dim):
        return self._init_overlapping_semantic_targets(num_targets, target_dim)
    
    def _init_overlapping_semantic_targets(self, num_targets, target_dim):
        if num_targets == 1:
            return F.normalize(torch.randn(1, target_dim), dim=-1)
        
        base_directions = self._create_base_semantic_directions(num_targets, target_dim)
        
        targets = self._create_overlapping_clusters(base_directions, num_targets, target_dim)
        
        targets = self._adjust_semantic_overlap(targets, target_similarity_range=(0.3, 0.7))
        
        return targets
    
    def _create_base_semantic_directions(self, num_targets, target_dim):
        if num_targets <= 3:
            directions = torch.zeros(num_targets, target_dim)
            
            if num_targets >= 1:
                directions[0, :target_dim*2//3] = 1.0
                directions[0, target_dim*2//3:] = 0.3  
                
            if num_targets >= 2:
                start_idx = target_dim // 3
                directions[1, start_idx:start_idx + target_dim*2//3] = 1.0
                directions[1, :start_idx] = 0.2  
                directions[1, start_idx + target_dim*2//3:] = 0.2  
                
            if num_targets >= 3:
                directions[2, target_dim//3:] = 1.0
                directions[2, :target_dim//3] = 0.3  
                
        else:
            num_centers = min(3, num_targets)
            centers = torch.randn(num_centers, target_dim)
            centers = F.normalize(centers, dim=-1)
            
            directions = torch.zeros(num_targets, target_dim)
            for i in range(num_targets):
                center_idx = i % num_centers
                base_center = centers[center_idx]
                
                noise_strength = 0.4  
                noise = torch.randn_like(base_center) * noise_strength
                directions[i] = base_center + noise
        
        return F.normalize(directions, dim=-1)
    
    def _create_overlapping_clusters(self, base_directions, num_targets, target_dim):

        K, D = base_directions.shape
        
        if num_targets == K:
            return base_directions
        
        targets = []
        targets_per_base = num_targets // K
        remaining = num_targets % K
        
        for i, base_dir in enumerate(base_directions):
            num_variants = targets_per_base + (1 if i < remaining else 0)
            
            for j in range(num_variants):
                if j == 0:
                    targets.append(base_dir)
                else:
                    noise_strength = 0.3 + 0.2 * (j / num_variants)
                    noise = torch.randn_like(base_dir) * noise_strength
                    variant = base_dir + noise
                    targets.append(F.normalize(variant, dim=-1))
        
        return torch.stack(targets)
    
    def _adjust_semantic_overlap(self, targets, target_similarity_range=(0.3, 0.7), max_iter=100, alpha_move=0.1):

        K = targets.size(0)
        min_sim, max_sim = target_similarity_range
        
        num_bins = min(max(3, K), 12) 
        
        bin_edges = torch.linspace(min_sim, max_sim, num_bins + 1) 
        
        num_pairs = K * (K - 1) // 2
        target_count = num_pairs / num_bins 
        
        TOLERANCE_THRESHOLD = 0.05 * target_count 
        
        targets_next = targets.clone()
        
        for iteration in range(max_iter):
            sim_matrix = torch.matmul(targets, targets.T)
            
            mask = ~torch.eye(K, dtype=torch.bool)
            off_diag_sims = sim_matrix[mask]
            
            counts = torch.histc(off_diag_sims, bins=num_bins, min=min_sim, max=max_sim)
            
            count_deviations = counts - target_count
            
            
            for i in range(K):
                for j in range(i + 1, K):
                    sim_ij = sim_matrix[i, j].item()
                    
                    bin_idx = torch.bucketize(torch.tensor(sim_ij), bin_edges, right=True).item() - 1
                    
                    bin_idx = max(0, min(num_bins - 1, bin_idx))
                    
                    current_deviation = count_deviations[bin_idx].item()
                    
                    if sim_ij > max_sim or sim_ij < min_sim:
                        if sim_ij > max_sim:
                            force = alpha_move * 2.0 * (sim_ij - max_sim)
                            targets_next[j] = targets_next[j] + torch.randn_like(targets[j]) * force
                        else:
                            force = alpha_move * 2.0 * (min_sim - sim_ij)
                            targets_next[j] = targets_next[j] * (1.0 - force) + targets[i] * force
                            
                    elif abs(current_deviation) > 0.01 * target_count :
                        
                        if current_deviation > 0:
                            left_deviation = count_deviations[bin_idx - 1].item() if bin_idx > 0 else float('inf')
                            right_deviation = count_deviations[bin_idx + 1].item() if bin_idx < num_bins - 1 else float('inf')
                            
                            move_force = alpha_move * abs(current_deviation / target_count)
                            
                            if left_deviation < right_deviation:
                                adjustment = targets[j] * move_force
                                targets_next[j] = targets_next[j] - adjustment
                            else:
                                adjustment = targets[i] * move_force
                                targets_next[j] = targets_next[j] + adjustment
                                
                        elif current_deviation < 0:
                            pass 
            
            targets = F.normalize(targets_next, dim=-1)

        return targets
    
class Head(nn.Module):
    def __init__(self, dropout = 0, 
                        in_channels = 256, 
                        target_channels = 256,
                        num_targets = 5,
                        t_min_dis = 0.5, 
                        activation = 'relu'):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        fc_channels = {1024:[256,64], 512:[256,64], 256:[64,32], 4096:[1024,256], 2048:[512,128]}
        self.fc_final_score = nn.Sequential(
                nn.Linear(256, 128),
                self.activation,
                nn.Linear(128, 32),
                self.activation,
                nn.Linear(32, 1)
        )     
        self.fuse = CosineTargetAggregator(in_dim=in_channels, 
                                            target_dim = target_channels, 
                                            num_targets = num_targets,
                                            t_min_dis=t_min_dis)
    
    def freeze_targets(self):
        self.fuse.freeze_targets()
        
    def forward(self, x, phase='train'):
        r"""
        Args:
            x: N,T,C_in

        Returns:
            score: N, 1
        """
        x1, target_loss, weight = self.fuse(x, phase=phase)
        
        x_fused = torch.sum(weight.unsqueeze(-1) * x1, dim=1)  # (B, C)

        final_score = self.fc_final_score(x_fused).squeeze(-1)  # (B,)
        
        return final_score, target_loss
