import numpy as np
import random
import torch

def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    C, T, V, M = data_numpy.shape

    move_time = random.choice(move_time_candidate)

    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
        '''
        
            z_coords = data_numpy[2, i_frame, :, :]  # (V, M)
        
        data_numpy[0:2, i_frame, :, :] = new_xy_reshaped
        '''
    return data_numpy


def random_move_batch_vectorized(data_tensor,
                                angle_candidate=[-10., -5., 0., 5., 10.],
                                scale_candidate=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
                                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                                sample_ratio=0.5):
    """
    
    Args:
    
    Returns:
        
    """
    N, C, T, V, M = data_tensor.shape
    device, dtype = data_tensor.device, data_tensor.dtype
    N_t = int(N * sample_ratio)

    if N_t == 0:
        return data_tensor

    transform_indices = torch.randperm(N, device=device)[:N_t]

    angles = torch.tensor([random.choice(angle_candidate) for _ in range(N_t)], 
                         device=device, dtype=dtype) * torch.pi / 180  # (N_t,)
    scales = torch.tensor([random.choice(scale_candidate) for _ in range(N_t)], 
                         device=device, dtype=dtype)  # (N_t,)
    tx = torch.tensor([random.choice(transform_candidate) for _ in range(N_t)], 
                     device=device, dtype=dtype)  # (N_t,)
    ty = torch.tensor([random.choice(transform_candidate) for _ in range(N_t)], 
                     device=device, dtype=dtype)  # (N_t,)

    cos_a = torch.cos(angles)  # (N_t,)
    sin_a = torch.sin(angles)  # (N_t,)
    
    theta = torch.stack([
        torch.stack([cos_a * scales, -sin_a * scales], dim=1),
        torch.stack([sin_a * scales,  cos_a * scales], dim=1)
    ], dim=1)  # (N_t, 2, 2)

    xy = data_tensor[transform_indices, 0:2]  # (N_t, 2, T, V, M)
    
    original_shape = xy.shape
    xy_flat = xy.reshape(N_t, 2, -1)  # (N_t, 2, T*V*M)
    
    new_xy = torch.bmm(theta, xy_flat)  # (N_t, 2, T*V*M)
    new_xy = new_xy.to(dtype=dtype)
    
    new_xy = new_xy.reshape(original_shape)  # (N_t, 2, T, V, M)
    
    new_xy[:, 0] += tx.view(N_t, 1, 1, 1)
    new_xy[:, 1] += ty.view(N_t, 1, 1, 1)
    '''
        z_coords = data_numpy[2, i_frame, :, :]  # (V, M)
    
    data_numpy[0:2, i_frame, :, :] = new_xy_reshaped
    '''
    data_tensor[transform_indices, 0:2] = new_xy
    return data_tensor

def random_dropout_batch_shared_mask(data_tensor, dropout_ratio=0.1, joint_mask_ratio=0.3, sample_ratio=0.5):
    """
    
    
    Args:
    
    Returns:
        
    """
    N, C, T, V, M = data_tensor.shape
    device = data_tensor.device
    dtype = data_tensor.dtype
    
    num_transform_samples = int(N * sample_ratio)
    if num_transform_samples == 0:
        return data_tensor
        
    transform_indices = torch.randperm(N, device=device)[:num_transform_samples]
    
    num_dropout_frames = int(T * dropout_ratio)
    num_mask_joints = int(V * joint_mask_ratio)
    
    if num_dropout_frames == 0 or num_mask_joints == 0:
        return data_tensor
    
    frame_joint_mask = torch.stack([
        torch.randperm(V, device=device)[:num_mask_joints]
        for _ in range(T)
    ])  # (T, num_mask_joints)
    
    selected_frames = torch.randperm(T, device=device)[:num_dropout_frames]
    
    n_idx = transform_indices.unsqueeze(1).unsqueeze(2).expand(-1, num_dropout_frames, num_mask_joints).reshape(-1)
    
    t_idx = selected_frames.unsqueeze(1).expand(-1, num_mask_joints).unsqueeze(0).expand(num_transform_samples, -1, -1).reshape(-1)
    
    v_idx = frame_joint_mask[selected_frames].unsqueeze(0).expand(num_transform_samples, -1, -1).reshape(-1)
    
    data_tensor[n_idx, :, t_idx, v_idx, :] = 0
    
    return data_tensor

