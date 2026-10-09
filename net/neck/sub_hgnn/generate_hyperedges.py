import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
def get_group():  
    return [
        [0, 14, 15],
        [5, 6, 7],
        [11, 12, 13],
        [1, 16, 17],
        [2, 3, 4],
        [8, 9, 10],
    ]
def find_node_in_group(node_idx):
    groups = get_group()
    for group_idx, group in enumerate(groups):
        if node_idx in group:
            node_pos = group.index(node_idx)
            return group_idx, node_pos
    return None, None

def get_group_info_by_node(node_idx):
    groups = get_group()
    symmetry_map = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}

    group_id, pos_in_group = find_node_in_group(node_idx)


    same_group = group_id
    symmetry_group = symmetry_map[group_id]
    ordered_group = 0 if group_id == 5 else group_id + 1

    return same_group, symmetry_group, ordered_group, pos_in_group

def get_node_set_by_kernel3(node_idx):
    groups = get_group()
    same_group, symmetry_group, ordered_group, pos_in_group = get_group_info_by_node(node_idx)
    
    node_list = []
    node_list.append(node_idx)
    node_list.append(groups[symmetry_group][pos_in_group])
    node_list.append(groups[ordered_group][pos_in_group])

    return node_list
def get_node_set_by_kernel5_25(node_idx, t_th = 1):
    groups = get_group()
    same_group, symmetry_group, ordered_group, pos_in_group \
         = get_group_info_by_node(node_idx)
    node_list = []
    if t_th == 1:
        node_list.extend(groups[same_group])
        node_list.extend(groups[symmetry_group])
        node_list.append(groups[ordered_group][pos_in_group])
    else:
        node_list.extend(groups[same_group])
        node_list.append(groups[symmetry_group][pos_in_group])
        node_list.append(groups[ordered_group][pos_in_group])

    return node_list

def generate_multi_hyperedges_by_kernel(center_frame, 
                                        num_frames, 
                                        num_joints,
                                        w_frames,
                                        is_mask=False):
    temporal_hyperedges = []
    w_half = w_frames // 2
    #print("w_half:", w_half)
    for j in range(num_joints):
        center = center_frame * num_joints + j
        indices = [center]
        #print("center:", center)
        #print("center_frame:", center_frame)
        if w_half == 1:
            node_list = get_node_set_by_kernel3(j)
            for node_idx in node_list:
                indices.append((center_frame - 1) * num_joints + node_idx)
                if is_mask == False:
                    indices.append((center_frame + 1) * num_joints + node_idx)
        elif w_half == 2:
            node_list = get_node_set_by_kernel5_25(j, t_th=1)
            for node_idx in node_list:
                indices.append((center_frame - 1) * num_joints + node_idx)
                if is_mask == False:
                    indices.append((center_frame + 1) * num_joints + node_idx)
            node_list = get_node_set_by_kernel5_25(j, t_th=2)
            for node_idx in node_list:
                indices.append((center_frame - 2) * num_joints + node_idx)
                if is_mask == False:
                    indices.append((center_frame + 2) * num_joints + node_idx)

        if len(indices) > 1:
            indices.sort()
            temporal_hyperedges.append(indices)
        #print("center_frame:", center_frame)
        #print("temporal_hyperedges:", temporal_hyperedges)
        #print("indices:", indices)
    return temporal_hyperedges
