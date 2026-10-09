# sys
import os
import sys
import numpy as np
import random
import pickle
from scipy import stats

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time
import tqdm
# operation
from . import tools


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        mmap: If true, use memory mapping to load data
        use_cache: If true, cache the data and label to avoid loading from disk every time
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 score_normalization=False,
                 max_score=1,
                 mmap=False,
                 use_cache=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.score_normalization = score_normalization
        self.max_score = float(max_score)
        self.use_cache = use_cache
        self.data_cache = None
        self.label_cache = None
        self.load_data(mmap)



    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        if self.score_normalization:
            self.label = np.array(self.label, dtype=np.float32) / self.max_score
        else:
            self.label = np.array(self.label, dtype=np.float32)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        self.data = np.array(self.data, dtype=np.float32)
        if self.use_cache:
            self.data_cache = []
            self.label_cache = []
            try:
                for i in tqdm.tqdm(range(len(self.label)), desc="cacheing"):
                    self.data_cache.append(np.array(self.data[i]))
                    self.label_cache.append(self.label[i])
                print("cache done")
            except Exception as e:
                print(f"cache fail: {e}")
                self.data_cache = None
                self.label_cache = None
                self.use_cache = False
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]
        if self.data.ndim == 5:
            self.N, self.C, self.T, self.V, self.M = self.data.shape
            self.window_size = self.T if self.window_size == -1 else self.window_size

    def __len__(self):
        return len(self.label)


    def __getitem__(self, index):
        # get data
        if self.use_cache and self.data_cache is not None and self.label_cache is not None:
            data_numpy = self.data_cache[index]
            label = self.label_cache[index]
        else:
            data_numpy = np.array(self.data[index])
            label = self.label[index]
        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
            #print("data_numpy1:", data_numpy.shape)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
            #print("data_numpy2:", data_numpy.shape)
        if self.random_move:
            
            data_numpy = tools.random_move(data_numpy)
            '''
            data_numpy = tools.random_dropout(data_numpy, 
                                                dropout_ratio=0.1, 
                                                joint_mask_ratio=0.3)
            '''
        '''
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0
        '''
        return data_numpy, label