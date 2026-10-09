import torch
import torch.nn as nn
import time
def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.1)
    elif name == 'elu':
        return nn.ELU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'glu':
        return nn.GLU()
    else:
        return nn.Identity()


_time_cache = {}
def start_timer(tag="default"):
    _time_cache[tag] = time.time()

def end_timer(tag="default"):
    if tag not in _time_cache:
        print(f"[{tag}] Timer was not started.")
        return
    elapsed = time.time() - _time_cache[tag]
    print(f"[{tag}] step time: {elapsed:.14f}s")
    del _time_cache[tag]