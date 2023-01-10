import torch
import os
import shutil
import random
import numpy as np

class Argument(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

    def __str__(self) -> str:
        s = ''
        for key in self.__dict__:
            s += f'{key}: {self.__dict__[key]}'
        return str(s)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_cudnn_config(flag):
    torch.backends.cudnn.enabled = flag
    torch.backends.cudnn.benchmark = flag

def save_model(model, path, is_best, **kwargs):
    state = {'net': model.state_dict()}
    for key, value in kwargs.items():
        state[key] = value
    last_model_path = os.path.join(path, 'last_checkpoint.pth')
    torch.save(state, last_model_path)
    if is_best:
        best_model_path = os.path.join(path, 'best_checkpoint.pth')
        shutil.copyfile(last_model_path, best_model_path)
