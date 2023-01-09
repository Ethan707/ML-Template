import wandb
import torch
import random
import numpy as np
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_cudnn_config(flag):
    torch.backends.cudnn.enabled = flag
    torch.backends.cudnn.benchmark = flag


def set_wandb_config(config):
    wandb.config.update(config)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    pass