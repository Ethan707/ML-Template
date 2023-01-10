import wandb
import torch
import random
import numpy as np
import argparse
import os
from tqdm import tqdm
from utils.utils import Argument, set_seed, set_cudnn_config, save_model
import yaml
from datetime import datetime
import importlib
import logging


def set_optimizer():
    # set optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def set_schedular():
    if args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    return scheduler


def get_dataloader():
    dataset = importlib.import_module("datasets."+args.dataset)
    train_dataset = dataset.Dataset(args, "train")
    test_dataset = dataset.Dataset(args, "test")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader


def train():
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def validate():
    pass


if __name__ == "__main__":
    # parser the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="config file name")
    arg = parser.parse_args()
    config_path = os.path.join("./configs", arg.config)
    args = Argument(yaml.safe_load(open(config_path)))

    # set seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)
    set_cudnn_config(args.enable_cudnn)

    time = datetime.now().isoformat()[:19]

    model = importlib.import_module("models."+args.net)
    model = model.Model(args)
    model.to(args.device)
    criterion = model.Criterion()

    optimizer = set_optimizer()
    schedular = set_schedular()

    train_loader, val_loader = get_dataloader()

    for epoch in range(args.epochs):
        train_result = train()
        val_result = validate()
        schedular.step()
    
    wandb.finish()
    # os.system("shutdown -s -t 0") # work for server
