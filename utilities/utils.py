import yaml
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision.models

def readConfig(file: str) -> dict:
    with open(file, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_dir(id):
    sub_dir = ['master', 'pred', 'logs','model', 'runs']
    for dir in sub_dir:
        path = os.path.join('results', id, dir)
        if not os.path.exists(path):
            os.makedirs(path, 0o777)

def data_dir(id, file):
    data_dir = os.path.join('results', id, 'master', file)
    return data_dir

def model_dir(id):
    model_dir = os.path.join('results', id, 'model', id+'.pt')
    return model_dir

def log_dir(id, dir_id):
    sr_log_dir = os.path.join('results', id, 'runs', dir_id)
    return sr_log_dir

def WideResnet50(num_class, ema=False):
    model = torchvision.models.wide_resnet50_2(pretrained=True)
    model.fc =  nn.Linear(2048, num_class)
    if ema:
        for param in model.parameters():
            param.requires_grad = False
    return model

class Logger():
    def __init__(self, id):
        self.id = id

    def logging(self, output):
        log_dir = os.path.join('results', self.id, 'logs', 'logs.txt')
        with open(log_dir, 'a+') as f:
            print (output, file=f)
