import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from utils import AverageMeter
from tensorboardX import SummaryWriter

from data import get_dataloader
from model import get_model

import sys
sys.path.append('configs')

import argparse

np.random.seed(1024)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--config', type=str,
                        default='drone')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false')
parser.set_defaults(use_cuda=True)
parser.add_argument('--bs', dest='batch_size', type=int, default=256)
parser.add_argument('--num_train', type=int, default=10)
parser.add_argument('--num_test', type=int, default=5)
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01)
parser.add_argument('--lambda1', dest='_lambda1', type=float, default=0.1)
parser.add_argument('--lambda2', dest='_lambda2', type=float, default=0.1)
parser.add_argument('--alpha', dest='alpha', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr_step', type=int, default=5)
parser.add_argument('--pretrained', type=str)
parser.add_argument('--data_file_train', type=str)
parser.add_argument('--data_file_eval', type=str)
parser.add_argument('--log', type=str)

args = parser.parse_args()

os.system('mkdir '+args.log)
os.system('echo "%s" > %s/cmd.txt'%(' '.join(sys.argv), args.log))
os.system('cp *.py '+args.log)
os.system('cp -r configs/ '+args.log)
os.system('cp -r examples/ '+args.log)

np.random.seed(1024)

config = importlib.import_module('config_'+args.config)

ACC = 0.97

def PWD(normalized_dis, t):
    T = np.sort(list(set(t.tolist())))
    num_t = len(T)
    DIS = np.zeros(num_t)
    for idx_t in range(num_t):
        idx = np.where(t==T[idx_t])[0]
        dis = normalized_dis[idx]
        dis = np.sort(dis)
        idx = int(len(dis)*ACC)
        if idx == len(dis):
            idx -= 1
        DIS[idx_t] = dis[idx]

    # import ipdb;ipdb.set_trace()

    T = np.array([0,] + T.tolist())
    DIS = np.array([1, ] + DIS.tolist())
    y = np.log(DIS)
    K = 1
    y = y-np.log(K)
    dy = y[1:] - y[:-1]
    dt = T[1:] - T[:-1]

    gamma = dy / dt

    return gamma, T

# train_loader, val_loader = get_dataloader(30, 5, 4096)
train_loader, val_loader = get_dataloader(config, args.num_train, args.num_test, args.batch_size, [args.data_file_train, args.data_file_eval])

normalized_dis = []
t = []
for (X0, T, ref, xt) in train_loader:
    DXi = (xt - ref).cpu().detach().numpy()
    dis = np.sqrt((DXi**2).sum(axis=1)).reshape(-1)
    R = X0[:,-1].cpu().detach().numpy().reshape(-1)
    T = T.cpu().detach().numpy().reshape(-1)
    normalized_dis.append(dis/R)
    t.append(T)

# import ipdb;ipdb.set_trace()

normalized_dis = np.concatenate(normalized_dis)
t = np.concatenate(t)

gammas, t = PWD(normalized_dis, t)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    filename = args.log + '/' + filename
    torch.save(state, filename)

save_checkpoint({'state_dict': [gammas, t]})
