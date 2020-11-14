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

os.system('cp *.py '+args.log)
os.system('cp -r configs/ '+args.log)
os.system('cp -r examples/ '+args.log)

np.random.seed(1024)

config = importlib.import_module('config_'+args.config)

def PWD(normalized_dis, t):
    T = len(set(t.tolist()))
    N = int(len(t) / T)
    assert len(t) == T*N
    idx = t.argsort()
    t = t[idx]
    normalized_dis = normalized_dis[idx]
    t = t.reshape(T,N)[:,0]
    normalized_dis = normalized_dis.reshape(T,N)
    # normalized_dis: N x T
    normalized_dis = normalized_dis.max(axis = 1)
    t = np.array([0,] + t.tolist())
    normalized_dis = np.array([1., ] + normalized_dis.tolist())
    y = np.log(normalized_dis)
    K = 1
    y = y-np.log(K)
    dy = y[1:] - y[:-1]
    dt = t[1:] - t[:-1]

    gamma = dy / dt

    return gamma, t

# train_loader, val_loader = get_dataloader(30, 5, 4096)
train_loader, val_loader = get_dataloader(config, args.num_train, args.num_test, args.batch_size, [args.data_file_train, args.data_file_eval])

normalized_dis = []
t = []
for X0, R, Xi0, Xi1, T in train_loader:
    DXi = config.observe_for_output(Xi1 - Xi0).cpu().detach().numpy()
    dis = np.sqrt((DXi**2).sum(axis=1)).reshape(-1)
    R = R.cpu().detach().numpy().reshape(-1)
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
