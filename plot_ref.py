import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from tqdm import tqdm
# from mayavi import mlab
from matplotlib import pyplot as plt
import multiprocessing
from multiprocessing import Pool

from model import get_model
from model_dryvr import get_model as get_model_dryvr

import sys
sys.path.append('configs')

def mute():
    sys.stdout = open(os.devnull, 'w')

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--config', type=str,
                        default='drone')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false')
parser.set_defaults(use_cuda=True)
parser.add_argument('--pretrained', type=str)
parser.add_argument('--no_plot', type=str)
parser.add_argument('--seed', type=int, default=1024)

args = parser.parse_args()
np.random.seed(args.seed)

config = importlib.import_module('config_'+args.config)

X0 = config.sample_X0()
# print(X0)
# ref = config.simulate(config.get_init_center(X0))
ref = config.simulate([600, 0.1, np.pi/4, np.pi/4, 0.1, 7000])
# higher = np.array([600,  0.1, np.pi/4, np.pi/4,  0.1, 8000])

if ref.shape[1]-1 == 2:
    # mlab.plot3d(ref[:,1], ref[:,2], np.zeros_like(ref[:,0]), color=(1,0,0))
    plt.plot(ref[:,1], ref[:,2], color=(1,0,0))
plt.show()
