# from dreal import *
import torch
import torch.nn.functional as F
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.tri import Triangulation
from scipy.spatial import ConvexHull
from utils import AverageMeter, ellipsoid2AArectangle, samplePointsOnAARectangle, loadTrainedModel, get_tube

from data import get_dataloader

from tensorboardX import SummaryWriter

from model import get_model

from tqdm import tqdm

np.random.seed(1024)

from examples.drone import TC_Simulate

beta = loadTrainedModel('log/checkpoint.pth.tar')
import ipdb; ipdb.set_trace()
initCond = np.array([1,1,1,0,0,0,0,0,np.pi])
initDelta = np.array([0.3, 0.3, 0.3, 0.01, 0.01, 0, 0, 0, np.pi])
tube = get_tube(initCond, initDelta, TC_Simulate, beta)

T_MAX = 10.0
sampled_traces = []
for _ in range(100):
    initial_point = (np.random.rand(len(set_lower)) * (set_higher - set_lower) + set_lower).tolist()
    trace = TC_Simulate(initial_point, T_MAX).tolist()
    sampled_traces.append(trace)
