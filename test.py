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
# set_lower = np.array([-3., -3., -3., -0.1, -0.1, 0., 0., 0., 0.])
# set_higher = np.array([3., 3., 3., 0.1, 0.1, 0., 0., 0., 2 * np.pi])
# initCond = np.random.rand(len(set_lower)) * (set_higher - set_lower) + set_lower
initCond = np.array([-1,-1,-2,0,0,0,0,0,np.pi])
initDelta = np.array([0.3, 0.3, 0.3, 0.01, 0.01, 0, 0, 0, np.pi])
ellipsoids, tube = get_tube(initCond, initDelta, TC_Simulate, beta)
# import ipdb; ipdb.set_trace()

set_lower = initCond - initDelta
set_higher = initCond + initDelta

def ellipsoid_surface(P):
    # Set of all spherical angles:
    K = 10
    u = np.linspace(0, 2 * np.pi, K)
    v = np.linspace(0, np.pi, K)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = np.outer(np.cos(u), np.sin(v)).reshape(-1)
    y = np.outer(np.sin(u), np.sin(v)).reshape(-1)
    z = np.outer(np.ones_like(u), np.cos(v)).reshape(-1)
    [x,y,z] = np.linalg.inv(P).dot(np.array([x,y,z]))
    return x.reshape(K,K), y.reshape(K,K), z.reshape(K,K)
    # return x, y, z

fig = plt.figure(figsize=(8.0, 5.0))
ax = fig.gca(projection='3d')

for reachset in ellipsoids:
    x,y,z = ellipsoid_surface(reachset[1])
    c = reachset[0]
    ax.plot_surface(x+c[0], y+c[1], z+c[2], color='g')

# randomly sample some traces
T_MAX = 10.0
for _ in range(100):
    initial_point = (np.random.rand(len(set_lower)) * (set_higher - set_lower) + set_lower).tolist()
    trace = TC_Simulate(initial_point, T_MAX).tolist()
    trace = np.array(trace)
    ax.plot(trace[:,1], trace[:,2], trace[:,3], color='b', label='samples')

# plt.title((benchmark_name+': initial set c=['+' '.join(['%.3f',] * 9)+'], r=%.3f; T_MAX=%.1f s')%tuple(center+[r, T_MAX]))
plt.show()
