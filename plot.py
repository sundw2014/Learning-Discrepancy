import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from tqdm import tqdm
from mayavi import mlab

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
parser.add_argument('--pretrained', type=str)

args = parser.parse_args()

config = importlib.import_module('config_'+args.config)
model, forward = get_model(config.num_dim_input, config.num_dim_output)
model = torch.nn.DataParallel(model)
if args.use_cuda:
    model = model.cuda()
torch.backends.cudnn.benchmark = True

model.load_state_dict(torch.load(args.pretrained)['state_dict'])

def ellipsoid_surface_3D(P):
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

def ellipsoid_surface_2D(P):
    K = 100
    thetas = np.linspace(0, 2 * np.pi, K)
    points = []
    for i, theta in enumerate(thetas):
        point = np.array([np.cos(theta), np.sin(theta)])
        points.append(point)
    points = np.array(points)
    points = np.linalg.inv(P).dot(points.T)
    return points[0,:], points[1,:]

simulate = config.simulate
benchmark_name = args.config
T_MAX = 10.0


normalized_center = np.random.rand(len(config.normalized_Theta[:,0])) * (config.normalized_Theta[:,1] - config.normalized_Theta[:,0]) + config.normalized_Theta[:,0]

normalized_r = np.random.rand() * config.normalized_X0_RMAX

traces = []
# ref trace
trace = simulate(config.unnormalize(normalized_center).tolist(), T_MAX)
traces.append(np.array(trace))
# calculate the reachset using the trained model
reachsets = []
# [trace[0, 1:], np.eye(config.num_dim_output)/normalized_r], ]
for i in tqdm(range(trace.shape[0])):
    P = forward(torch.tensor(config.observe_for_input(trace[0,1:]).tolist() + [normalized_r, trace[i,0]]).view(1,-1).cuda().float())
    P = P.view(config.num_dim_output,config.num_dim_output)
    reachsets.append([trace[i, 1:], P.cpu().detach().numpy()])

# from tvtk.api import tvtk
# mlab.pipeline.user_defined(data, filter=tvtk.CubeAxesActor())
mlab.figure(1, size=(400, 400), bgcolor=(0, 0, 0))
mlab.clf()

# plot the ref trace
trace = np.array(trace)
# ax.plot(trace[:,1], trace[:,2], trace[:,3], color='r', label='ref')
if config.num_dim_output == 2:
    tmp = config.observe_for_output(trace[:,1:])
    mlab.plot3d(tmp[:,0], tmp[:,1], np.zeros_like(tmp[:,0]), color=(1,0,0))
elif config.num_dim_output == 3:
    tmp = config.observe_for_output(trace[:,1:])
    mlab.plot3d(tmp[:,0], tmp[:,1], tmp[:,2], color=(1,0,0))

# plot ellipsoids for each time step
for reachset in reachsets:
    c = config.observe_for_output(reachset[0])
    if config.num_dim_output == 2:
        x,y = ellipsoid_surface_2D(reachset[1])
        mlab.plot3d(x+c[0], y+c[1], np.zeros_like(x+c[0]), color=(0,1,0))
    elif config.num_dim_output == 3:
        x,y,z = ellipsoid_surface_3D(reachset[1])
        mlab.mesh(x+c[0], y+c[1], z+c[2], color=(0,1,0), opacity=0.2)

# randomly sample some traces
for _ in range(100):
    d = np.random.randn(len(config.nonzero_dims))
    d = d / np.sqrt((d**2).sum())

    while True:
        # import ipdb; ipdb.set_trace()
        samples = normalized_center[config.nonzero_dims].reshape(1,-1) + normalized_r * np.random.rand(1000,1) * d.reshape(1,-1)
        tmp = np.tile(config.normalized_Theta[:,0].reshape(1,-1), (1000, 1))
        tmp[:, config.nonzero_dims] = samples

        hit = np.where(np.logical_and(\
         (samples>=config.normalized_Theta[:,0].reshape(1,-1)).sum(axis=1) == len(normalized_center),
         (samples<=config.normalized_Theta[:,1].reshape(1,-1)).sum(axis=1) == len(normalized_center)))[0]
        if len(hit)>0:
            break


    # import ipdb; ipdb.set_trace()
    point = config.unnormalize(samples[hit[0],:])
    _trace = config.observe_for_output(simulate(point.tolist(), T_MAX)[:,1:])
    if config.num_dim_output == 2:
        mlab.plot3d(_trace[:,0], _trace[:,1], np.zeros_like(_trace[:,0]), color=(0,0,1))
    elif config.num_dim_output == 3:
        mlab.plot3d(_trace[:,0], _trace[:,1], _trace[:,2], color=(0,0,1))

mlab.show()
