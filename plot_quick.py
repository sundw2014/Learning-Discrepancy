import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from tqdm import tqdm
# from mayavi import mlab
from matplotlib import pyplot as plt

from model import get_model

import sys
sys.path.append('configs')

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
model, forward = get_model(len(config.sample_D0_from_P())+1, config.sample_ref(config.sample_D0_from_P()).shape[1]-1)
if args.use_cuda:
    model = model.cuda()
torch.backends.cudnn.benchmark = True
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(args.pretrained)['state_dict'])

def calc_volume(Ps):
    vol = 0.
    for P in Ps:
        tmp = np.sqrt(1 / np.linalg.det(P.T.dot(P)))
        if P.shape[0] == 3:
            tmp *= np.pi * 4 / 3
        elif P.shape[0] == 2:
            tmp *= np.pi
        else:
            raise ValueError('wrong shape')
        vol += tmp
    return vol

def calc_acc(sampled_traces, Ps, ref):
    ref = np.array(ref) # T x n
    trj = np.array(sampled_traces)[:,1:,:].transpose([1,2,0]) # T x n x N
    trj = trj - np.expand_dims(ref, -1)
    Ps = np.array(Ps) # T x n x n
    Px = np.matmul(Ps,trj) # T x n x N
    Pxn = (Px**2).sum(axis=1).reshape(-1)
    return (Pxn<=1).sum()/len(Pxn)

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
    # if np.array([x,y,z]).max()>10:
        # import ipdb;ipdb.set_trace()
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

D0 = config.sample_D0_from_P()
ref = config.sample_ref(D0)
sampled_trajs = [config.sample_traj(config.sample_from_D0(D0)) for _ in range(100)]
benchmark_name = args.config

reachsets = []

D0_mean, D0_std = config.get_D0_normalization_factor()
D0 = (D0 - D0_mean) / D0_std

for idx_t in tqdm(range(1, ref.shape[0])):
    s = time.time()
    P = forward(torch.tensor(D0.tolist()+[ref[idx_t, 0],]).view(1,-1).cuda().float())
    e = time.time()
    # times[0].append(e-s)
    P = P.squeeze(0)
    reachsets.append([ref[idx_t, 1:], P.cpu().detach().numpy()])

# from tvtk.api import tvtk
# mlab.pipeline.user_defined(data, filter=tvtk.CubeAxesActor())
# mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(0,0,0))
# mlab.clf()

# import ipdb;ipdb.set_trace()
# plot the ref trace
# ax.plot(trace[:,1], trace[:,2], trace[:,3], color='r', label='ref')
if ref.shape[1]-1 == 2:
    # mlab.plot3d(ref[:,1], ref[:,2], np.zeros_like(ref[:,0]), color=(1,0,0))
    plt.plot(ref[:,1], ref[:,2], color=(1,0,0))
elif ref.shape[1]-1 == 3:
    mlab.plot3d(ref[:,1], ref[:,2], ref[:,3], color=(1,1,1))

# mlab.outline(s, color=(.7, .7, .7), extent=(0 ,1 , 0 ,1 , 0 ,1))

vol = calc_volume([r[1] for r in reachsets])
print('volume:', vol)

# plot ellipsoids for each time step
for reachset in reachsets:
    c = reachset[0]
    if ref.shape[1]-1 == 2:
        x,y = ellipsoid_surface_2D(reachset[1])
        # mlab.plot3d(x+c[0], y+c[1], np.zeros_like(x+c[0]), color=(0,1,0))
        plt.plot(x+c[0], y+c[1], color=(0,1,0))
    elif ref.shape[1]-1 == 3:
        x,y,z = ellipsoid_surface_3D(reachset[1])
        mlab.mesh(x+c[0], y+c[1], z+c[2], color=(0,1,0), opacity=0.9)

# mlab.axes(s, color=(.7, .7, .7), extent=(-1, 2, -2, 2, 0, 2), z_axis_visibility=False)
# mlab.show()
# exit()

for sampled_traj in sampled_trajs:
    if ref.shape[1]-1 == 2:
        # mlab.plot3d(sampled_traj[:,1], sampled_traj[:,2], np.zeros_like(sampled_traj[:,1]), color=(0,0,1), line_width=0.05)
        plt.plot(sampled_traj[:,1], sampled_traj[:,2], color=(0,0,1), alpha=0.1)
    elif ref.shape[1]-1 == 3:
        mlab.plot3d(sampled_traj[:,1], sampled_traj[:,2], sampled_traj[:,3], color=(0,0,1), line_width=0.05)
# print('over')
# import ipdb;ipdb.set_trace()
# from IPython import embed;embed()
# ref = config.observe_for_output(trace[1:,1:])
# acc_ours = calc_acc(sampled_traces, [r[1] for r in reachsets_ours], ref)
# acc_spherical = calc_acc(sampled_traces, [r[1] for r in reachsets_spherical], ref)
# acc_dryvr = calc_acc(sampled_traces, [r[1] for r in reachsets_dryvr], ref)
# print(acc_ours, acc_dryvr, acc_spherical)

# mlab.show()
# plt.plot([0,1], [1,0])
plt.show()
