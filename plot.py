import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from tqdm import tqdm
from mayavi import mlab

from model import get_model as get_model_ours
from model_spherical import get_model as get_model_spherical
from model_dryvr import get_model as get_model_dryvr

import sys
sys.path.append('configs')

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--config', type=str,
                        default='drone')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false')
parser.set_defaults(use_cuda=True)
parser.add_argument('--pretrained_ours', type=str)
parser.add_argument('--pretrained_spherical', type=str)
parser.add_argument('--pretrained_dryvr', type=str)
parser.add_argument('--no_plot', type=str)
parser.add_argument('--seed', type=int, default=1024)

args = parser.parse_args()
np.random.seed(args.seed)

config = importlib.import_module('config_'+args.config)
model_ours, forward_ours = get_model_ours(config.num_dim_input, config.num_dim_output)
model_ours = torch.nn.DataParallel(model_ours)
if args.use_cuda:
    model_ours = model_ours.cuda()
torch.backends.cudnn.benchmark = True
model_ours.load_state_dict(torch.load(args.pretrained_ours)['state_dict'])

model_spherical, forward_spherical = get_model_spherical(config.num_dim_input, config.num_dim_output)
model_spherical = torch.nn.DataParallel(model_spherical)
if args.use_cuda:
    model_spherical = model_spherical.cuda()
torch.backends.cudnn.benchmark = True
model_spherical.load_state_dict(torch.load(args.pretrained_spherical)['state_dict'])

model_dryvr, forward_dryvr = get_model_dryvr(config.num_dim_input, config.num_dim_output)
model_dryvr.load_state_dict(torch.load(args.pretrained_dryvr)['state_dict'])

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

simulate = config.simulate
benchmark_name = args.config
T_MAX = 10.0
if hasattr(config, 'T_MAX'):
    T_MAX = config.T_MAX

normalized_center = np.random.rand(len(config.normalized_Theta[:,0])) * (config.normalized_Theta[:,1] - config.normalized_Theta[:,0]) + config.normalized_Theta[:,0]

normalized_r = np.random.rand() * config.normalized_X0_RMAX
# normalized_r = np.random.rand() * 0.1
# normalized_r = 0.1
print(normalized_center, normalized_r)

traces = []
# ref trace
trace = simulate(config.unnormalize(normalized_center).tolist(), T_MAX)
traces.append(np.array(trace))
# calculate the reachset using the trained model
reachsets_ours = []
reachsets_spherical = []
reachsets_dryvr = []
times = [[],[],[]]
# [trace[0, 1:], np.eye(config.num_dim_output)/normalized_r], ]
for i in tqdm(range(1, trace.shape[0])):
    s = time.time()
    P = forward_ours(torch.tensor(config.observe_for_input(trace[0,1:]).tolist() + config.observe_for_input(trace[i,1:]).tolist() + [normalized_r, trace[i,0]]).view(1,-1).cuda().float())
    e = time.time()
    times[0].append(e-s)
    P = P.view(config.num_dim_output,config.num_dim_output)
    reachsets_ours.append([trace[i, 1:], P.cpu().detach().numpy()])

    s = time.time()
    P = forward_spherical(torch.tensor(config.observe_for_input(trace[0,1:]).tolist() + config.observe_for_input(trace[i,1:]).tolist() + [normalized_r, trace[i,0]]).view(1,-1).cuda().float())
    e = time.time()
    times[1].append(e-s)

    P = P.view(config.num_dim_output,config.num_dim_output)
    reachsets_spherical.append([trace[i, 1:], P.cpu().detach().numpy()])

    s = time.time()
    P = forward_dryvr(torch.tensor(config.observe_for_input(trace[0,1:]).tolist() + config.observe_for_input(trace[i,1:]).tolist() + [normalized_r, trace[i,0]]).view(1,-1).cuda().float())
    e = time.time()
    times[2].append(e-s)

    reachsets_dryvr.append([trace[i, 1:], P])

# from IPython import embed; embed()
# tmp = []
# for i in range(len(reachsets)):
#     P = reachsets[i][1]
#     w, v = np.linalg.eig(P)
#     m = np.abs(w).min()
#     print(m)
#     if m > 1.:
#         tmp.append(reachsets[i])
#reachsets = tmp

print('time: %.lf, %.lf, %.lf'%(np.mean(times[0]), np.mean(times[1]), np.mean(times[2])))
# from tvtk.api import tvtk
# mlab.pipeline.user_defined(data, filter=tvtk.CubeAxesActor())
mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(0,0,0))
mlab.clf()

# plot the ref trace
trace = np.array(trace)
# ax.plot(trace[:,1], trace[:,2], trace[:,3], color='r', label='ref')
if config.num_dim_output == 2:
    tmp = config.observe_for_output(trace[:,1:])
    mlab.plot3d(tmp[:,0], tmp[:,1], np.zeros_like(tmp[:,0]), color=(1,0,0))
elif config.num_dim_output == 3:
    tmp = config.observe_for_output(trace[:,1:])
    # mlab.plot3d(tmp[:,0], tmp[:,1], tmp[:,2], color=(1,1,1))

# mlab.outline(s, color=(.7, .7, .7), extent=(0 ,1 , 0 ,1 , 0 ,1))

vol_ours = calc_volume([r[1] for r in reachsets_ours])#[10:]])
vol_spherical = calc_volume([r[1] for r in reachsets_spherical])#[10:]])
vol_dryvr = calc_volume([r[1] for r in reachsets_dryvr])#[10:]])
print(vol_ours, vol_dryvr, vol_spherical)

if args.no_plot is not None:
    exit()

# plot ellipsoids for each time step
for reachset_ours, reachset_dryvr in zip(reachsets_ours, reachsets_dryvr):
    c = config.observe_for_output(reachset_ours[0])
    if config.num_dim_output == 2:
        x,y = ellipsoid_surface_2D(reachset_ours[1])
        mlab.plot3d(x+c[0], y+c[1], np.zeros_like(x+c[0]), color=(0,1,0))
        x,y = ellipsoid_surface_2D(reachset_dryvr[1])
        s=mlab.plot3d(x+c[0], y+c[1], np.zeros_like(x+c[0]), color=(1,1,0))
    elif config.num_dim_output == 3:
        x,y,z = ellipsoid_surface_3D(reachset_ours[1])
        mlab.mesh(x+c[0], y+c[1], z+c[2], color=(0,1,0), opacity=0.9)
        x,y,z = ellipsoid_surface_3D(reachset_dryvr[1])
        s=mlab.mesh(x+c[0]+2., y+c[1], z+c[2], color=(1,1,0), opacity=0.9)

mlab.axes(s, color=(.7, .7, .7), extent=(-1, 2, -2, 2, 0, 2), z_axis_visibility=False)
mlab.show()
exit()
sampled_traces = []

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
    sampled_traces.append(_trace)
    # if config.num_dim_output == 2:
    #     mlab.plot3d(_trace[:,0], _trace[:,1], np.zeros_like(_trace[:,0]), color=(0,0,1), line_width=0.05)
    # elif config.num_dim_output == 3:
    #     mlab.plot3d(_trace[:,0], _trace[:,1], _trace[:,2], color=(0,0,1), line_width=0.05)
# print('over')
# import ipdb;ipdb.set_trace()
# from IPython import embed;embed()
ref = config.observe_for_output(trace[1:,1:])
acc_ours = calc_acc(sampled_traces, [r[1] for r in reachsets_ours], ref)
acc_spherical = calc_acc(sampled_traces, [r[1] for r in reachsets_spherical], ref)
acc_dryvr = calc_acc(sampled_traces, [r[1] for r in reachsets_dryvr], ref)
print(acc_ours, acc_dryvr, acc_spherical)

mlab.show()
