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
parser.add_argument('--id', type=int, default=0)

args = parser.parse_args()
np.random.seed(args.seed)

config = importlib.import_module('config_'+args.config)
use_dryvr = 'dryvr' in args.pretrained
if use_dryvr:
    model, forward = get_model_dryvr(len(config.sample_X0())+1, config.simulate(config.get_init_center(config.sample_X0())).shape[1]-1)
else:
    model, forward = get_model(len(config.sample_X0())+1, config.simulate(config.get_init_center(config.sample_X0())).shape[1]-1, config)
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
    # import ipdb; ipdb.set_trace()
    ref = np.array(ref) # T x n
    trj = np.array(sampled_traces)[:,:,1:].transpose([1,2,0]) # T x n x N
    trj = trj - np.expand_dims(ref[:,1:], -1)
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

query = []
c_base = np.array([590, -0.1, 0, -np.pi/4, -0.1, 69])
inc = np.array([0, 0.025, np.pi/32, np.pi/16, 0.025, 0])
r = np.array([10, 0.025, np.pi/32, np.pi/16, 0.025, 1])

for i1 in range(4):
    for i2 in range(4):
        for i3 in range(4):
            for i4 in range(4):
                c = c_base + (inc * np.array([0,2*i1+1,2*i2+1,2*i3+1,2*i4+1,0]))
                query.append([c,r])

reachsets = []
for c,r in tqdm(query):
    X0 = np.array(c.tolist()+r.tolist())
    print(X0)
    ref = config.simulate(config.get_init_center(X0))
    X0_mean, X0_std = config.get_X0_normalization_factor()
    X0 = (X0 - X0_mean) / X0_std
    for idx_t in range(1, ref.shape[0]):
        s = time.time()
        P = forward(torch.tensor(X0.tolist()+[ref[idx_t, 0],]).view(1,-1).cuda().float())
        e = time.time()
        # times[0].append(e-s)
        P = P.squeeze(0)
        reachsets.append([ref[idx_t, 1:], P.cpu().detach().numpy()])

# plot ellipsoids for each time step
for reachset in tqdm(reachsets):
    c = reachset[0]
    if ref.shape[1]-1 == 2:
        x,y = ellipsoid_surface_2D(reachset[1])
        plt.plot(x+c[0], y+c[1], color=(0,1,0))

plt.plot(ref[:,1], ref[:,2], color=(1,0,0))

c = np.array([590, 0., np.pi/8, 0, 0., 69])
r = np.array([10, 0.1, np.pi/8, np.pi/4, 0.1, 1])
X0 = np.array(c.tolist()+r.tolist())

sampled_inits = [config.sample_x0(X0) for _ in range(1000)]
num_proc = min([1, multiprocessing.cpu_count()-3])
with Pool(num_proc, initializer=mute) as p:
    sampled_trajs = list(tqdm(p.imap(config.simulate, sampled_inits), total=len(sampled_inits)))

for sampled_traj in sampled_trajs:
    plt.plot(sampled_traj[:,1], sampled_traj[:,2], color=(0,0,1), alpha=0.1)
# plt.savefig('data/images/%d.png'%(args.id))
plt.show()
