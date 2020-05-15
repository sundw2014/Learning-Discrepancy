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
from utils import AverageMeter

from data import get_dataloader

from tensorboardX import SummaryWriter

from model import get_model, num_dim_projected

from tqdm import tqdm

#log_dir = 'log_vanderpol-x_0_2-y_0_3_circle'
log_dir = 'log'

np.random.seed(1024)

num_dim = 9
model, forward = get_model(num_dim)

def load_checkpoint_residuals(filename='checkpoint.pth.tar'):
    filename = log_dir + '/' + filename
    checkpoint = torch.load(filename)
    print('checkpoint epoch: %d'%checkpoint['epoch'])
    return checkpoint['state_dict'], checkpoint['residuals']

def load_checkpoint(filename='checkpoint.pth.tar'):
    filename = log_dir + '/' + filename
    checkpoint = torch.load(filename)
    print('checkpoint epoch: %d'%checkpoint['epoch'])
    return checkpoint['state_dict']

def calc_half_conformal_interval_width(residuals, miscoverage_rate):
    d = residuals[int(np.ceil((len(residuals)+1) * (1-miscoverage_rate)))-1]
    return d

def ellipsoid(P):
    # thetas = np.arange(0,1,0.01) * 2 * np.pi
    # points = []
    # for i, theta in enumerate(thetas):
    #     point = np.array([np.cos(theta), np.sin(theta)])
    #     points.append(point)
    # points = np.array(points)
    K = 1000
    points = np.random.randn(K, P.shape[0])
    points = points / np.sqrt((points ** 2).sum(axis=1, keepdims=True))
    points = np.linalg.inv(P).dot(points.T)
    return points

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

model = torch.nn.DataParallel(model).cuda()
state_dict = load_checkpoint()
model.load_state_dict(state_dict)

torch.backends.cudnn.benchmark = True

# plot
# from examples.vanderpol import TC_Simulate
# benchmark_name = 'vanderpol'
# T_MAX = 10.0
#
# centers = [[0.15, 0.15], [1.4, 2.3]]
# rs = [0.075, 0.15, 0.3]

# from examples.jet_engine import TC_Simulate
# benchmark_name = 'jet_engine'
# T_MAX = 10.0
#
# centers = [[0.8, 0.8], [0.4, 0.8], [0.4, 1.2]]
# rs = [0.075, 0.15, 0.3]

# from examples.Brusselator import TC_Simulate
# benchmark_name = 'Brusselator'
# T_MAX = 10.0
#
# centers = [[0.9, 0.15], [0.4, 0.15], [1.4, 0.15], [0.4, 0.3], [1.4, 0.3], [0.9, 0.3]]
# rs = [0.075, 0.15, 0.3]

#
# center = [0.15, 0.15]
# r = 0.15
#
# center = [0.15, 0.15]
# r = 0.075
#
# center = [0.15, 0.15]
# r = 0.075

# center = [1.4, 2.3] # vandepol on limit circle
# r = 0.3

from examples.drone import TC_Simulate
benchmark_name = 'drone'
T_MAX = 10.0

set_lower = np.array([1-0.5, 1-0.5, 4.5, -0.1, -0.1, -0.01, -0.01, -0.01, 0.0])
set_higher = np.array([1+0.5, 1+0.5, 5.5, 0.1, 0.1, 0.01, 0.01, 0.01, 2 * np.pi - 1e-6])

centers = []
np.random.seed(1024)
# for _ in range(3):
for _ in range(1):
    # import ipdb; ipdb.set_trace()
    centers.append((np.random.rand(len(set_lower)) * (set_higher - set_lower) + set_lower).tolist())

rs = [0.125/2, 0.125/4, 0.125/8, 1e-5]

cnt = 0

for center in centers:
    for r in rs:
        traces = []
        # ref trace
        trace = TC_Simulate(center, T_MAX).tolist()[::10]
        traces.append(np.array(trace))
        # calculate the reachset using the trained model
        reachsets = [[center, np.eye(num_dim_projected)/r], ]
        for point in tqdm(trace[1::]):
            P = forward(torch.tensor(center + point[1::] +[r, point[0]]).view(1,-1).cuda())
            P = P.view(num_dim_projected,num_dim_projected)
            reachsets.append([point[1::], P.cpu().detach().numpy()])

        fig = plt.figure(figsize=(8.0, 5.0))
        ax = fig.gca(projection='3d')

        trace = np.array(trace)
        ax.plot(trace[:,1], trace[:,2], trace[:,3], color='r', label='ref')

        for reachset in reachsets:
            x,y,z = ellipsoid_surface(reachset[1])
            # for i in range(ellipsoid_points.shape[1]):
            # center = np.tile(np.array(reachset[0]).reshape(-1,1), [1, ellipsoid_points.shape[1]])
            # ax.quiver(center[0,:], center[1,:], center[2,:], ellipsoid_points[0,:], ellipsoid_points[1,:], ellipsoid_points[2,:], color='g')
            c = reachset[0]
            ax.plot_surface(x+c[0], y+c[1], z+c[2], color='g')
        # plot the ref trace
        # for reachset in reachsets:
        #     plt.plot(reachset[0][0], reachset[0][1], 'o', markersize=1.0, color='r', label='ref trace' if reachset is reachsets[0] else None)

        for _ in range(100):
            while True:
                samples = r * (np.random.rand(num_dim, 10) - 0.5)
                hit = np.where(np.sqrt((samples ** 2).sum(axis=0)) < r)[0]
                if len(hit)>0:
                    break
            # import ipdb; ipdb.set_trace()
            c = samples[:,hit[0]] + center
            trace = TC_Simulate(c, T_MAX).tolist()[::5]
            trace = np.array(trace)
            ax.plot(trace[:,1], trace[:,2], trace[:,3], color='b', label='samples')

        #plt.show()

        # for i, dist_ratio in enumerate(np.array(range(1, 11))/10):
        #     dist = r * dist_ratio
        #     theta = np.random.rand() * 2 * np.pi
        #     point = dist * np.array([np.cos(theta), np.sin(theta)])
        #     point = np.array(center) + point
        #     _trace = np.array(TC_Simulate('Default', point.tolist(), T_MAX*0.8))
        #     plt.plot(_trace[:, 1], _trace[:, 2], 'o', markersize=1.0, color='b', label='sampled traces' if i==0 else None)
        # plot the actual reachset. Assumption: boundary is maintained
        # thetas = np.arange(0,1,0.01) * 2 * np.pi
        # for i, theta in enumerate(thetas):
        #     point = r * np.array([np.cos(theta), np.sin(theta)])
        #     point = np.array(center) + point
        #     traces.append(np.array(TC_Simulate('Default', point.tolist(), T_MAX)))
        #     # plt.plot(traces[-1][:, 1], traces[-1][:, 2], 'o', markersize=1.0, color=colors[i])
        #
        #     # point = dist * 2 * np.array([np.cos(theta), np.sin(theta)])
        #     # point = np.array(c) + point
        #     # traces.append(np.array(TC_Simulate('Default', point.tolist(), T_MAX)))
        #     # plt.plot(traces[-1][:, 1], traces[-1][:, 2], 'o', markersize=1.0, color=colors[i])
        #
        #     for t in range(traces[0].shape[0]):
        #         plt.arrow(traces[0][t, 1], traces[0][t, 2], traces[-1][t, 1] - traces[0][t, 1], traces[-1][t, 2] - traces[0][t, 2], color='b', alpha=0.3)
        #
        # # initial set
        # reachset = reachsets[0]
        # circ = plt.Circle(reachset[0], radius=r, color='k', label='intial set')
        # ax.add_patch(circ)
        #
        # plt.legend()
        plt.title((benchmark_name+': initial set c=['+' '.join(['%.3f',] * 9)+'], r=%.3f; T_MAX=%.1f s')%tuple(center+[r, T_MAX]))
        # # plt.show()
        plt.savefig(log_dir + '/' + benchmark_name + '_%d'%cnt + '.pdf')
        cnt += 1
