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

model = torch.nn.DataParallel(model).cuda()
state_dict = load_checkpoint()
model.load_state_dict(state_dict)

torch.backends.cudnn.benchmark = True

# plot
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

rs = [0.125/2, 0.125/4, 0.125/8, 0.125/16, 0.125/32]

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

        # from mayavi import mlab
        # from tvtk.api import tvtk
        # mlab.pipeline.user_defined(data, filter=tvtk.CubeAxesActor())
        # mlab.figure(1, size=(400, 400), bgcolor=(0, 0, 0))
        # mlab.clf()
        fig = plt.figure(figsize=(8.0, 5.0))
        ax = fig.gca(projection='3d')

        # plot the ref trace
        trace = np.array(trace)
        ax.plot(trace[:,1], trace[:,2], trace[:,3], color='r', label='ref')
        # mlab.plot3d(trace[:,1], trace[:,2], trace[:,3], color=(1,0,0))

        # plot ellipsoids for each time step
        # reachtube = [[], [], []]
        for reachset in reachsets:
            x,y,z = ellipsoid_surface(reachset[1])
            c = reachset[0]
            ax.plot_surface(x+c[0], y+c[1], z+c[2], color='g')
            # mlab.mesh(x+c[0], y+c[1], z+c[2], color=(0,1,0), opacity=0.2)
            # reachtube[0].append(x+c[0])
            # reachtube[1].append(y+c[1])
            # reachtube[2].append(z+c[2])

        # for i in range(3):
        #     reachtube[i] = np.concatenate(reachtube[i]).reshape(-1,1)
        # reachtube = np.concatenate(reachtube, axis=1)
        # hull = ConvexHull(reachtube)
        # # import ipdb; ipdb.set_trace()
        # reachtube = reachtube[hull.vertices, :]
        # ax.plot_trisurf(reachtube[:, 0], reachtube[:, 1], reachtube[:, 2], color='g')

        # randomly sample some traces
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
            # mlab.plot3d(trace[:,1], trace[:,2], trace[:,3], color=(0,0,1))

        # mlab.show()
        plt.title((benchmark_name+': initial set c=['+' '.join(['%.3f',] * 9)+'], r=%.3f; T_MAX=%.1f s')%tuple(center+[r, T_MAX]))
        # plt.savefig(log_dir + '/' + benchmark_name + '_%d'%cnt + '.pdf')
        plt.show()
        # cnt += 1
