# from dreal import *
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from utils import samplePointsOnAARectangle

from data import get_dataloader

from tensorboardX import SummaryWriter

np.random.seed(1024)

# plot
# from examples.vanderpol import TC_Simulate
# benchmark_name = 'vanderpol'
# T_MAX = 20.0
#
# centers = [[0.15, 0.15], [1.4, 2.3]]
# rs = [0.075, 0.15, 0.3]

# from examples.jet_engine import TC_Simulate
# benchmark_name = 'jet_engine'
# T_MAX = 20.0
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
# c = centers[0]
# dist = 0.15
# thetas = np.arange(0,1,0.01) * 2 * np.pi

# cmap = plt.get_cmap('gnuplot')
# colors = [cmap(i) for i in np.linspace(0, 1, len(thetas))]
# colors = [cmap(0) for i in np.linspace(0, 1, len(thetas))]



# time_step = 30
# traces = [np.array(TC_Simulate('Default', c, T_MAX))[::time_step],]
# plt.plot(traces[0][:, 1], traces[0][:, 2], 'o', markersize=1.0, color='r', label='ref')
# for i, theta in enumerate(thetas):
#     point = dist * np.array([np.cos(theta), np.sin(theta)])
#     point = np.array(c) + point
#     traces.append(np.array(TC_Simulate('Default', point.tolist(), T_MAX))[::time_step])
#     plt.plot(traces[-1][:, 1], traces[-1][:, 2], 'o', markersize=1.0, color=colors[i])
#
#     # point = dist * 2 * np.array([np.cos(theta), np.sin(theta)])
#     # point = np.array(c) + point
#     # traces.append(np.array(TC_Simulate('Default', point.tolist(), T_MAX)))
#     # plt.plot(traces[-1][:, 1], traces[-1][:, 2], 'o', markersize=1.0, color=colors[i])
#
#     for t in range(traces[0].shape[0]):
#         plt.arrow(traces[0][t, 1], traces[0][t, 2], traces[-1][t, 1] - traces[0][t, 1], traces[-1][t, 2] - traces[0][t, 2], color=colors[i])
#
# plt.show()

from examples.drone import TC_Simulate

set_lower = np.array([1-0.5, 1-0.5, 4.5, -0.1, -0.1, -0., -0., -0., 0.0])
set_higher = np.array([1+0.5, 1+0.5, 5.5, 0.1, 0.1, 0., 0., 0., 2 * np.pi - 1e-6])

# goal = [2,2,6]
# import ipdb; ipdb.set_trace()
initial_states = samplePointsOnAARectangle(np.array([set_lower, set_higher]).T.reshape(-1), K=10)
traces = []
for i in range(initial_states.shape[0]):
    initial_state = initial_states[i,:]
    trace = TC_Simulate(initial_state, 10.)#, goal)
    traces.append(trace)

traces = np.array(traces)

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

plt.ion()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim3d([0, 3])
ax.set_ylim3d([0, 3])
ax.set_zlim3d([4, 7])
ax.scatter(initial_states[:,0], initial_states[:,1], initial_states[:,2], marker='o', color='k')#, s=12)

current = ax.plot(initial_states[:,0], initial_states[:,1], initial_states[:,2], 'o', color='b')[0]

for t in range(traces[0].shape[0]):
    current.set_data(traces[:,t,1], traces[:,t,2])
    current.set_3d_properties(traces[:,t,3])
    fig.canvas.draw()
    fig.canvas.flush_events()
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # import ipdb; ipdb.set_trace()


    # ax.legend()
    # plt.show()
    # dimensions = len(trace[0])
    # init_delta_array = [0.5,0.5,0.5] + [0.1] * (dimensions - 4)
    # k = [1] * (dimensions - 1)
    # gamma = [0] * (dimensions - 1)
    # tube = bloatToTube(k, gamma, init_delta_array, trace, dimensions)
    # gazebotube = tube[:][1:4]
    # gazebotrace = trace[:][1:4]
    # print(tube)
    # plt.plot(trace[:,1], trace[:,3])
    # safety, reach = _verify_reach_tube(np.zeros((9,)), "[2; 2; 5]", 2.5, [])
    #print("reach: ", reach.tube)
    # plt.show()
