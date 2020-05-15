# from dreal import *
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from utils import AverageMeter

from data import get_dataloader

from tensorboardX import SummaryWriter

np.random.seed(1024)

# plot
from examples.vanderpol import TC_Simulate
benchmark_name = 'vanderpol'
T_MAX = 20.0

centers = [[0.15, 0.15], [1.4, 2.3]]
rs = [0.075, 0.15, 0.3]

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
c = centers[0]
dist = 0.15
thetas = np.arange(0,1,0.01) * 2 * np.pi

cmap = plt.get_cmap('gnuplot')
# colors = [cmap(i) for i in np.linspace(0, 1, len(thetas))]
colors = [cmap(0) for i in np.linspace(0, 1, len(thetas))]



time_step = 30
traces = [np.array(TC_Simulate('Default', c, T_MAX))[::time_step],]
plt.plot(traces[0][:, 1], traces[0][:, 2], 'o', markersize=1.0, color='r', label='ref')
for i, theta in enumerate(thetas):
    point = dist * np.array([np.cos(theta), np.sin(theta)])
    point = np.array(c) + point
    traces.append(np.array(TC_Simulate('Default', point.tolist(), T_MAX))[::time_step])
    plt.plot(traces[-1][:, 1], traces[-1][:, 2], 'o', markersize=1.0, color=colors[i])

    # point = dist * 2 * np.array([np.cos(theta), np.sin(theta)])
    # point = np.array(c) + point
    # traces.append(np.array(TC_Simulate('Default', point.tolist(), T_MAX)))
    # plt.plot(traces[-1][:, 1], traces[-1][:, 2], 'o', markersize=1.0, color=colors[i])

    for t in range(traces[0].shape[0]):
        plt.arrow(traces[0][t, 1], traces[0][t, 2], traces[-1][t, 1] - traces[0][t, 1], traces[-1][t, 2] - traces[0][t, 2], color=colors[i])

plt.show()
