# python plot_jetengine.py --config jetengine --pretrained_ours ~/Downloads/log_jetengine-x_0.3_1.3-y_0.3_1.3_ellipsoid/checkpoint.pth.tar --pretrained_spherical ~/Downloads/log_jetengine-x_0.3_1.3-y_0.3_1.3_circle/checkpoint.pth.tar --pretrained_dryvr log_jetEngine_dryvr/checkpoint.pth.tar
import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from tqdm import tqdm
from matplotlib import pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
HUGE_SIZE = 25

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

plt.subplots_adjust(
    top=0.909,
    bottom=0.132,
    left=0.133,
    right=0.893,
    hspace=0.2,
    wspace=0.2)

lambdas = np.array([0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729])
res = np.load('log/log_lambdas/res.npy', allow_pickle=True)
vol_mean = res[:,:,0].mean(axis=0)
vol_std = res[:,:,0].std(axis=0)
error_mean = (1-res[:,:,1]).mean(axis=0)
error_std = (1-res[:,:,1]).std(axis=0)

vol_min = res[:,:,0].min(axis=0)
vol_max = res[:,:,0].max(axis=0)
error_min = (1-res[:,:,1]).min(axis=0)
error_max = (1-res[:,:,1]).max(axis=0)

# import ipdb;ipdb.set_trace()
# vol = np.array([50.06970137, 40.49394116, 24.43284243, 20.65072908, 22.05023782, 15.73883482,
#  11.35559558])
# acc = np.array([0.998755, 0.999475, 0.985045, 0.965935, 0.989445, 0.911625, 0.68357])
# err = 1. - acc

fig, ax1 = plt.subplots()

ax1.set_xlabel(r'$\lambda$')
plt.xscale('log')
ax1.set_ylabel('Error')
# ax1.fill_between(lambdas, error_min, error_max, facecolor='b', alpha=0.5, label='Error')

# ax1.errorbar(lambdas, error_mean, error_std, capsize=5.0, fmt='-ob', label='Error')
ax1.errorbar(lambdas, error_mean, [error_mean - error_min, error_max - error_mean], capsize=5.0, fmt='-ob', label='Error')
# ax1.plot(lambdas, err, 'o-', color='g', label='Error')
# ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
ax2.set_ylabel('Volume')  # we already handled the x-label with ax1
# ax2.fill_between(lambdas, vol_min, vol_max, facecolor='g', alpha=0.5, label='volume')
# ax2.fill_between([], [], [], facecolor='b', alpha=0.5, label='Error')
# ax2.errorbar(lambdas, vol_mean, vol_std, capsize=5.0, fmt='-ok', label='Volume')
ax2.errorbar(lambdas, vol_mean, [vol_mean - vol_min, vol_max - vol_mean], capsize=5.0, fmt='-ok', label='Volume')
ax2.errorbar([], [], [], capsize=5.0, fmt='-ob', label='Error')

# ax2.errorbar([], [], [], capsize=5.0, fmt='-ob', label='Error')
# ax2.plot(lambdas, vol, 'o-', color='b', label='Volume')
# ax2.plot([], [], 'o-', color='g', label='Error')

# ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title(r'Impact of $\lambda$')
plt.legend()
plt.show()
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.show()
