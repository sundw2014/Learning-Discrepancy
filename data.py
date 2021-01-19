from __future__ import print_function
from multiprocessing import Pool
from functools import partial
import tqdm
import os
import os.path
import errno
import numpy as np
import sys
import torch

import torch.utils.data as data
from utils import loadpklz, savepklz

def gen_trace_in_a_ball(num_traces, R_MAX, normalized_Theta, nonzero_dims, unnormalize, simulate, T_MAX, c):
    traces = []
    for id_trace in range(num_traces+1):
        # sample a direction
        d = np.random.randn(len(nonzero_dims))
        d = d / np.sqrt((d**2).sum())

        if np.random.rand() > 0.5:
            r = R_MAX
        else:
            r = R_MAX * np.random.rand()
        if id_trace == 0:
            r = 0.
        sample = c[nonzero_dims].reshape(-1) + r * d.reshape(-1)
        point = normalized_Theta[:,0].reshape(-1).copy()
        point[nonzero_dims] = sample
        point = unnormalize(point)
        _trace = simulate(point.tolist(), T_MAX)
        traces.append(_trace)
    return np.array(traces)

class DiscriData(data.Dataset):
    """DiscriData."""
    def __init__(self, config, num_traces, num_sampling_balls=100, T_MAX=10.0, data_file=None, shuffle=True):
        super(DiscriData, self).__init__()

        if hasattr(config, 'T_MAX'):
            T_MAX = config.T_MAX
        self.config = config

        def sample_X0_center():
            num_dim_state = self.config.num_dim_state
            normalized_Theta = self.config.normalized_Theta
            sample = np.random.rand(num_dim_state) * (normalized_Theta[:,1] - normalized_Theta[:,0]) + normalized_Theta[:,0]
            return sample

        # generate traces
        self.num_sampling_balls = num_sampling_balls
        self.X0_centers = [sample_X0_center() for _ in range(self.num_sampling_balls)]
        self.X0_rs = [self.config.normalized_X0_RMAX*np.random.rand() for _ in range(self.num_sampling_balls)]
        self.num_traces = num_traces
        self.traces = []

        if data_file is not None:
            self.traces = loadpklz(data_file)
        else:
            func = partial(gen_trace_in_a_ball, self.num_traces, self.config.normalized_X0_RMAX, self.config.normalized_Theta, self.config.nonzero_dims, self.config.unnormalize, self.config.simulate, T_MAX)
            with Pool(4) as p:
                self.traces = list(tqdm.tqdm(p.imap(func, self.X0_centers), total=len(self.X0_centers)))
            # self.traces = list(tqdm.tqdm(map(func, self.X0_centers), total=len(self.X0_centers)))

        savepklz(self.traces, './traces_%d.pklz'%num_traces)

        self.num_t = self.traces[0].shape[1] - 1
        # from IPython import embed; embed()
        self.shuffle = shuffle
        self.idx = 0
        self.idx_map = []
        self.len = self.num_sampling_balls * self.num_t

    def distance(self, x1, x2):
        return np.sqrt(((np.array(x1) - np.array(x2))**2).sum())

    def __len__(self):
        return self.len

    def __iter__(self):
        if self.shuffle:
            self.idx_map = np.random.permutation(self.len)
        else:
            self.idx_map = np.array(range(self.len))
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < self.len:
            idx = self.idx_map[self.idx]
            self.idx += 1
            idx_trace = idx // self.num_t
            idx_t = idx % self.num_t + 1
            traces = self.traces[idx_trace]
            x0 = traces[0,0,1::]
            r = self.X0_rs[idx_trace]
            t = traces[0,idx_t,0]
            xi0 = traces[0,idx_t,1::]
            xi1s = traces[1:,idx_t,1::]
            return torch.from_numpy(np.array(x0).astype('float32')).view(-1),\
                torch.from_numpy(np.array(r).astype('float32')).view(-1),\
                torch.from_numpy(np.array(xi0).astype('float32')).view(-1),\
                torch.from_numpy(np.array(xi1s).astype('float32')),\
                torch.from_numpy(np.array(t).astype('float32')).view(-1)
        else:
            self.idx = 0
            raise StopIteration

def get_dataloader(config, num_traces_train, num_traces_val, batch_size=16, data_file=[None, None]):
    train_loader = DiscriData(config, num_traces_train, num_sampling_balls=batch_size, data_file=data_file[0], shuffle=True)
    val_loader = DiscriData(config, num_traces_val, num_sampling_balls=batch_size, data_file=data_file[0], shuffle=True)
    return train_loader, val_loader

if __name__ == '__main__':
    DiscriData(30)
