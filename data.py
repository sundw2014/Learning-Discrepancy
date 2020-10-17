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

def gen_trace_in_a_ball(num_traces, R_MAX, normalized_Theta, nonzero_dims, unnormalize, simulate, T_MAX, c):
    traces = []
    for _ in range(num_traces):
        # sample a direction
        d = np.random.randn(len(nonzero_dims))
        d = d / np.sqrt((d**2).sum())

        while True:
            # import ipdb; ipdb.set_trace()
            samples = c[nonzero_dims].reshape(1,-1) + R_MAX * np.random.rand(1000,1) * d.reshape(1,-1)
            tmp = np.tile(normalized_Theta[:,0].reshape(1,-1), (1000, 1))
            tmp[:, nonzero_dims] = samples

            hit = np.where(np.logical_and(\
             (samples>=normalized_Theta[:,0].reshape(1,-1)).sum(axis=1) == len(c),
             (samples<=normalized_Theta[:,1].reshape(1,-1)).sum(axis=1) == len(c)))[0]
            if len(hit)>0:
                break
        # import ipdb; ipdb.set_trace()
        point = unnormalize(samples[hit[0],:])
        _trace = simulate(point.tolist(), T_MAX)
        traces.append(_trace)
    return traces


class DiscriData(data.Dataset):
    """DiscriData."""
    def __init__(self, config, num_traces, num_sampling_balls=100, T_MAX=10.0):
        super(DiscriData, self).__init__()

        self.config = config

        def sample_X0_center():
            num_dim_state = self.config.num_dim_state
            normalized_Theta = self.config.normalized_Theta
            sample = np.random.rand(num_dim_state) * (normalized_Theta[:,1] - normalized_Theta[:,0]) + normalized_Theta[:,0]
            return sample

        # generate traces
        self.num_sampling_balls = num_sampling_balls
        self.X0_centers = [sample_X0_center() for _ in range(self.num_sampling_balls)]
        self.num_traces = num_traces
        self.traces = []

        func = partial(gen_trace_in_a_ball, self.num_traces, self.config.normalized_X0_RMAX, self.config.normalized_Theta, self.config.nonzero_dims, self.config.unnormalize, self.config.simulate, T_MAX)
        # with Pool(30) as p:
            # self.traces = list(tqdm.tqdm(p.imap(func, self.X0_centers), total=len(self.X0_centers)))
        self.traces = list(tqdm.tqdm(map(func, self.X0_centers), total=len(self.X0_centers)))

        self.num_t = len(self.traces[0][0]) - 1
        # from IPython import embed; embed()

    def distance(self, x1, x2):
        return np.sqrt(((np.array(x1) - np.array(x2))**2).sum())

    def __getitem__(self, index):
        while True:
            _index = index
            NB = self.num_sampling_balls
            NT = self.num_t
            NTr = self.num_traces
            ib = _index // (NT * NTr * (NTr - 1))
            _index = _index % (NT * NTr * (NTr - 1))
            it =  _index // (NTr * (NTr - 1))
            it = it + 1
            _index = _index % (NTr * (NTr - 1))
            i1 = _index // (NTr - 1)
            _index = _index % (NTr - 1)
            i2 = _index
            i2 = i2 if i2<i1 else i2+1

            trace0 = self.traces[ib][i1]
            trace1 = self.traces[ib][i2]
            x0 = trace0[0][1::]
            x1 = trace1[0][1::]
            t = trace0[it][0]
            xi0 = trace0[it][1::]
            xi1 = trace1[it][1::]
            r = self.distance(self.config.normalize(x0), self.config.normalize(x1))

            break
            # eps = 1e-4
            # if init_r > eps:
            #     break
            # else:
            #     index = (index + 1) % self.__len__()
        # print(it, i1, i2)
        return torch.from_numpy(np.array(x0).astype('float32')).view(-1),\
            torch.from_numpy(np.array(r).astype('float32')).view(-1),\
            torch.from_numpy(np.array(xi0).astype('float32')).view(-1),\
            torch.from_numpy(np.array(xi1).astype('float32')).view(-1),\
            torch.from_numpy(np.array(t).astype('float32')).view(-1),\
            #TODO normalization

    def __len__(self):
        return self.num_sampling_balls * self.num_t * self.num_traces * (self.num_traces - 1)

def get_dataloader(config, num_traces_train, num_traces_val, batch_size=16):
    train_loader = torch.utils.data.DataLoader(
        DiscriData(config, num_traces_train), batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DiscriData(config, num_traces_val), batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True)

    return train_loader, val_loader

if __name__ == '__main__':
    DiscriData(30)
