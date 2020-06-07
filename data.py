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

from config import TC_Simulate, normalized_D, normalize, unnormalize, sampling_RMAX, num_dim, observe

def gen_trace_in_a_ball(num_traces, R_MAX, normalized_D, unnormalize, TC_Simulate, T_MAX, c):
    traces = []
    for _ in range(num_traces):
        # sample a direction
        d = np.random.randn(len(c))
        d = d / np.sqrt((d**2).sum())

        while True:
            samples = c.reshape(1,-1) + R_MAX * np.random.rand(1000,1) * d.reshape(1,-1)
            hit = np.where(np.logical_and(\
             (samples>0).sum(axis=1) == len(c),
             (samples<normalized_D.reshape(1,-1)).sum(axis=1) == len(c)))[0]
            if len(hit)>0:
                break
        # import ipdb; ipdb.set_trace()
        point = unnormalize(samples[hit[0],:])
        _trace = TC_Simulate(point.tolist(), T_MAX)[::5,:]
        traces.append(_trace)
    return traces


class DiscriData(data.Dataset):
    """DiscriData."""
    def __init__(self, num_traces, num_sampling_balls=100, T_MAX=10.0):
        super(DiscriData, self).__init__()

        def sample_in_D():
            # start = np.ones(num_dim) * sampling_RMAX
            # end = normalized_D - sampling_RMAX
            # assert (end >= start).sum() == num_dim
            start = np.zeros(num_dim)
            end = normalized_D
            assert (end >= start).sum() == num_dim

            sample = np.random.rand(num_dim) * (end - start) + start
            return sample

        # generate traces
        self.num_sampling_balls = num_sampling_balls
        self.centers = [sample_in_D() for _ in range(self.num_sampling_balls)]
        self.num_traces = num_traces
        self.traces = []

        func = partial(gen_trace_in_a_ball, self.num_traces, sampling_RMAX, normalized_D, unnormalize, TC_Simulate, T_MAX)
        with Pool(30) as p:
            self.traces = list(tqdm.tqdm(p.imap(func, self.centers), total=len(self.centers)))
        # self.traces = list(tqdm.tqdm(map(func, self.centers), total=len(self.centers)))

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
            xi0 = observe(trace0[it][1::])
            xi1 = observe(trace1[it][1::])
            r = self.distance(normalize(x0), normalize(x1))

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

def get_dataloader(num_traces_train, num_traces_val, batch_size=16):
    train_loader = torch.utils.data.DataLoader(
        DiscriData(num_traces_train), batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DiscriData(num_traces_val), batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True)

    return train_loader, val_loader

class DiscriDataTonly(data.Dataset):
    """DiscriDataTonly."""
    def __init__(self, num_traces, D=None, T_MAX=10.0):
        super(DiscriDataTonly, self).__init__()

        # generate traces
        raw_dataset = DiscriData(num_traces, D, T_MAX)
        self.num_t = raw_dataset.num_t

        normalized_discri = []
        t = []
        for it in range(raw_dataset.num_t):
            _normalized_discri = []
            for ipair in range(raw_dataset.num_traces * (raw_dataset.num_traces - 1)):
                data = [d.detach().numpy() for d in raw_dataset[it * raw_dataset.num_traces * (raw_dataset.num_traces - 1) + ipair]]
                _normalized_discri.append(data[3]/data[1])
                if ipair == 0:
                    t.append(data[2])
            normalized_discri.append(np.max(_normalized_discri))

        self.data = [t, normalized_discri]

    def __getitem__(self, index):
        index = index % self.num_t
        return torch.from_numpy(np.array(self.data[0][index]).astype('float32')).view(-1), \
            torch.from_numpy(np.array(self.data[1][index]).astype('float32')).view(-1)
            #TODO normalization

    def __len__(self):
        return self.num_t * 100

def get_dataloader_t_only(num_traces_train, num_traces_val, batch_size=16):
    train_loader = torch.utils.data.DataLoader(
        DiscriDataTonly(num_traces_train), batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DiscriDataTonly(num_traces_val), batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True)

    return train_loader, val_loader

if __name__ == '__main__':
    DiscriData(30)
