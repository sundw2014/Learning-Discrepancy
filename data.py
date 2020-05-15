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

# from examples.vanderpol import TC_Simulate
# from examples.Brusselator import TC_Simulate
# from examples.jet_engine import TC_Simulate
from examples.drone import TC_Simulate

def gen_trace_in_a_ball(num_traces, R_MAX, normalized_D, D, T_MAX, c):
    unnormalize = lambda x: (x / normalized_D) * (D[:,1] - D[:,0]) + D[:,0]

    traces = []
    for _ in range(num_traces):
        while True:
            samples = R_MAX * (np.random.rand(len(c), 10) - 0.5)
            hit = np.where(np.sqrt((samples ** 2).sum(axis=0)) < R_MAX)[0]
            if len(hit)>0:
                break
        point = unnormalize(np.array(c) + samples[:,hit[0]])
        _trace = TC_Simulate(point.tolist(), T_MAX)[::5,:]
        traces.append(_trace)
    return traces


class DiscriData(data.Dataset):
    """DiscriData."""
    def __init__(self, num_traces, num_sampling_balls=100, D=None, T_MAX=10.0):
        super(DiscriData, self).__init__()
        # FIXME: D is the initial state space
        if D is None:
            # D = np.array([[1.25, 1.55], [2.25, 2.35]]) # van de pol
            # D = np.array([[0, 0.3], [0, 0.3]]) # van de pol
            # D = np.array([[0, 1.0], [0, 1.0]]) # van de pol
            # D = np.array([[0, 2.0], [0, 3.0]]) # van de pol
            # D = np.array([[0.3, 1.3], [0.3, 1.3]]) # jet engine
            # D = np.array([[0.3, 1.5], [0., 0.3]]) # Brusselator
            # drone
            set_lower = np.array([1-0.125, 1-0.125, 4.95, -0.1, -0.1, -0.01, -0.01, -0.01, 0.0])
            set_higher = np.array([1+0.125, 1+0.125, 5.05, 0.1, 0.1, 0.01, 0.01, 0.01, 2 * np.pi - 1e-6])
            # set_lower = np.array([1-0.5, 1-0.5, 4.5, -0.1, -0.1, -0.01, -0.01, -0.01, 0.0])
            # set_higher = np.array([1+0.5, 1+0.5, 5.5, 0.1, 0.1, 0.01, 0.01, 0.01, 2 * np.pi - 1e-6])
            D = np.array([set_lower, set_higher]).T

        normalized_D = np.ones(D.shape[0])
        def unnormalize(x):
            return (x / normalized_D) * (D[:,1] - D[:,0]) + D[:,0]

        # R_MAX = (normalized_D.prod() / num_sampling_balls) ** (1./D.shape[0])
        R_MAX = (normalized_D.prod() / num_sampling_balls) ** (1./4)

        def sample_in_D():
            sample = np.random.rand(D.shape[0]) * normalized_D
            return sample

        # generate traces
        # self.traces = [TC_Simulate('Default', sample_in_D(), T_MAX) for _ in range(num_traces)]
        self.num_sampling_balls = num_sampling_balls
        self.centers = [sample_in_D() for _ in range(self.num_sampling_balls)]
        self.num_traces = num_traces
        self.traces = []

        func = partial(gen_trace_in_a_ball, self.num_traces, R_MAX, normalized_D, D, T_MAX)
        with Pool(30) as p:
            self.traces = list(tqdm.tqdm(p.imap(func, self.centers), total=len(self.centers)))

        self.num_t = len(self.traces[0][0]) - 1
        # from IPython import embed; embed()

    def norm(self, x1, x2):
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

            r = self.norm(x0, x1)

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
