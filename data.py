from __future__ import print_function
import multiprocessing
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

def mute():
    sys.stdout = open(os.devnull, 'w')

def sample_trajs(num_traces, sample_from_D0, sample_traj, sample_ref, D0):
    # import ipdb; ipdb.set_trace()
    traces = []
    traces.append(sample_ref(D0))
    for id_trace in range(num_traces):
        traces.append(sample_traj(sample_from_D0(D0)))
    return np.array(traces)

class DiscriData(data.Dataset):
    """DiscriData."""
    def __init__(self, config, num_D0s=100, num_traces=10, num_t=100, data_file=None, shuffle=True):
        super(DiscriData, self).__init__()

        self.config = config

        self.D0s = [self.config.sample_D0_from_P() for _ in range(num_D0s)]
        self.D0_mean, self.D0_std = self.config.get_D0_normalization_factor()

        use_precomputed_data = os.path.exists(data_file)

        if use_precomputed_data:
            [self.traces, self.data] = loadpklz(data_file)
        else:
            func = partial(sample_trajs, num_traces, self.config.sample_from_D0, self.config.sample_traj, self.config.sample_ref)
            with Pool(min([1, multiprocessing.cpu_count()-3]), initializer=mute) as p:
                self.traces = list(tqdm.tqdm(p.imap(func, self.D0s), total=len(self.D0s)))
            # self.traces = list(tqdm.tqdm(map(func, self.D0s), total=len(self.D0s)))

            # import ipdb; ipdb.set_trace()

            self.data = []
            for i in range(len(self.traces)):
                traces = self.traces[i]
                for j in range(traces.shape[0]-1):
                    sampled_ts = np.array([config.sample_t_from_P() for _ in range(num_t)]).reshape(-1,1)
                    ts = traces[j+1,:,0].reshape(1,-1)
                    idx_ts_j = np.abs(sampled_ts - ts).argmin(axis=1)
                    ts = traces[0,:,0].reshape(1,-1)
                    idx_ts_0 = np.abs(sampled_ts - ts).argmin(axis=1)

                    for (idx_t0, idx_tj, sampled_t) in zip(idx_ts_0, idx_ts_j, sampled_ts):
                        self.data.append([self.D0s[i], sampled_t, traces[0, idx_t0, 1:], traces[j+1, idx_tj, 1:]])
            if not use_precomputed_data:
                savepklz([self.traces, self.data], data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        D0 = data[0]
        t = data[1]
        ref = data[2]
        xt = data[3]
        return torch.from_numpy(((np.array(D0)-self.D0_mean)/self.D0_std).astype('float32')).view(-1),\
            torch.from_numpy(np.array(t).astype('float32')).view(-1),\
            torch.from_numpy(np.array(ref).astype('float32')).view(-1),\
            torch.from_numpy(np.array(xt).astype('float32')).view(-1)

def get_dataloader(config, num_D0s_train, num_D0s_val, batch_size=16, data_file=[None, None]):
    train_loader = torch.utils.data.DataLoader(
        DiscriData(config, num_D0s_train, data_file=data_file[0]), batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        DiscriData(config, num_D0s_val, data_file=data_file[1]), batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True)

    return train_loader, val_loader
