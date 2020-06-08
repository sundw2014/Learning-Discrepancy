from examples.drone import TC_Simulate
import numpy as np

num_dim = 9
num_dim_observable = 3
def observe(state):
    if len(state.shape) == 1:
        return state[:num_dim_observable]
    elif len(state.shape) == 2:
        return state[:,:num_dim_observable]
    else:
        raise ValueError('wrong state.shape')

set_lower = np.array([-3., -3., -3., -1, -1, -1, 0., 0., 0.])
set_higher = np.array([3., 3., 3., 1, 1, 1, 2*np.pi, 2*np.pi, 2 * np.pi])
D = np.array([set_lower, set_higher]).T
normalized_D = np.array([1,1,1,1,1,1,1,1,0.1])
sampling_RMAX = 1.

idx_zeros = (D[:,1] - D[:,0]) == 0
eps = np.zeros(num_dim)
eps[idx_zeros] = 1e-3

def normalize(x):
    assert (x[idx_zeros] == D[idx_zeros,0]).sum() == idx_zeros.sum()
    return (x - D[:,0]) / (D[:,1] - D[:,0] + eps) * normalized_D
def unnormalize(x):
    assert np.logical_and(x < normalized_D, x > 0).sum() == num_dim
    return (x / normalized_D) * (D[:,1] - D[:,0]) + D[:,0]
