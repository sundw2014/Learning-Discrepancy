import sys
sys.path.append('..')
from examples.drone import TC_Simulate
import numpy as np

def simulate(init, t_max):
    return TC_Simulate(init, t_max)[::5, :]

num_dim_state = 9
num_dim_input = 6
num_dim_output = 3
def observe_for_output(state):
    if len(state.shape) == 1:
        return state[:num_dim_reachibility]
    elif len(state.shape) == 2:
        return state[:,:num_dim_reachibility]
    else:
        raise ValueError('wrong state.shape')

def observe_for_input(state):
    if len(state.shape) == 1:
        return state[3:]
    elif len(state.shape) == 2:
        return state[:,3:]
    else:
        raise ValueError('wrong state.shape')

# range of initial states
Theta_lower = np.array([0., 0., 0., -0.1, -0.1, 0., 0., 0., 0.])
Theta_higher = np.array([0., 0., 0., 0.1, 0.1, 0., 0., 0., 0.])
Theta = np.array([Theta_lower, Theta_higher]).T

# normlization
normalization_scale = np.array([1.,1,1,10,10,1,1,1,1])
normalization_offset = np.zeros(num_dim_state)

nonzero_dims = (Theta[:,1] - Theta[:,0]) > 0

def normalize(x):
    return (x - normalization_offset) * normalization_scale

def unnormalize(x):
    return (x / normalization_scale) + normalization_offset

normalized_Theta = np.array([normalize(Theta[:,0]), normalize(Theta[:,1])]).T
normalized_X0_RMAX = 0.5
