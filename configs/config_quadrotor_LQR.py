import sys
sys.path.append('..')
from examples.quadrotor_LQR import TC_Simulate
import numpy as np

def simulate(init, t_max):
    return np.array(TC_Simulate("random", init, t_max))[::5,:]

num_dim_state = 12
num_dim_input = 12
num_dim_output = 3
def observe_for_output(state):
    if len(state.shape) == 1:
        return state[[0,2,4]]
    elif len(state.shape) == 2:
        return state[:,[0,2,4]]
    else:
        raise ValueError('wrong state.shape')

def observe_for_input(state):
    return state

# range of initial states
Theta_lower = np.array([-1.,]*12)
Theta_higher = np.array([1.,]*12)
Theta = np.array([Theta_lower, Theta_higher]).T

# normlization
normalization_scale = np.array([1.,]*12)
normalization_offset = np.zeros(num_dim_state)

nonzero_dims = (Theta[:,1] - Theta[:,0]) > 0

def normalize(x):
    return (x - normalization_offset) * normalization_scale

def unnormalize(x):
    return (x / normalization_scale) + normalization_offset

normalized_Theta = np.array([normalize(Theta[:,0]), normalize(Theta[:,1])]).T
normalized_X0_RMAX = 0.6
