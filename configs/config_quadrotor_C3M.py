import sys
sys.path.append('..')
from examples.quadrotor_C3M import TC_Simulate
import numpy as np

class Simulator(object):
    def __init__(self):
        super(Simulator, self).__init__()
        self.simu = TC_Simulate()
    def __call__(self, init, t_max):
        return self.simu("random", init, t_max)[::10,:]

simulate = Simulator()

num_dim_state = 8
num_dim_input = 8
num_dim_output = 3
def observe_for_output(state):
    if len(state.shape) == 1:
        return state[:3]
    elif len(state.shape) == 2:
        return state[:,:3]
    else:
        raise ValueError('wrong state.shape')

def observe_for_input(state):
    return state

# range of initial states
Theta_lower = np.array([-1.,]*8)
Theta_higher = np.array([1.,]*8)
Theta = np.array([Theta_lower, Theta_higher]).T

# normlization
normalization_scale = np.array([1.,]*8)
normalization_offset = np.zeros(num_dim_state)

nonzero_dims = (Theta[:,1] - Theta[:,0]) > 0

def normalize(x):
    return (x - normalization_offset) * normalization_scale

def unnormalize(x):
    return (x / normalization_scale) + normalization_offset

normalized_Theta = np.array([normalize(Theta[:,0]), normalize(Theta[:,1])]).T
normalized_X0_RMAX = 0.6
