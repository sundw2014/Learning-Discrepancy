import sys
sys.path.append('..')
from examples.jet_engine import TC_Simulate
import numpy as np

def simulate(init, t_max):
    return np.array(TC_Simulate("Default", init, t_max))[::10,:]

num_dim_state = 2
num_dim_input = 2
num_dim_output = 2
def observe_for_output(state):
    return state

def observe_for_input(state):
    return state

# range of initial states
Theta_lower = np.array([0.3, 0.3])
Theta_higher = np.array([1.3, 1.3])
Theta = np.array([Theta_lower, Theta_higher]).T

# normlization
normalization_scale = np.array([1.,1])
normalization_offset = np.zeros(num_dim_state)

nonzero_dims = (Theta[:,1] - Theta[:,0]) > 0

def normalize(x):
    return (x - normalization_offset) * normalization_scale

def unnormalize(x):
    return (x / normalization_scale) + normalization_offset

normalized_Theta = np.array([normalize(Theta[:,0]), normalize(Theta[:,1])]).T
normalized_X0_RMAX = 0.5
