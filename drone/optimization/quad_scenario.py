import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np

# state-control dimensions
n_x = 6 # (px, py, pz, vx, vy, vz) - number of state variables
n_u = 3 # (ux, uy, uz) - number of control variables\
# time problem constants
S = 100 # number of control switches
n_time_rpod = S
R = np.eye(n_u) # cost term
# constants
u_max = 10
mass = 32.0
drag_coefficient = 0.2 
# obstacle
dataset_scenario = '/random_target_forest'
if 'forest' in dataset_scenario:
    T = 100.0 # max final time horizon in sec
    dt = T / S
    obs_positions = np.array([
        [-1.4, -0.1, 0.3],
        [-0.7, 0.3, 0.5],
        [-0.3, 0.25, 0.65],
        [0, -0.3, 0.4]])
    obs_radii = np.array([
        0.3,
        0.2,
        0.2,
        0.2])
elif 'minimum' in dataset_scenario:
    T = 50.0 # max final time horizon in sec
    dt = T / S
    obs_positions = np.array([
        #[-0.5, 0.2, 0.1],
        #[-0.1, -0.3, -0.2],
        [0.5, 0.3, 1+0.3],
        [0.5, -0.3, 1+0.3],
        [0.5, 0.3, 1-0.3],
        [0.5, -0.3, 1-0.3]])
    obs_radii = np.array([
        #0.2,
        #0.3,
        0.5,
        0.5,
        0.5,
        0.5])
n_obs = obs_positions.shape[0]
obs_radii_deltas = 0.025
n_obs = obs_positions.shape[0] # number of obstacles
# initial and final conditions (uncomment to manually fix)
#x_init = jnp.array([-1.9, 0.05, 0.2, 0, 0, 0])
#x_final = jnp.zeros(n_x)

Q = np.zeros((n_x, n_x))
x_ref = np.zeros(n_x)

# Optimization interface
iter_max_SCP = 20
trust_region0 = 10.
trust_regionf = 0.005
J_tol = 10**(-6)