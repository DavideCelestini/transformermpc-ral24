import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt
import random as rnd

from dynamics.orbit_dynamics import dynamics_roe_optimization, roe_to_rtn_horizon
from optimization.rpod_scenario import oe_0_ref, t_0, n_time_rpod
from optimization.rpod_scenario import sample_init_final
from optimization.ocp import *
from multiprocessing import Pool
from tqdm import tqdm

def for_computation(current_data_index):
    # Communicate current data index
    #print(current_data_index)

    hrz_i, state_roe_0, dock_param, _ = sample_init_final()

    # Dynamics Matrices Precomputations
    stm_i, cim_i, psi_i, oe_i, time_i, dt_i = dynamics_roe_optimization(oe_0_ref, t_0, hrz_i, n_time_rpod)

    # Solve transfer cvx
    states_roe_cvx_i, actions_cvx_i, feas_cvx_i = ocp_cvx(stm_i, cim_i, psi_i, state_roe_0, dock_param, n_time_rpod)

    # Output dictionary initialization
    out = {'feasible' : True,
           'states_roe_cvx' : [],
           'states_rtn_cvx' : [],
           'actions_cvx' : [],
           'states_roe_scp': [],
           'states_rtn_scp' : [],
           'actions_scp' : [],
           'target_state' : [],
           'horizons' : [],
           'dtime' : [],
           'time' : [],
           'oe' : []
           }
    
    if np.char.equal(feas_cvx_i,'optimal'):
        
        # Mapping done after the feasibility check to avoid NoneType errors
        states_rtn_cvx_i = roe_to_rtn_horizon(states_roe_cvx_i, oe_i, n_time_rpod)

        #  Solve transfer scp
        states_roe_scp_i, actions_scp_i, feas_scp_i, iter_scp_i, J_vect_scp_i, runtime_scp_i = solve_scp(stm_i, cim_i, psi_i, state_roe_0, dock_param, states_roe_cvx_i, n_time_rpod)
        # states_rtn_scp_i = roe_to_rtn_horizon(states_roe_scp_i, oe_i, n_time_rpod)

        if np.char.equal(feas_scp_i,'optimal'):
            # Mapping done after feasibility check to avoid NoneType errors
            states_rtn_scp_i = roe_to_rtn_horizon(states_roe_scp_i, oe_i, n_time_rpod)

            # Save cvx and scp problems in the output dictionary
            out['states_roe_cvx'] = np.transpose(states_roe_cvx_i)
            out['states_rtn_cvx'] = np.transpose(states_rtn_cvx_i)
            out['actions_cvx'] = np.transpose(actions_cvx_i)
            
            out['states_roe_scp'] = np.transpose(states_roe_scp_i)
            out['states_rtn_scp'] = np.transpose(states_rtn_scp_i)
            out['actions_scp'] = np.transpose(actions_scp_i)

            out['target_state'] = dock_param['state_rtn_target']
            out['horizons'] = hrz_i
            out['dtime'] = dt_i
            out['time'] = np.transpose(time_i)
            out['oe'] = np.transpose(oe_i)
        else:
            out['feasible'] = False
    else:
        out['feasible'] = False
    
    return out

if __name__ == '__main__':

    N_data = 200000

    n_S = 6 # state size
    n_A = 3 # action size

    states_roe_cvx = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m]
    states_rtn_cvx = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
    actions_cvx = np.empty(shape=(N_data, n_time_rpod, n_A), dtype=float) # [m/s]

    states_roe_scp = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m]
    states_rtn_scp = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
    actions_scp = np.empty(shape=(N_data, n_time_rpod, n_A), dtype=float) # [m/s]

    target_state = np.empty(shape=(N_data, n_S), dtype=float)
    horizons = np.empty(shape=(N_data, ), dtype=float)
    dtime = np.empty(shape=(N_data, ), dtype=float)
    time = np.empty(shape=(N_data, n_time_rpod), dtype=float)
    oe = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float)

    i_unfeas = []

    # Pool creation --> Should automatically select the maximum number of processes
    p = Pool(processes=24)
    for i, res in enumerate(tqdm(p.imap(for_computation, np.arange(N_data)), total=N_data)):
        # If the solution is feasible save the optimization output
        if res['feasible']:
            states_roe_cvx[i,:,:] = res['states_roe_cvx']
            states_rtn_cvx[i,:,:] = res['states_rtn_cvx']
            actions_cvx[i,:,:] = res['actions_cvx']
                
            states_roe_scp[i,:,:] = res['states_roe_scp']
            states_rtn_scp[i,:,:] = res['states_rtn_scp']
            actions_scp[i,:,:] = res['actions_scp']

            target_state[i,:] = res['target_state']
            horizons[i] = res['horizons']
            dtime[i] = res['dtime']
            time[i,:] = res['time']
            oe[i,:,:] = res['oe']

        # Else add the index to the list
        else:
            i_unfeas += [ i ]
        
        if i % 100000 == 0:
            np.savez_compressed(root_folder + '/dataset/dataset-rpod-v05-scp' + str(i), states_roe_scp = states_roe_scp, states_rtn_scp = states_rtn_scp, actions_scp = actions_scp, i_unfeas = i_unfeas)
            np.savez_compressed(root_folder + '/dataset/dataset-rpod-v05-cvx' + str(i), states_roe_cvx = states_roe_cvx, states_rtn_cvx = states_rtn_cvx, actions_cvx = actions_cvx, i_unfeas = i_unfeas)
            np.savez_compressed(root_folder + '/dataset/dataset-rpod-v05-param' + str(i), target_state = target_state, time = time, oe = oe, dtime = dtime, horizons = horizons, i_unfeas = i_unfeas)

    # Remove unfeasible data points
    if i_unfeas:
        states_roe_cvx = np.delete(states_roe_cvx, i_unfeas, axis=0)
        states_rtn_cvx = np.delete(states_rtn_cvx, i_unfeas, axis=0)
        actions_cvx = np.delete(actions_cvx, i_unfeas, axis=0)
        
        states_roe_scp = np.delete(states_roe_scp, i_unfeas, axis=0)
        states_rtn_scp = np.delete(states_rtn_scp, i_unfeas, axis=0)
        actions_scp = np.delete(actions_scp, i_unfeas, axis=0)
        
        target_state = np.delete(target_state, i_unfeas, axis=0)
        horizons = np.delete(horizons, i_unfeas, axis=0)
        dtime = np.delete(dtime, i_unfeas, axis=0)
        time = np.delete(time, i_unfeas, axis=0)
        oe = np.delete(oe, i_unfeas, axis=0)

    #  Save dataset (local folder for the workstation)
    np.savez_compressed(root_folder + '/dataset/dataset-rpod-v05-scp', states_roe_scp = states_roe_scp, states_rtn_scp = states_rtn_scp, actions_scp=actions_scp)
    np.savez_compressed(root_folder + '/dataset/dataset-rpod-v05-cvx', states_roe_cvx = states_roe_cvx, states_rtn_cvx = states_rtn_cvx, actions_cvx=actions_cvx)
    np.savez_compressed(root_folder + '/dataset/dataset-rpod-v05-param', target_state = target_state, time = time, oe = oe, dtime = dtime, horizons = horizons)