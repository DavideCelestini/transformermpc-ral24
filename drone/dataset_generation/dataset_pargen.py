import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from dynamics.quadrotor import *
from multiprocessing import Pool, set_start_method
import itertools
from tqdm import tqdm

def for_computation(input):
    # Input unpacking
    current_data_index = input[0]
    other_args = input[1]
    quad_model = other_args['quad_model']

    # Randomic sample of initial and final conditions
    init_state, target_state = sample_init_target()

    # Output dictionary initialization
    out = {'feasible' : True,
           'states_cvx' : [],
           'actions_cvx' : [],
           'states_scp': [],
           'actions_scp' : [],
           'target_state' : [],
           'dtime' : [],
           'time' : []
           }

    # Solve simplified problem -> without obstacle avoidance
    states_cvx_i, actions_cvx_i, J_cvx_i, feas_cvx_i = ocp_no_obstacle_avoidance(quad_model, init_state, target_state, initial_guess='line')
    
    if np.char.equal(feas_cvx_i,'optimal'):
        
        #  Solve scp with obstacles
        states_scp_i, actions_scp_i, J_scp_i, feas_scp_i, iter_scp_i = ocp_obstacle_avoidance(quad_model, states_cvx_i, actions_cvx_i, init_state, target_state)

        if np.char.equal(feas_scp_i,'optimal'):
            # Save cvx and scp problems in the output dictionary
            out['states_cvx'] = states_cvx_i[:-1,:]
            out['actions_cvx'] = actions_cvx_i
            
            out['states_scp'] = states_scp_i[:-1,:]
            out['actions_scp'] = actions_scp_i

            out['target_state'] = target_state
            out['dtime'] = dt
            out['time'] = np.linspace(0, T, S+1)[:-1]
        else:
            out['feasible'] = False
    else:
        out['feasible'] = False
    
    return out

if __name__ == '__main__':

    N_data = 200000
    set_start_method('spawn')

    n_S = 6 # state size
    n_A = 3 # action size

    # Model initialization
    quad_model = QuadModel()
    other_args = {
        'quad_model' : quad_model
    }

    print('Dataset generating for: ', dataset_scenario)

    states_cvx = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
    actions_cvx = np.empty(shape=(N_data, n_time_rpod, n_A), dtype=float) # [m/s]

    states_scp = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
    actions_scp = np.empty(shape=(N_data, n_time_rpod, n_A), dtype=float) # [m/s]

    target_state = np.empty(shape=(N_data, n_S), dtype=float)
    dtime = np.empty(shape=(N_data, ), dtype=float)
    time = np.empty(shape=(N_data, n_time_rpod), dtype=float)

    i_unfeas = []

    # Pool creation --> Should automatically select the maximum number of processes
    p = Pool(processes=16)
    for i, res in enumerate(tqdm(p.imap(for_computation, zip(np.arange(N_data), itertools.repeat(other_args))), total=N_data)):
    #for i in np.arange(N_data):
    #    res = for_computation(i)
        # If the solution is feasible save the optimization output
        if res['feasible']:
            states_cvx[i,:,:] = res['states_cvx']
            actions_cvx[i,:,:] = res['actions_cvx']

            states_scp[i,:,:] = res['states_scp']
            actions_scp[i,:,:] = res['actions_scp']
        
            target_state[i,:] = res['target_state']
            dtime[i] = res['dtime']
            time[i,:] = res['time']

        # Else add the index to the list
        else:
            i_unfeas += [ i ]
        
        if i % 40000 == 0:
            np.savez_compressed(root_folder + '/dataset' + dataset_scenario +'/dataset-quad-v05-scp' + str(i), states_scp = states_scp, actions_scp = actions_scp, i_unfeas = i_unfeas)
            np.savez_compressed(root_folder + '/dataset' + dataset_scenario +'/dataset-quad-v05-cvx' + str(i), states_cvx = states_cvx, actions_cvx = actions_cvx, i_unfeas = i_unfeas)
            np.savez_compressed(root_folder + '/dataset' + dataset_scenario +'/dataset-quad-v05-param' + str(i), target_state = target_state, time = time, dtime = dtime, i_unfeas = i_unfeas)

    # Remove unfeasible data points
    if i_unfeas:
        states_cvx = np.delete(states_cvx, i_unfeas, axis=0)
        actions_cvx = np.delete(actions_cvx, i_unfeas, axis=0)

        states_scp = np.delete(states_scp, i_unfeas, axis=0)
        actions_scp = np.delete(actions_scp, i_unfeas, axis=0)
        
        target_state = np.delete(target_state, i_unfeas, axis=0)
        dtime = np.delete(dtime, i_unfeas, axis=0)
        time = np.delete(time, i_unfeas, axis=0)

    #  Save dataset (local folder for the workstation)
    np.savez_compressed(root_folder + '/dataset' + dataset_scenario +'/dataset-quad-v05-scp', states_scp = states_scp, actions_scp = actions_scp)
    np.savez_compressed(root_folder + '/dataset' + dataset_scenario +'/dataset-quad-v05-cvx', states_cvx = states_cvx, actions_cvx = actions_cvx)
    np.savez_compressed(root_folder + '/dataset' + dataset_scenario +'/dataset-quad-v05-param', target_state = target_state, time = time, dtime = dtime)
