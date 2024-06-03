import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import torch

import decision_transformer.manage as DT_manager
from dynamics.freeflyer import FreeflyerModel, ocp_no_obstacle_avoidance, ocp_obstacle_avoidance, compute_constraint_to_go, sample_init_target
from optimization.ff_scenario import n_time_rpod, N_STATE, N_ACTION, obs, iter_max_SCP, robot_radius, safety_margin
import time
import itertools
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

def for_computation(input_iterable):

    # Extract input
    current_idx = input_iterable[0]
    input_dict = input_iterable[1]
    model = input_dict['model']
    model_dag = input_dict['model_dag']
    test_loader = input_dict['test_loader']
    transformer_ws = input_dict['transformer_ws']
    mdp_constr = input_dict['mdp_constr']
    sample_init_final = input_dict['sample_init_final']

    # Output dictionary initialization
    out = {'feasible_cvx' : True,
           'feasible_scp_cvx' : True,
           'feasible_scp_dag' : True,
           'feasible_DT' : True,
           'J_vect_scp_cvx': [],
           'J_vect_scp_dag' : [],
           'J_vect_scp_DT': [],
           'J_cvx' : [],
           'J_DT' : [],
           'J_dag' : [],
           'iter_scp_cvx': [],
           'iter_scp_dag' : [],
           'iter_scp_DT': [],
           'runtime_cvx': [],
           'runtime_DT': [],
           'runtime_dag' : [],
           'runtime_scp_cvx': [],
           'runtime_scp_dag' : [],
           'runtime_scp_DT': [],
           'ctgs0_cvx': [],
           'cvx_problem' : False,
           'test_dataset_ix' : [],
           'state_init' : [],
           'state_final' : []
          }
   
    test_sample = test_loader.dataset.getix(current_idx)
    data_stats = test_loader.dataset.data_stats
    if sample_init_final:
        state_init, state_final = sample_init_target()
        test_sample[0][0,:,:] = (torch.tensor(np.repeat(state_init[None,:], n_time_rpod, axis=0)) - data_stats['states_mean'])/(data_stats['states_std'] + 1e-6)
        test_sample[1][0,:,:] = torch.zeros((n_time_rpod,N_ACTION))
        test_sample[2][0,:,0] = torch.zeros((n_time_rpod,))
        test_sample[3][:,0] = torch.zeros((n_time_rpod,))
        test_sample[4][0,:,:] = (torch.tensor(np.repeat(state_final[None,:], n_time_rpod, axis=0)) - data_stats['goal_mean'])/(data_stats['goal_std'] + 1e-6)
        out['test_dataset_ix'] = test_sample[-1][0]
    else:
        if not mdp_constr:
            states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = test_sample
        else:
            states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = test_sample
        # print('Sampled trajectory ' + str(ix) + ' from test_dataset.')
        out['test_dataset_ix'] = ix[0]
        state_init = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
        state_final = np.array((goal_i[0, 0, :] * data_stats['goal_std'][0]) + data_stats['goal_mean'][0])

    out['state_init'] = state_init
    out['state_final'] = state_final
    ff_model = FreeflyerModel()


    ####### Warmstart Convex Problem RPOD
    try:
        runtime0_cvx = time.time()
        traj_cvx, _, _, feas_cvx = ocp_no_obstacle_avoidance(ff_model, state_init, state_final)
        runtime1_cvx = time.time()
        runtime_cvx = runtime1_cvx-runtime0_cvx
        states_cvx, actions_cvx = traj_cvx['states'], traj_cvx['actions_G']
    except:
        states_cvx = None
        actions_cvx = None
        feas_cvx = 'failure'
        runtime_cvx = None
    
    if np.char.equal(feas_cvx,'optimal'):
        states_ws_cvx = states_cvx # set warm start
        actions_ws_cvx = actions_cvx # set warm start
        out['J_cvx'] = np.sum(la.norm(actions_ws_cvx, ord=1, axis=0))
        # Evaluate Constraint Violation
        ctgs_cvx = compute_constraint_to_go(states_ws_cvx.T, obs['position'], (obs['radius'] + robot_radius)*safety_margin)
        ctgs0_cvx = ctgs_cvx[0,0]
        # Save cvx in the output dictionary
        out['runtime_cvx'] = runtime_cvx
        out['ctgs0_cvx'] = ctgs0_cvx
        out['cvx_problem'] = ctgs0_cvx == 0

        # Solve SCP
        runtime0_scp_cvx = time.time()
        traj_scp_cvx, J_vect_scp_cvx, iter_scp_cvx, feas_scp_cvx = ocp_obstacle_avoidance(ff_model, states_ws_cvx, actions_ws_cvx, state_init, state_final)
        runtime1_scp_cvx = time.time()
        runtime_scp_cvx = runtime1_scp_cvx - runtime0_scp_cvx
        
        if np.char.equal(feas_scp_cvx,'optimal'):
            # Save scp_cvx data in the output dictionary
            out['J_vect_scp_cvx'] = J_vect_scp_cvx
            out['iter_scp_cvx'] = iter_scp_cvx    
            out['runtime_scp_cvx'] = runtime_scp_cvx
        else:
            out['feasible_scp_cvx'] = False
    else:
        out['feasible_scp_cvx'] = False
        out['feasible_cvx'] = False

    ####### Warmstart Transformer
    # Import the Transformer
    if np.char.equal(feas_cvx,'optimal') and mdp_constr:
        rtg_0 = -out['J_cvx']
        if transformer_ws == 'dyn':
            DT_trajectory, runtime_DT = DT_manager.torch_model_inference_dyn(model, test_loader, test_sample, rtg_perc=None, ctg_perc=0., rtg=rtg_0, ctg_clipped=True)
        elif transformer_ws == 'ol':
            DT_trajectory, runtime_DT = DT_manager.torch_model_inference_ol(model, test_loader, test_sample, rtg_perc=None, ctg_perc=0., rtg=rtg_0, ctg_clipped=True)
    else:
        if transformer_ws == 'dyn':
            DT_trajectory, runtime_DT = DT_manager.torch_model_inference_dyn(model, test_loader, test_sample, rtg_perc=1., ctg_perc=0., rtg=None, ctg_clipped=True)
        elif transformer_ws == 'ol':
            DT_trajectory, runtime_DT = DT_manager.torch_model_inference_ol(model, test_loader, test_sample, rtg_perc=1., ctg_perc=0., rtg=None, ctg_clipped=True)    
    out['J_DT'] = np.sum(la.norm(DT_trajectory['dv_' + transformer_ws], ord=1, axis=0))
    states_ws_DT = np.hstack((DT_trajectory['xypsi_' + transformer_ws], state_final.reshape(-1,1))) # set warm start
    actions_ws_DT = DT_trajectory['dv_' + transformer_ws] # set warm start
    # Save DT in the output dictionary
    out['runtime_DT'] = runtime_DT

    # Solve SCP
    runtime0_scp_DT = time.time()
    traj_scp_DT, J_vect_scp_DT, iter_scp_DT, feas_scp_DT = ocp_obstacle_avoidance(ff_model, states_ws_DT, actions_ws_DT, state_init, state_final)
    runtime1_scp_DT = time.time()
    runtime_scp_DT = runtime1_scp_DT - runtime0_scp_DT
    
    if np.char.equal(feas_scp_DT,'optimal'):
        # Save scp_DT in the output dictionary
        out['J_vect_scp_DT'] = J_vect_scp_DT
        out['iter_scp_DT'] = iter_scp_DT
        out['runtime_scp_DT'] = runtime_scp_DT
    else:
        out['feasible_DT'] = False

    # Import the Transformer_dag
    if np.char.equal(feas_cvx,'optimal') and mdp_constr:
        if transformer_ws == 'dyn':
            DT_dag_trajectory, runtime_DT_dag = DT_manager.torch_model_inference_dyn(model_dag, test_loader, test_sample, rtg_perc=None, ctg_perc=0., rtg=rtg_0, ctg_clipped=True)
        elif transformer_ws == 'ol':
            DT_dag_trajectory, runtime_DT_dag = DT_manager.torch_model_inference_ol(model_dag, test_loader, test_sample, rtg_perc=None, ctg_perc=0., rtg=rtg_0, ctg_clipped=True)
    else:
        if transformer_ws == 'dyn':
            DT_dag_trajectory, runtime_DT_dag = DT_manager.torch_model_inference_dyn(model_dag, test_loader, test_sample, rtg_perc=1., ctg_perc=0., rtg=None, ctg_clipped=True)
        elif transformer_ws == 'ol':
            DT_dag_trajectory, runtime_DT_dag = DT_manager.torch_model_inference_ol(model_dag, test_loader, test_sample, rtg_perc=1., ctg_perc=0., rtg=None, ctg_clipped=True)    
    out['J_dag'] = np.sum(la.norm(DT_dag_trajectory['dv_' + transformer_ws], ord=1, axis=0))
    states_ws_DT_dag = np.hstack((DT_dag_trajectory['xypsi_' + transformer_ws], state_final.reshape(-1,1))) # set warm start
    actions_ws_DT_dag = DT_dag_trajectory['dv_' + transformer_ws] # set warm start
    # Save DT in the output dictionary
    out['runtime_dag'] = runtime_DT_dag

    # Solve SCP
    runtime0_scp_dag = time.time()
    traj_scp_dag, J_vect_scp_dag, iter_scp_dag, feas_scp_dag = ocp_obstacle_avoidance(ff_model, states_ws_DT_dag, actions_ws_DT_dag, state_init, state_final)
    runtime1_scp_dag = time.time()
    runtime_scp_dag = runtime1_scp_dag - runtime0_scp_dag
    
    if np.char.equal(feas_scp_dag,'optimal'):
        # Save scp_DT in the output dictionary
        out['J_vect_scp_dag'] = J_vect_scp_dag
        out['iter_scp_dag'] = iter_scp_dag
        out['runtime_scp_dag'] = runtime_scp_dag
    else:
        out['feasible_scp_dag'] = False  

    return out

if __name__ == '__main__':

    transformer_ws = 'dyn' # 'dyn'/'ol'
    transformer_model_name = 'checkpoint_ff_ctgrtg_art'
    transformer_model_name_dag = 'checkpoint_ff_ctgrtg_art_cl_9'
    import_config = DT_manager.transformer_import_config(transformer_model_name)
    set_start_method('spawn')
    num_processes = 20

    # Get the datasets and loaders from the torch data
    datasets, dataloaders = DT_manager.get_train_val_test_data(mdp_constr=import_config['mdp_constr'], timestep_norm=import_config['timestep_norm'])
    train_loader, eval_loader, test_loader = dataloaders
    model = DT_manager.get_DT_model(transformer_model_name, train_loader, eval_loader)
    model_dag = DT_manager.get_DT_model(transformer_model_name_dag, train_loader, eval_loader)

    # Parallel for inputs
    N_data_test = test_loader.dataset.n_data
    other_args = {
        'model' : model,
        'model_dag' : model_dag,
        'test_loader' : test_loader,
        'transformer_ws' : transformer_ws,
        'mdp_constr' : import_config['mdp_constr'],
        'sample_init_final' : True
    }
    print('Sample_init_final =', other_args['sample_init_final'])

    J_vect_scp_cvx = np.empty(shape=(N_data_test, iter_max_SCP), dtype=float)
    J_vect_scp_dag = np.empty(shape=(N_data_test, iter_max_SCP), dtype=float)
    J_vect_scp_DT = np.empty(shape=(N_data_test, iter_max_SCP), dtype=float)
    J_cvx = np.empty(shape=(N_data_test, ), dtype=float)
    J_DT = np.empty(shape=(N_data_test, ), dtype=float)
    J_dag = np.empty(shape=(N_data_test, ), dtype=float)
    iter_scp_cvx = np.empty(shape=(N_data_test, ), dtype=float)
    iter_scp_dag = np.empty(shape=(N_data_test, ), dtype=float) 
    iter_scp_DT = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_cvx = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_DT = np.empty(shape=(N_data_test, ), dtype=float)
    runtime_dag = np.empty(shape=(N_data_test, ), dtype=float)  
    runtime_scp_cvx = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_scp_dag = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_scp_DT = np.empty(shape=(N_data_test, ), dtype=float) 
    ctgs0_cvx = np.empty(shape=(N_data_test, ), dtype=float)
    cvx_problem = np.full(shape=(N_data_test, ), fill_value=False)
    test_dataset_ix = np.empty(shape=(N_data_test, ), dtype=float)
    state_init = np.empty(shape=(N_data_test, N_STATE), dtype=float)
    state_final = np.empty(shape=(N_data_test, N_STATE), dtype=float)

    i_unfeas_cvx = []
    i_unfeas_scp_cvx = []
    i_unfeas_scp_dag = []
    i_unfeas_DT = []

    # Pool creation --> Should automatically select the maximum number of processes
    p = Pool(processes=num_processes)
    for i, res in enumerate(tqdm(p.imap(for_computation, zip(np.arange(N_data_test), itertools.repeat(other_args))), total=N_data_test)):
        # Save the input in the dataset
        test_dataset_ix[i] = res['test_dataset_ix']
        state_init[i] = res['state_init']
        state_final[i] = res['state_final']

        # If the solution is feasible save the optimization output
        if res['feasible_cvx']:
            J_cvx[i] = res['J_cvx']
            runtime_cvx[i] = res['runtime_cvx']
            ctgs0_cvx[i] = res['ctgs0_cvx']
            cvx_problem[i] = res['cvx_problem']
        else:
            i_unfeas_cvx += [ i ]

        if res['feasible_scp_cvx']:
            J_vect_scp_cvx[i,:] = res['J_vect_scp_cvx']
            iter_scp_cvx[i] = res['iter_scp_cvx']
            runtime_scp_cvx[i] = res['runtime_scp_cvx']
        else:
            i_unfeas_scp_cvx += [ i ]

        if res['feasible_DT']:
            J_DT[i] = res['J_DT']
            J_vect_scp_DT[i,:] = res['J_vect_scp_DT']
            iter_scp_DT[i] = res['iter_scp_DT']
            runtime_DT[i] = res['runtime_DT']
            runtime_scp_DT[i] = res['runtime_scp_DT']
        else:
            i_unfeas_DT += [ i ]
        
        if res['feasible_scp_dag']:
            J_dag[i] = res['J_dag']
            J_vect_scp_dag[i,:] = res['J_vect_scp_dag']
            iter_scp_dag[i] = res['iter_scp_dag']
            runtime_dag[i] = res['runtime_dag']
            runtime_scp_dag[i] = res['runtime_scp_dag']
        else:
            i_unfeas_scp_dag += [ i ]
        
        if i % 5000 == 0:
            #  Save dataset (local folder for the workstation)
            np.savez_compressed(root_folder + '/optimization/saved_files/warmstarting/ws_analysis_' + transformer_model_name + '_vs_dag_' + transformer_ws + '_test'+ str(i),
                                J_vect_scp_cvx = J_vect_scp_cvx,
                                J_vect_scp_dag = J_vect_scp_dag,
                                J_vect_scp_DT = J_vect_scp_DT,
                                J_cvx = J_cvx,
                                J_DT = J_DT,
                                J_dag = J_dag,
                                iter_scp_cvx = iter_scp_cvx,
                                iter_scp_dag = iter_scp_dag,
                                iter_scp_DT = iter_scp_DT,
                                runtime_cvx = runtime_cvx,
                                runtime_DT = runtime_DT,
                                runtime_dag = runtime_dag,
                                runtime_scp_cvx = runtime_scp_cvx,
                                runtime_scp_dag = runtime_scp_dag,
                                runtime_scp_DT = runtime_scp_DT,
                                ctgs0_cvx = ctgs0_cvx, 
                                cvx_problem = cvx_problem,
                                test_dataset_ix = test_dataset_ix,
                                state_init = state_init,
                                state_final = state_final,
                                i_unfeas_cvx = i_unfeas_cvx,
                                i_unfeas_scp_cvx = i_unfeas_scp_cvx,
                                i_unfeas_scp_dag = i_unfeas_scp_dag,
                                i_unfeas_DT = i_unfeas_DT
                                )

    
    #  Save dataset (local folder for the workstation)
    np.savez_compressed(root_folder + '/optimization/saved_files/warmstarting/ws_analysis_' + transformer_model_name + '_vs_dag_' + transformer_ws + '_test',
                        J_vect_scp_cvx = J_vect_scp_cvx,
                        J_vect_scp_dag = J_vect_scp_dag,
                        J_vect_scp_DT = J_vect_scp_DT,
                        J_cvx = J_cvx,
                        J_DT = J_DT,
                        J_dag = J_dag,
                        iter_scp_cvx = iter_scp_cvx,
                        iter_scp_dag = iter_scp_dag,
                        iter_scp_DT = iter_scp_DT,
                        runtime_cvx = runtime_cvx,
                        runtime_DT = runtime_DT,
                        runtime_dag = runtime_dag,
                        runtime_scp_cvx = runtime_scp_cvx,
                        runtime_scp_dag = runtime_scp_dag,
                        runtime_scp_DT = runtime_scp_DT,
                        ctgs0_cvx = ctgs0_cvx, 
                        cvx_problem = cvx_problem,
                        test_dataset_ix = test_dataset_ix,
                        state_init = state_init,
                        state_final = state_final,
                        i_unfeas_cvx = i_unfeas_cvx,
                        i_unfeas_scp_cvx = i_unfeas_scp_cvx,
                        i_unfeas_scp_dag = i_unfeas_scp_dag,
                        i_unfeas_DT = i_unfeas_DT
                        )