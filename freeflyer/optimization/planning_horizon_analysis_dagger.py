import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(root_folder)
#print(sys.path)

import numpy as np
import numpy.linalg as la

import decision_transformer.manage as ART_manager
import itertools
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from dynamics.freeflyer import sample_init_target, ocp_no_obstacle_avoidance, compute_constraint_to_go, FreeflyerModel
import torch
import time
from dynamics.FreeflyerEnv import FreeflyerEnv
import optimization.ff_scenario as ff
from decision_transformer.art_closed_loop import AutonomousFreeflyerTransformerMPC, ConvexMPC, MyopicConvexMPC

def for_computation(input_iterable):

    # Extract input
    current_idx = input_iterable[0]
    input_dict = input_iterable[1]
    models_names = input_dict['models_names']
    models = input_dict['models']
    test_loader = input_dict['test_loader']
    mdp_constr = input_dict['mdp_constr']
    mpc_steps = input_dict['mpc_steps']
    sample_init_final = input_dict['sample_init_final']

    # Output dictionary initialization
    out = {'test_dataset_ix' : [],
           'feasible_cvx' : True,
           'J_cvx' : [],
           'cvx_problem' : False,
           'feasible_cvxMPC' : True,
           'J_cvxMPC' : [],
           'J_com_cvxMPC' : [],
           'time_cvxMPC' : [],
           'feasible_myocvxMPC' : True,
           'J_myocvxMPC' : [],
           'J_com_myocvxMPC' : [],
           'time_myocvxMPC' : [],
           'feasible_artMPC' : True,
           'J_artMPC' : [],
           'J_com_artMPC' : [],
           'time_artMPC' : [],
           'feasible_artMPC_dag0' : True,
           'J_artMPC_dag0' : [],
           'J_com_artMPC_dag0' : [],
           'time_artMPC_dag0' : [],
           'feasible_artMPC_dag1' : True,
           'J_artMPC_dag1' : [],
           'J_com_artMPC_dag1' : [],
           'time_artMPC_dag1' : [],
           'feasible_artMPC_dag2' : True,
           'J_artMPC_dag2' : [],
           'J_com_artMPC_dag2' : [],
           'time_artMPC_dag2' : [],
           'feasible_artMPC_dag3' : True,
           'J_artMPC_dag3' : [],
           'J_com_artMPC_dag3' : [],
           'time_artMPC_dag3' : [],
           'feasible_artMPC_dag4' : True,
           'J_artMPC_dag4' : [],
           'J_com_artMPC_dag4' : [],
           'time_artMPC_dag4' : [],
           'feasible_artMPC_dag5' : True,
           'J_artMPC_dag5' : [],
           'J_com_artMPC_dag5' : [],
           'time_artMPC_dag5' : [],
           'feasible_artMPC_dag6' : True,
           'J_artMPC_dag6' : [],
           'J_com_artMPC_dag6' : [],
           'time_artMPC_dag6' : [],
           'feasible_artMPC_dag7' : True,
           'J_artMPC_dag7' : [],
           'J_com_artMPC_dag7' : [],
           'time_artMPC_dag7' : [],
           'feasible_artMPC_dag8' : True,
           'J_artMPC_dag8' : [],
           'J_com_artMPC_dag8' : [],
           'time_artMPC_dag8' : [],
           'feasible_artMPC_dag9' : True,
           'J_artMPC_dag9' : [],
           'J_com_artMPC_dag9' : [],
           'time_artMPC_dag9' : [],
           'ctgs0_cvx': []
          }
    
    test_sample = test_loader.dataset.getix(current_idx)
    data_stats = test_loader.dataset.data_stats
    if sample_init_final:
        state_init, state_final = sample_init_target()
        test_sample[0][0,:,:] = (torch.tensor(np.repeat(state_init[None,:], ff.n_time_rpod, axis=0)) - data_stats['states_mean'])/(data_stats['states_std'] + 1e-6)
        test_sample[1][0,:,:] = torch.zeros((ff.n_time_rpod, ff.N_ACTION))
        test_sample[2][0,:,0] = torch.zeros((ff.n_time_rpod,))
        test_sample[3][:,0] = torch.zeros((ff.n_time_rpod,))
        test_sample[4][0,:,:] = (torch.tensor(np.repeat(state_final[None,:], ff.n_time_rpod, axis=0)) - data_stats['goal_mean'])/(data_stats['goal_std'] + 1e-6)
        out['test_dataset_ix'] = test_sample[-1][0]
        dt = ff.dt
    else:
        if not mdp_constr:
            states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = test_sample
        else:
            states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = test_sample
        # print('Sampled trajectory ' + str(ix) + ' from test_dataset.')
        out['test_dataset_ix'] = ix[0]
        state_init = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
        state_final = np.array((goal_i[0, 0, :] * data_stats['goal_std'][0]) + data_stats['goal_mean'][0])
        dt = dt.item()

    out['state_init'] = state_init
    out['state_final'] = state_final
    ff_model = FreeflyerModel()

    ####### Warmstart Convex Problem QUAD
    try:
        traj_cvx, _, _, feas_cvx = ocp_no_obstacle_avoidance(ff_model, state_init, state_final)
        states_cvx, actions_cvx = traj_cvx['states'], traj_cvx['actions_G']
    except:
        states_cvx = None
        actions_cvx = None
        feas_cvx = 'infeasible'
    
    if not np.char.equal(feas_cvx,'infeasible'):
        out['J_cvx'] = np.sum(la.norm(actions_cvx, ord=1, axis=0))
        rtg_0 = -out['J_cvx']
        # Evaluate Constraint Violation
        ctgs_cvx = compute_constraint_to_go(states_cvx.T, ff.obs['position'], (ff.obs['radius'] + ff.robot_radius)*ff.safety_margin)
        ctgs0_cvx = ctgs_cvx[0,0]
        # Save cvx in the output dictionary
        out['ctgs0_cvx'] = ctgs0_cvx
        out['cvx_problem'] = ctgs0_cvx == 0

        if not out['cvx_problem']:
            ############## MPC methods
            # Create and initialize the environments and the MCPCs
            env_cvxMPC = FreeflyerEnv()
            env_cvxMPC.reset('det', reset_condition=(dt, state_init, state_final))
            cvxMPC = ConvexMPC(mpc_steps, scp_mode='soft')
            # For cycle one by one to isolate failures: Type error means the optimizer failed and results are None -> TypeError
            try:
                time_cvxMPC = np.empty((ff.n_time_rpod,))
                for i in np.arange(ff.n_time_rpod):         
                    # CVX-ws
                    current_obs_cvx = env_cvxMPC.get_observation()
                    tic = time.time()
                    cvx_traj = cvxMPC.warmstart(current_obs_cvx, env_cvxMPC)
                    cvxMPC_traj, cvxMPC_scp_dict = cvxMPC.solve_scp(current_obs_cvx, env_cvxMPC, cvx_traj['state'], cvx_traj['dv'])
                    time_cvxMPC[i] = time.time() - tic
                    #env_cvxMPC.load_prediction(cvx_traj, cvxMPC_traj)
                    _ = env_cvxMPC.step(cvxMPC_traj['dv'][:,0])
                out['J_cvxMPC'] = np.sum(la.norm(env_cvxMPC.dv, ord=1, axis=0))
                out['J_com_cvxMPC'] = np.sum(la.norm(env_cvxMPC.dv, ord=1, axis=0))
                out['time_cvxMPC'] = time_cvxMPC
            except:
                out['feasible_cvxMPC'] = False
            
            env_myocvxMPC = FreeflyerEnv()
            env_myocvxMPC.reset('det', reset_condition=(dt, state_init, state_final))
            myocvxMPC = MyopicConvexMPC(mpc_steps, scp_mode='soft')
            # For cycle one by one to isolate failures: Type error means the optimizer failed and results are None -> TypeError
            try:
                time_myocvxMPC = np.empty((ff.n_time_rpod,))
                for i in np.arange(ff.n_time_rpod):    
                    # CVX-ws
                    current_obs_myocvx = env_myocvxMPC.get_observation()
                    tic = time.time()
                    myocvx_traj = myocvxMPC.warmstart(current_obs_myocvx, env_myocvxMPC)
                    myocvxMPC_traj, myocvxMPC_scp_dict = myocvxMPC.solve_scp(current_obs_myocvx, env_myocvxMPC, myocvx_traj['state'], myocvx_traj['dv'])
                    time_myocvxMPC[i] = time.time() - tic
                    #env_cvxMPC.load_prediction(cvx_traj, cvxMPC_traj)
                    _ = env_myocvxMPC.step(myocvxMPC_traj['dv'][:,0])
                out['J_myocvxMPC'] = np.sum(la.norm(env_myocvxMPC.dv, ord=1, axis=0))
                out['J_com_myocvxMPC'] = np.sum(la.norm(env_myocvxMPC.dv, ord=1, axis=0))
                out['time_myocvxMPC'] = time_myocvxMPC
            except:
                out['feasible_myocvxMPC'] = False

            env_artMPC = FreeflyerEnv()
            env_artMPC.reset('det', reset_condition=(dt, state_init, state_final))
            artMPC = AutonomousFreeflyerTransformerMPC(models[0], test_loader, mpc_steps, transformer_mode='dyn', ctg_clipped=True, scp_mode='soft')
            try:
                time_artMPC = np.empty((ff.n_time_rpod,))
                for i in np.arange(ff.n_time_rpod):
                    # ART-ws
                    current_obs_art = env_artMPC.get_observation()
                    tic = time.time()
                    if mdp_constr:
                        art_traj = artMPC.warmstart(current_obs_art, env_artMPC, rtg0=rtg_0, ctg0=0)
                    else:
                        art_traj = artMPC.warmstart(current_obs_art, env_artMPC, rtgs_i=rtgs_i)
                    artMPC_traj, artMPC_scp_dict = artMPC.solve_scp(current_obs_art, env_artMPC, art_traj['state'], art_traj['dv'])
                    time_artMPC[i] = time.time() - tic
                    #env_artMPC.load_prediction(art_traj, artMPC_traj)
                    _ = env_artMPC.step(artMPC_traj['dv'][:,0])
                out['J_artMPC'] = np.sum(la.norm(env_artMPC.dv, ord=1, axis=0))
                out['J_com_artMPC'] = np.sum(la.norm(env_artMPC.dv, ord=1, axis=0))
                out['time_artMPC'] = time_artMPC
            except:
                out['feasible_artMPC'] = False
            
            for n_model in np.arange(len(models)-1):
                env_artMPC = FreeflyerEnv()
                env_artMPC.reset('det', reset_condition=(dt, state_init, state_final))        
                artMPC = AutonomousFreeflyerTransformerMPC(models[n_model+1], test_loader, mpc_steps, transformer_mode='dyn', ctg_clipped=True, scp_mode='soft')
                try:
                    time_artMPC = np.empty((ff.n_time_rpod,))
                    for i in np.arange(ff.n_time_rpod):
                        # ART-ws
                        current_obs_art = env_artMPC.get_observation()
                        tic = time.time()
                        if mdp_constr:
                            art_traj = artMPC.warmstart(current_obs_art, env_artMPC, rtg0=rtg_0, ctg0=0)
                        else:
                            art_traj = artMPC.warmstart(current_obs_art, env_artMPC, rtgs_i=rtgs_i)
                        artMPC_traj, artMPC_scp_dict = artMPC.solve_scp(current_obs_art, env_artMPC, art_traj['state'], art_traj['dv'])
                        time_artMPC[i] = time.time() - tic
                        #env_artMPC.load_prediction(art_traj, artMPC_traj)
                        _ = env_artMPC.step(artMPC_traj['dv'][:,0])
                    out['J_artMPC_dag'+str(n_model)] = np.sum(la.norm(env_artMPC.dv, ord=1, axis=0))
                    out['J_com_artMPC_dag'+str(n_model)] = np.sum(la.norm(env_artMPC.dv, ord=1, axis=0))
                    out['time_artMPC_dag'+str(n_model)] = time_artMPC
                except:
                    out['feasible_artMPC_dag'+str(n_model)] = False
        
    else:
        out['feasible_cvx'] = False
        out['feasible_cvxMPC'] = False
        out['feasible_myocvxMPC'] = False
        out['feasible_artMPC'] = False
        out['feasible_artMPC_dag0'] = False
        out['feasible_artMPC_dag1'] = False
        out['feasible_artMPC_dag2'] = False
        out['feasible_artMPC_dag3'] = False
        out['feasible_artMPC_dag4'] = False
        out['feasible_artMPC_dag5'] = False
        out['feasible_artMPC_dag6'] = False
        out['feasible_artMPC_dag7'] = False
        out['feasible_artMPC_dag8'] = False
        out['feasible_artMPC_dag9'] = False

    return out

if __name__ == '__main__':

    model_name_import = 'checkpoint_ff_ctgrtg_art'
    import_config = ART_manager.transformer_import_config(model_name_import)
    mdp_constr = import_config['mdp_constr']
    timestep_norm = import_config['timestep_norm']
    transformer_model_names = ['checkpoint_ff_ctgrtg_art',
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               'checkpoint_ff_ctgrtg_art_cl_8',
                               'checkpoint_ff_ctgrtg_art_cl_9',
                               None
                               ]
    set_start_method('spawn')
    num_processes = 24

    # Get the datasets and loaders from the torch data
    _, dataloaders = ART_manager.get_train_val_test_data(mdp_constr, timestep_norm)
    train_loader, eval_loader, test_loader = dataloaders
    models = []
    for name in transformer_model_names:
        if name == None:
            models.append(None)
        else:
            models.append(ART_manager.get_DT_model(name, train_loader, eval_loader))
    
    # Select the indexes to have uniform 
    indexes = np.arange(test_loader.dataset.n_data)
    ctgs_indexes = {
        '1-10' : indexes[(test_loader.dataset.data['ctgs'][:,0] >= 1) & (test_loader.dataset.data['ctgs'][:,0] <= 10)],
        '11-20' : indexes[(test_loader.dataset.data['ctgs'][:,0] >= 11) & (test_loader.dataset.data['ctgs'][:,0] <= 20)],
        '21-30' : indexes[(test_loader.dataset.data['ctgs'][:,0] >= 21) & (test_loader.dataset.data['ctgs'][:,0] <= 30)],
        '31-40' : indexes[(test_loader.dataset.data['ctgs'][:,0] >= 31) & (test_loader.dataset.data['ctgs'][:,0] <= 40)],
        '41-50' : indexes[(test_loader.dataset.data['ctgs'][:,0] >= 41) & (test_loader.dataset.data['ctgs'][:,0] <= 50)]
    }
    n_samples_x_ctg_interval = 100
    uniform_idx = np.array([],dtype=int)
    for key in ctgs_indexes.keys():
        uniform_idx = np.concatenate((uniform_idx, ctgs_indexes[key][:n_samples_x_ctg_interval]))
    N_data_test = len(uniform_idx)
    
    # Loop through planning horizons
    mpc_steps_analysis = np.arange(10,101,10)
    for n_steps in mpc_steps_analysis:
        # Parallel for inputs
        other_args = {
            'models_names' : transformer_model_names,
            'models' : models,
            'test_loader' : test_loader,
            'mdp_constr' : mdp_constr,
            'mpc_steps' : n_steps,
            'sample_init_final' : False
        }

        J_cvx = np.empty(shape=(N_data_test, ), dtype=float)
        J_cvxMPC = np.empty(shape=(N_data_test, ), dtype=float)
        J_myocvxMPC = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC_dag0 = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC_dag1 = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC_dag2 = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC_dag3 = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC_dag4 = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC_dag5 = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC_dag6 = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC_dag7 = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC_dag8 = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC_dag9 = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_cvx = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_cvxMPC = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_myocvxMPC = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_artMPC = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_artMPC_dag0 = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_artMPC_dag1 = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_artMPC_dag2 = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_artMPC_dag3 = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_artMPC_dag4 = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_artMPC_dag5 = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_artMPC_dag6 = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_artMPC_dag7 = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_artMPC_dag8 = np.empty(shape=(N_data_test, ), dtype=float)
        J_com_artMPC_dag9 = np.empty(shape=(N_data_test, ), dtype=float)
        time_cvxMPC = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_myocvxMPC = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_artMPC = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_artMPC_dag0 = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_artMPC_dag1 = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_artMPC_dag2 = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_artMPC_dag3 = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_artMPC_dag4 = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_artMPC_dag5 = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_artMPC_dag6 = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_artMPC_dag7 = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_artMPC_dag8 = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        time_artMPC_dag9 = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        ctgs0_cvx = np.empty(shape=(N_data_test, ), dtype=float)
        cvx_problem = np.full(shape=(N_data_test, ), fill_value=False)
        test_dataset_ix = -np.ones(shape=(N_data_test, ), dtype=int)

        i_unfeas_cvx = []
        i_unfeas_cvxMPC = []
        i_unfeas_myocvxMPC = []
        i_unfeas_artMPC = []
        i_unfeas_artMPC_dag0 = []
        i_unfeas_artMPC_dag1 = []
        i_unfeas_artMPC_dag2 = []
        i_unfeas_artMPC_dag3 = []
        i_unfeas_artMPC_dag4 = []
        i_unfeas_artMPC_dag5 = []
        i_unfeas_artMPC_dag6 = []
        i_unfeas_artMPC_dag7 = []
        i_unfeas_artMPC_dag8 = []
        i_unfeas_artMPC_dag9 = []
        
        # Pool creation --> Should automatically select the maximum number of processes
        p = Pool(processes=num_processes)
        for i, res in enumerate(tqdm(p.imap(for_computation, zip(uniform_idx, itertools.repeat(other_args))), total=N_data_test)):
            '''for i in tqdm(range(len(uniform_idx)), total=len(uniform_idx)):
            # Save the input in the dataset
            res = for_computation((uniform_idx[i], other_args))'''
            test_dataset_ix[i] = res['test_dataset_ix']

            # If the solution is feasible save the optimization output
            if res['feasible_cvx']:
                J_cvx[i] = res['J_cvx']
                ctgs0_cvx[i] = res['ctgs0_cvx']
                cvx_problem[i] = res['cvx_problem']
            else:
                i_unfeas_cvx += [ i ]
            
            # Save the close loop results only if the problem in non-convex
            if not res['cvx_problem']:
                if res['feasible_cvxMPC']:
                    J_cvxMPC[i] = res['J_cvxMPC']
                    J_com_cvxMPC[i] = res['J_com_cvxMPC']
                    time_cvxMPC[i,:] = res['time_cvxMPC']
                else:
                    i_unfeas_cvxMPC += [ i ]
                
                if res['feasible_myocvxMPC']:
                    J_myocvxMPC[i] = res['J_myocvxMPC']
                    J_com_myocvxMPC[i] = res['J_com_myocvxMPC']
                    time_myocvxMPC[i,:] = res['time_myocvxMPC']
                else:
                    i_unfeas_myocvxMPC += [ i ]
                
                if res['feasible_artMPC']:
                    J_artMPC[i] = res['J_artMPC']
                    J_com_artMPC[i] = res['J_com_artMPC']
                    time_artMPC[i,:] = res['time_artMPC']
                else:
                    i_unfeas_artMPC += [ i ]
                
                if res['feasible_artMPC_dag0']:
                    J_artMPC_dag0[i] = res['J_artMPC_dag0']
                    J_com_artMPC_dag0[i] = res['J_com_artMPC_dag0']
                    time_artMPC_dag0[i,:] = res['time_artMPC_dag0']
                else:
                    i_unfeas_artMPC_dag0 += [ i ]
                
                if res['feasible_artMPC_dag1']:
                    J_artMPC_dag1[i] = res['J_artMPC_dag1']
                    J_com_artMPC_dag1[i] = res['J_com_artMPC_dag1']
                    time_artMPC_dag1[i,:] = res['time_artMPC_dag1']
                else:
                    i_unfeas_artMPC_dag1 += [ i ]
                
                if res['feasible_artMPC_dag2']:
                    J_artMPC_dag2[i] = res['J_artMPC_dag2']
                    J_com_artMPC_dag2[i] = res['J_com_artMPC_dag2']
                    time_artMPC_dag2[i,:] = res['time_artMPC_dag2']
                else:
                    i_unfeas_artMPC_dag2 += [ i ]
                
                if res['feasible_artMPC_dag3']:
                    J_artMPC_dag3[i] = res['J_artMPC_dag3']
                    J_com_artMPC_dag3[i] = res['J_com_artMPC_dag3']
                    time_artMPC_dag3[i,:] = res['time_artMPC_dag3']
                else:
                    i_unfeas_artMPC_dag3 += [ i ]
                        
                if res['feasible_artMPC_dag4']:
                    J_artMPC_dag4[i] = res['J_artMPC_dag4']
                    J_com_artMPC_dag4[i] = res['J_com_artMPC_dag4']
                    time_artMPC_dag4[i,:] = res['time_artMPC_dag4']
                else:
                    i_unfeas_artMPC_dag4 += [ i ]
                
                if res['feasible_artMPC_dag5']:
                    J_artMPC_dag5[i] = res['J_artMPC_dag5']
                    J_com_artMPC_dag5[i] = res['J_com_artMPC_dag5']
                    time_artMPC_dag5[i,:] = res['time_artMPC_dag5']
                else:
                    i_unfeas_artMPC_dag5 += [ i ]
                
                if res['feasible_artMPC_dag6']:
                    J_artMPC_dag6[i] = res['J_artMPC_dag6']
                    J_com_artMPC_dag6[i] = res['J_com_artMPC_dag6']
                    time_artMPC_dag6[i,:] = res['time_artMPC_dag6']
                else:
                    i_unfeas_artMPC_dag6 += [ i ]
                
                if res['feasible_artMPC_dag7']:
                    J_artMPC_dag7[i] = res['J_artMPC_dag7']
                    J_com_artMPC_dag7[i] = res['J_com_artMPC_dag7']
                    time_artMPC_dag7[i,:] = res['time_artMPC_dag7']
                else:
                    i_unfeas_artMPC_dag7 += [ i ]
                
                if res['feasible_artMPC_dag8']:
                    J_artMPC_dag8[i] = res['J_artMPC_dag8']
                    J_com_artMPC_dag8[i] = res['J_com_artMPC_dag8']
                    time_artMPC_dag8[i,:] = res['time_artMPC_dag8']
                else:
                    i_unfeas_artMPC_dag8 += [ i ]
                
                if res['feasible_artMPC_dag9']:
                    J_artMPC_dag9[i] = res['J_artMPC_dag9']
                    J_com_artMPC_dag9[i] = res['J_com_artMPC_dag9']
                    time_artMPC_dag9[i,:] = res['time_artMPC_dag9']
                else:
                    i_unfeas_artMPC_dag9 += [ i ]
            
            # Periodically save the results
            if i%200 == 0:
                np.savez_compressed(root_folder + '/optimization/saved_files/closed_loop/ws_analysis_DAGGER' + str(other_args['mpc_steps']) + '_' + str(i),
                            J_cvx = J_cvx,
                            J_cvxMPC = J_cvxMPC,
                            J_myocvxMPC = J_myocvxMPC,
                            J_artMPC = J_artMPC,
                            J_artMPC_dag0 = J_artMPC_dag0,
                            J_artMPC_dag1 = J_artMPC_dag1,
                            J_artMPC_dag2 = J_artMPC_dag2,
                            J_artMPC_dag3 = J_artMPC_dag3,
                            J_artMPC_dag4 = J_artMPC_dag4,
                            J_artMPC_dag5 = J_artMPC_dag5,
                            J_artMPC_dag6 = J_artMPC_dag6,
                            J_artMPC_dag7 = J_artMPC_dag7,
                            J_artMPC_dag8 = J_artMPC_dag8,
                            J_artMPC_dag9 = J_artMPC_dag9,
                            J_com_cvxMPC = J_com_cvxMPC,
                            J_com_myocvxMPC = J_com_myocvxMPC,
                            J_com_artMPC = J_com_artMPC,
                            J_com_artMPC_dag0 = J_com_artMPC_dag0,
                            J_com_artMPC_dag1 = J_com_artMPC_dag1,
                            J_com_artMPC_dag2 = J_com_artMPC_dag2,
                            J_com_artMPC_dag3 = J_com_artMPC_dag3,
                            J_com_artMPC_dag4 = J_com_artMPC_dag4,
                            J_com_artMPC_dag5 = J_com_artMPC_dag5,
                            J_com_artMPC_dag6 = J_com_artMPC_dag6,
                            J_com_artMPC_dag7 = J_com_artMPC_dag7,
                            J_com_artMPC_dag8 = J_com_artMPC_dag8,
                            J_com_artMPC_dag9 = J_com_artMPC_dag9,
                            time_cvxMPC = time_cvxMPC,
                            time_myocvxMPC = time_myocvxMPC,
                            time_artMPC = time_artMPC,
                            time_artMPC_dag0 = time_artMPC_dag0,
                            time_artMPC_dag1 = time_artMPC_dag1,
                            time_artMPC_dag2 = time_artMPC_dag2,
                            time_artMPC_dag3 = time_artMPC_dag3,
                            time_artMPC_dag4 = time_artMPC_dag4,
                            time_artMPC_dag5 = time_artMPC_dag5,
                            time_artMPC_dag6 = time_artMPC_dag6,
                            time_artMPC_dag7 = time_artMPC_dag7,
                            time_artMPC_dag8 = time_artMPC_dag8,
                            time_artMPC_dag9 = time_artMPC_dag9,
                            ctgs0_cvx = ctgs0_cvx, 
                            cvx_problem = cvx_problem,
                            test_dataset_ix = test_dataset_ix,
                            i_unfeas_cvx = i_unfeas_cvx,
                            i_unfeas_cvxMPC = i_unfeas_cvxMPC,
                            i_unfeas_myocvxMPC = i_unfeas_myocvxMPC,
                            i_unfeas_artMPC = i_unfeas_artMPC,
                            i_unfeas_artMPC_dag0 = i_unfeas_artMPC_dag0,
                            i_unfeas_artMPC_dag1 = i_unfeas_artMPC_dag1,
                            i_unfeas_artMPC_dag2 = i_unfeas_artMPC_dag2,
                            i_unfeas_artMPC_dag3 = i_unfeas_artMPC_dag3,
                            i_unfeas_artMPC_dag4 = i_unfeas_artMPC_dag4,
                            i_unfeas_artMPC_dag5 = i_unfeas_artMPC_dag5,
                            i_unfeas_artMPC_dag6 = i_unfeas_artMPC_dag6,
                            i_unfeas_artMPC_dag7 = i_unfeas_artMPC_dag7,
                            i_unfeas_artMPC_dag8 = i_unfeas_artMPC_dag8,
                            i_unfeas_artMPC_dag9 = i_unfeas_artMPC_dag9
                            )

        #  Save dataset (local folder for the workstation)
        np.savez_compressed(root_folder + '/optimization/saved_files/closed_loop/ws_analysis_DAGGER' + str(other_args['mpc_steps']),
                        J_cvx = J_cvx,
                        J_cvxMPC = J_cvxMPC,
                        J_myocvxMPC = J_myocvxMPC,
                        J_artMPC = J_artMPC,
                        J_artMPC_dag0 = J_artMPC_dag0,
                        J_artMPC_dag1 = J_artMPC_dag1,
                        J_artMPC_dag2 = J_artMPC_dag2,
                        J_artMPC_dag3 = J_artMPC_dag3,
                        J_artMPC_dag4 = J_artMPC_dag4,
                        J_artMPC_dag5 = J_artMPC_dag5,
                        J_artMPC_dag6 = J_artMPC_dag6,
                        J_artMPC_dag7 = J_artMPC_dag7,
                        J_artMPC_dag8 = J_artMPC_dag8,
                        J_artMPC_dag9 = J_artMPC_dag9,
                        J_com_cvxMPC = J_com_cvxMPC,
                        J_com_myocvxMPC = J_com_myocvxMPC,
                        J_com_artMPC = J_com_artMPC,
                        J_com_artMPC_dag0 = J_com_artMPC_dag0,
                        J_com_artMPC_dag1 = J_com_artMPC_dag1,
                        J_com_artMPC_dag2 = J_com_artMPC_dag2,
                        J_com_artMPC_dag3 = J_com_artMPC_dag3,
                        J_com_artMPC_dag4 = J_com_artMPC_dag4,
                        J_com_artMPC_dag5 = J_com_artMPC_dag5,
                        J_com_artMPC_dag6 = J_com_artMPC_dag6,
                        J_com_artMPC_dag7 = J_com_artMPC_dag7,
                        J_com_artMPC_dag8 = J_com_artMPC_dag8,
                        J_com_artMPC_dag9 = J_com_artMPC_dag9,
                        time_cvxMPC = time_cvxMPC,
                        time_myocvxMPC = time_myocvxMPC,
                        time_artMPC = time_artMPC,
                        time_artMPC_dag0 = time_artMPC_dag0,
                        time_artMPC_dag1 = time_artMPC_dag1,
                        time_artMPC_dag2 = time_artMPC_dag2,
                        time_artMPC_dag3 = time_artMPC_dag3,
                        time_artMPC_dag4 = time_artMPC_dag4,
                        time_artMPC_dag5 = time_artMPC_dag5,
                        time_artMPC_dag6 = time_artMPC_dag6,
                        time_artMPC_dag7 = time_artMPC_dag7,
                        time_artMPC_dag8 = time_artMPC_dag8,
                        time_artMPC_dag9 = time_artMPC_dag9,
                        ctgs0_cvx = ctgs0_cvx, 
                        cvx_problem = cvx_problem,
                        test_dataset_ix = test_dataset_ix,
                        i_unfeas_cvx = i_unfeas_cvx,
                        i_unfeas_cvxMPC = i_unfeas_cvxMPC,
                        i_unfeas_myocvxMPC = i_unfeas_myocvxMPC,
                        i_unfeas_artMPC = i_unfeas_artMPC,
                        i_unfeas_artMPC_dag0 = i_unfeas_artMPC_dag0,
                        i_unfeas_artMPC_dag1 = i_unfeas_artMPC_dag1,
                        i_unfeas_artMPC_dag2 = i_unfeas_artMPC_dag2,
                        i_unfeas_artMPC_dag3 = i_unfeas_artMPC_dag3,
                        i_unfeas_artMPC_dag4 = i_unfeas_artMPC_dag4,
                        i_unfeas_artMPC_dag5 = i_unfeas_artMPC_dag5,
                        i_unfeas_artMPC_dag6 = i_unfeas_artMPC_dag6,
                        i_unfeas_artMPC_dag7 = i_unfeas_artMPC_dag7,
                        i_unfeas_artMPC_dag8 = i_unfeas_artMPC_dag8,
                        i_unfeas_artMPC_dag9 = i_unfeas_artMPC_dag9
                        )