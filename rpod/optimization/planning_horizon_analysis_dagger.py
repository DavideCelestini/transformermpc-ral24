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
from dynamics.RpodEnv import RpodEnv
import dynamics.orbit_dynamics as dyn
import rpod_scenario as rpod
import ocp as ocp
import time
from decision_transformer.art_closed_loop import AutonomousRendezvousTransformerMPC, ConvexMPC, MyopicConvexMPC

def for_computation(input_iterable):

    # Extract input
    current_idx = input_iterable[0]
    input_dict = input_iterable[1]
    models_names = input_dict['models_names']
    models = input_dict['models']
    test_loader = input_dict['test_loader']
    state_representation = input_dict['state_representation']
    mdp_constr = input_dict['mdp_constr']
    mpc_steps = input_dict['mpc_steps']

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
    if not mdp_constr:
        states_i, _, rtgs_i, goal_i, _, _, oe, _, _, horizons, ix = test_sample
    else:
        states_i, _, rtgs_i, _, goal_i, _, _, oe, _, _, horizons, ix = test_sample
    out['test_dataset_ix'] = ix[0]

    hrz = horizons.item()
    if state_representation == 'roe':
        state_roe_0 = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
    elif state_representation == 'rtn':
        state_rtn_0 = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
        state_roe_0 = dyn.map_rtn_to_roe(state_rtn_0, np.array(oe[0, :, 0]))
    dock_param, _ = rpod.dock_param_maker(np.array((goal_i[0, 0, :] * data_stats['goal_std'][0]) + data_stats['goal_mean'][0]))

    # Dynamics Matrices Precomputations
    stm_hrz, cim_hrz, psi_hrz, oe_hrz, time_hrz, dt_hrz = dyn.dynamics_roe_optimization(rpod.oe_0_ref, rpod.t_0, hrz, rpod.n_time_rpod)

    ####### Warmstart Convex Problem RPOD
    try:
        states_roe_cvx, actions_cvx, feas_cvx = ocp.ocp_cvx(stm_hrz, cim_hrz, psi_hrz, state_roe_0, dock_param, rpod.n_time_rpod)
    except:
        states_roe_cvx = None
        actions_cvx = None
        feas_cvx = 'infeasible'
    
    if not np.char.equal(feas_cvx,'infeasible'):
        out['J_cvx'] = sum(la.norm(actions_cvx,axis=0))
        rtg_0 = -out['J_cvx']
        states_rtn_ws_cvx = dyn.roe_to_rtn_horizon(states_roe_cvx, oe_hrz, rpod.n_time_rpod)
        # Evaluate Constraint Violation
        ctgs_cvx = ocp.compute_constraint_to_go(states_rtn_ws_cvx.T[None,:,:], 1, rpod.n_time_rpod)
        ctgs0_cvx = ctgs_cvx[0,0]
        # Save cvx in the output dictionary
        out['ctgs0_cvx'] = ctgs0_cvx
        out['cvx_problem'] = ctgs0_cvx == 0

        if not out['cvx_problem']:
            ############## MPC methods
            # Create and initialize the environments and the MCPCs
            env_cvxMPC = RpodEnv()
            env_cvxMPC.reset('det', reset_condition=(hrz, state_roe_0, dock_param))
            cvxMPC = ConvexMPC(mpc_steps, scp_mode='soft')
            # For cycle one by one to isolate failures: Type error means the optimizer failed and results are None -> TypeError
            try: 
                time_cvxMPC = np.empty((rpod.n_time_rpod,))
                for i in np.arange(rpod.n_time_rpod):        
                    # CVX-ws
                    current_obs_cvx = env_cvxMPC.get_observation()
                    tic = time.time()
                    cvx_traj, stm, cim, psi = cvxMPC.warmstart(current_obs_cvx, env_cvxMPC, return_dynamics=True)
                    cvxMPC_traj, cvxMPC_scp_dict = cvxMPC.solve_scp(current_obs_cvx, env_cvxMPC, stm, cim, psi, cvx_traj['state_roe'], cvx_traj['dv_rtn'])
                    time_cvxMPC[i] = time.time() - tic
                    #env_cvxMPC.load_prediction(cvx_traj, cvxMPC_traj)
                    _ = env_cvxMPC.step(cvxMPC_traj['dv_rtn'][:,0],'rtn')
                out['J_cvxMPC'] = la.norm(env_cvxMPC.dv_rtn, axis=0).sum()
                out['J_com_cvxMPC'] = la.norm(env_cvxMPC.dv_com_rtn, axis=0).sum()
                out['time_cvxMPC'] = time_cvxMPC
            except:
                out['feasible_cvxMPC'] = False
            
            env_myocvxMPC = RpodEnv()
            env_myocvxMPC.reset('det', reset_condition=(hrz, state_roe_0, dock_param))
            myocvxMPC = MyopicConvexMPC(mpc_steps, scp_mode='soft')
            # For cycle one by one to isolate failures: Type error means the optimizer failed and results are None -> TypeError
            try: 
                time_myocvxMPC = np.empty((rpod.n_time_rpod,))
                for i in np.arange(rpod.n_time_rpod):        
                    # CVX-ws
                    current_obs_myocvx = env_myocvxMPC.get_observation()
                    tic = time.time()
                    myocvx_traj, stm, cim, psi = myocvxMPC.warmstart(current_obs_myocvx, env_myocvxMPC, return_dynamics=True)
                    myocvxMPC_traj, myocvxMPC_scp_dict = myocvxMPC.solve_scp(current_obs_myocvx, env_myocvxMPC, stm, cim, psi, myocvx_traj['state_roe'], myocvx_traj['dv_rtn'])
                    time_myocvxMPC[i] = time.time() - tic
                    #env_cvxMPC.load_prediction(cvx_traj, cvxMPC_traj)
                    _ = env_myocvxMPC.step(myocvxMPC_traj['dv_rtn'][:,0],'rtn')
                out['J_myocvxMPC'] = la.norm(env_myocvxMPC.dv_rtn, axis=0).sum()
                out['J_com_myocvxMPC'] = la.norm(env_myocvxMPC.dv_com_rtn, axis=0).sum()
                out['time_myocvxMPC'] = time_myocvxMPC
            except:
                out['feasible_myocvxMPC'] = False
            
            env_artMPC = RpodEnv()
            env_artMPC.reset('det', reset_condition=(hrz, state_roe_0, dock_param))
            artMPC = AutonomousRendezvousTransformerMPC(models[0], test_loader, mpc_steps, transformer_mode='dyn', ctg_clipped=True, scp_mode='soft')
            try:
                time_artMPC = np.empty((rpod.n_time_rpod,))
                for i in np.arange(rpod.n_time_rpod):
                    # ART-ws
                    current_obs_art = env_artMPC.get_observation()
                    tic = time.time()
                    if mdp_constr:
                        art_traj, stm, cim, psi = artMPC.warmstart(current_obs_art, env_artMPC, rtg0=rtg_0, ctg0=0, return_dynamics=True)
                    else:
                        art_traj, stm, cim, psi = artMPC.warmstart(current_obs_art, env_artMPC, rtgs_i=rtgs_i, return_dynamics=True)
                    artMPC_traj, artMPC_scp_dict = artMPC.solve_scp(current_obs_art, env_artMPC, stm, cim, psi, art_traj['state_roe'], art_traj['dv_rtn'])
                    time_artMPC[i] = time.time() - tic
                    #env_artMPC.load_prediction(art_traj, artMPC_traj)
                    _ = env_artMPC.step(artMPC_traj['dv_rtn'][:,0],'rtn')
                out['J_artMPC'] = la.norm(env_artMPC.dv_rtn, axis=0).sum()
                out['J_com_artMPC'] = la.norm(env_artMPC.dv_com_rtn, axis=0).sum()
                out['time_artMPC'] = time_artMPC
            except:
                out['feasible_artMPC'] = False
            
            for n_model in np.arange(len(models)-1):
                env_artMPC = RpodEnv()
                env_artMPC.reset('det', reset_condition=(hrz, state_roe_0, dock_param))        
                artMPC = AutonomousRendezvousTransformerMPC(models[n_model+1], test_loader, mpc_steps, transformer_mode='dyn', ctg_clipped=True, scp_mode='soft')
                try:
                    time_artMPC = np.empty((rpod.n_time_rpod,))
                    for i in np.arange(rpod.n_time_rpod):
                        # ART-ws
                        current_obs_art = env_artMPC.get_observation()
                        tic = time.time()
                        if mdp_constr:
                            art_traj, stm, cim, psi = artMPC.warmstart(current_obs_art, env_artMPC, rtg0=rtg_0, ctg0=0, return_dynamics=True)
                        else:
                            art_traj, stm, cim, psi = artMPC.warmstart(current_obs_art, env_artMPC, rtgs_i=rtgs_i, return_dynamics=True)
                        artMPC_traj, artMPC_scp_dict = artMPC.solve_scp(current_obs_art, env_artMPC, stm, cim, psi, art_traj['state_roe'], art_traj['dv_rtn'])
                        time_artMPC[i] = time.time() - tic
                        #env_artMPC.load_prediction(art_traj, artMPC_traj)
                        _ = env_artMPC.step(artMPC_traj['dv_rtn'][:,0],'rtn')
                    out['J_artMPC_dag'+str(n_model)] = la.norm(env_artMPC.dv_rtn, axis=0).sum()
                    out['J_com_artMPC_dag'+str(n_model)] = la.norm(env_artMPC.dv_com_rtn, axis=0).sum()
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

    model_name_import = 'checkpoint_rtn_ctgrtg'
    import_config = ART_manager.transformer_import_config(model_name_import)
    mdp_constr = import_config['mdp_constr']
    timestep_norm = import_config['timestep_norm']
    state_representation = import_config['state_representation']
    dataset_to_use = import_config['dataset_to_use']
    transformer_model_names = ['checkpoint_rtn_ctgrtg',
                               'checkpoint_rtn_ctgrtg_cl_10',
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None
                               ]
    set_start_method('spawn')
    num_processes = 20

    # Get the datasets and loaders from the torch data
    _, dataloaders = ART_manager.get_train_val_test_data(state_representation, dataset_to_use, mdp_constr, transformer_model_names[1], timestep_norm=timestep_norm)
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
        '41-50' : indexes[(test_loader.dataset.data['ctgs'][:,0] >= 41) & (test_loader.dataset.data['ctgs'][:,0] <= 50)],
        '51-60' : indexes[(test_loader.dataset.data['ctgs'][:,0] >= 51) & (test_loader.dataset.data['ctgs'][:,0] <= 60)]
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
            'state_representation' : state_representation,
            'mdp_constr' : mdp_constr,
            'mpc_steps' : n_steps
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
        time_cvxMPC = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_myocvxMPC = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_artMPC = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_artMPC_dag0 = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_artMPC_dag1 = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_artMPC_dag2 = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_artMPC_dag3 = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_artMPC_dag4 = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_artMPC_dag5 = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_artMPC_dag6 = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_artMPC_dag7 = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_artMPC_dag8 = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
        time_artMPC_dag9 = np.empty(shape=(N_data_test, rpod.n_time_rpod), dtype=float)
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
            '''for i in tqdm(uniform_idx, total=len(uniform_idx)):
            # Save the input in the dataset
            res = for_computation((i, other_args))'''
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