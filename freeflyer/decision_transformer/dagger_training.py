import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import torch

from transformers import DecisionTransformerConfig
import decision_transformer.manage as TTO_manager
from decision_transformer.art import AutonomousFreeflyerTransformer
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
import itertools
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from dynamics.FreeflyerEnv import FreeflyerEnv
from decision_transformer.art_closed_loop import AutonomousFreeflyerTransformerMPC, ConvexMPC
import optimization.ff_scenario as ff
from dynamics.freeflyer import FreeflyerModel, ocp_no_obstacle_avoidance, compute_constraint_to_go
import time
from datetime import datetime
import copy

def for_computation(input_iterable):

    # Extract input
    current_idx = input_iterable[0]
    input_dict = input_iterable[1]
    model = input_dict['model']
    train_loader = input_dict['train_loader']
    mdp_constr = input_dict['mdp_constr']
    mpc_steps_min = input_dict['mpc_steps_min']
    mpc_steps_max = input_dict['mpc_steps_max']
    switch_window_min = input_dict['switch_window_min']
    switch_window_max = input_dict['switch_window_max']
    oracle_fixed = input_dict['oracle_fixed']

    # Output dictionary initialization
    out = {'dataset_ix' : [],
           'feasible_cvx' : True,
           'J_cvx' : [],
           'cvx_problem' : False,
           'feasible_artMPC' : True,
           'J_artMPC' : [],
           'time_artMPC' : [],
           'ctgs0_cvx' : [],
           'plan_steps' : [],

           'context_state': [],
           'context_state_norm': [],
           'context_action': [],
           'context_action_norm': [],
           'target_state': [],
           'target_action': [],
           'context_rtg': [],
           'context_ctg': [],
           'context_goal' : [],
           'context_goal_norm' : []
          }
   
    np.random.seed(int(datetime.now().timestamp()))
    sample = train_loader.dataset.getix(np.random.randint(0,train_loader.dataset.n_data))
    if not mdp_constr:
        states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = sample
    else:
        states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = sample
    data_stats = train_loader.dataset.data_stats
    out['dataset_ix'] = ix[0]
    dt = dt.item()
    state_init = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
    state_final = np.array((goal_i[0, 0, :] * data_stats['goal_std'][0]) + data_stats['goal_mean'][0])
    ff_model = FreeflyerModel()

    ####### Warmstart Convex Problem QUAD
    try:
        traj_cvx, _, _, feas_cvx = ocp_no_obstacle_avoidance(ff_model, state_init, state_final) #### Remember states_cvx:(6,101) actions_cvx:(3,100)
        states_cvx = traj_cvx['states']
        actions_cvx = traj_cvx['actions_G']
        if feas_cvx == 'infeasible':
            states_cvx = None
            actions_cvx = None
    except Exception as error:
        states_cvx = None
        actions_cvx = None
        feas_cvx = 'infeasible'
        print('======================================\n ======================================\n ======================================\n ======================================\n')
        print('\n The following error occurred: ', type(error).__name__, '-', error)
        print('\n Saving log_error_cvx file...........')
        np.savez_compressed(root_folder + '/optimization/saved_files/closed_loop/log_error_cvx',
                            train_loader = train_loader,
                            states_i = states_i,
                            rtgs_i = rtgs_i,
                            ix = ix,
                            state_init = state_init,
                            state_final = state_final,
                            n_time_rpod = ff.n_time_rpod,
                            )
        
    
    if not np.char.equal(feas_cvx,'infeasible'):
        out['J_cvx'] = sum(la.norm(actions_cvx,axis=0,ord=1))
        rtg_0 = -out['J_cvx']
        ctgs_cvx = compute_constraint_to_go(states_cvx.T, ff.obs['position'], (ff.obs['radius'] + ff.robot_radius)*ff.safety_margin)
        ctgs0_cvx = ctgs_cvx[0,0]
        out['ctgs0_cvx'] = ctgs0_cvx
        out['cvx_problem'] = ctgs0_cvx == 0

        ############## MPC methods
        # Create and initialize the environments and the MCPCs 
        quad_env_art = FreeflyerEnv()
        traj_sample = (dt, state_init, state_final)
        quad_env_art.reset('det',traj_sample)
        mpc_steps = np.random.randint(mpc_steps_min, mpc_steps_max+1)
        out['plan_steps'] = mpc_steps
        artMPC = AutonomousFreeflyerTransformerMPC(model,train_loader,mpc_steps,transformer_mode='dyn',ctg_clipped=True,scp_mode='soft')
        cvxMPC = ConvexMPC(100,scp_mode='soft')
        observed_state = np.zeros((quad_env_art.n_time_rpod,6))
        observed_goal = np.zeros((quad_env_art.n_time_rpod,6))

        # Select time windows to alternate the policies
        n_intervals = np.random.randint(1,4)
        switch_window = np.random.randint(switch_window_min, switch_window_max+1)
        if oracle_fixed:
            oracle_timesteps = set(np.arange(np.random.randint(quad_env_art.n_time_rpod-switch_window, quad_env_art.n_time_rpod-10), quad_env_art.n_time_rpod))
            intervals_remaining = n_intervals - 1
        else:
            oracle_timesteps = set()
            intervals_remaining = n_intervals
        for _ in np.arange(intervals_remaining):
            t_i = np.random.randint(0, quad_env_art.n_time_rpod - switch_window)
            oracle_timesteps = oracle_timesteps.union(np.arange(t_i, t_i + switch_window))
        
        oracle_timesteps = set()
        
        try:
            time_artMPC = np.empty((ff.n_time_rpod,))
            correction_pred_history = []
            for i in np.arange(ff.n_time_rpod):
                # Get observation and real state
                current_obs_art = quad_env_art.get_observation()
                observed_state[i,:] = current_obs_art['state']
                observed_goal[i,:] = current_obs_art['goal']
                real_obs_art = {
                    'state' : quad_env_art.state[:,-1].copy(),
                    'goal' : quad_env_art.goal[:,-1].copy()
                }
                # ART-ws
                tic = time.time()
                if mdp_constr:
                    art_traj = artMPC.warmstart(current_obs_art, quad_env_art, rtg0=rtg_0, ctg0=0)
                else:
                    art_traj = artMPC.warmstart(current_obs_art, quad_env_art, rtgs_i=rtgs_i)
                artMPC_traj, artMPC_scp_dict = artMPC.solve_scp(current_obs_art, quad_env_art, art_traj['state'], art_traj['dv'])
                time_artMPC[i] = time.time() - tic

                # Compute correction via CVX+SCP
                cvx_traj = cvxMPC.warmstart(real_obs_art, quad_env_art)
                cvxMPC_traj, cvxMPC_scp_dict = cvxMPC.solve_scp(real_obs_art, quad_env_art, cvx_traj['state'], cvx_traj['dv'])
                if (i > 0) and (cvxMPC_scp_dict['feas'] == 'infeasible'):
                    print('cvxMPC failed, using I-SCP.')
                    cvx_traj = prec_cvxMPC_traj
                    cvxMPC_traj, cvxMPC_scp_dict = cvxMPC.solve_scp(real_obs_art, quad_env_art, cvx_traj['state'][:,1:], cvx_traj['dv'][:,1:])
                prec_cvxMPC_traj = copy.deepcopy(cvxMPC_traj)
                correction_instant_pred = {}
                correction_instant_pred['state_CVX'] = cvx_traj['state']
                correction_instant_pred['dv_CVX'] = cvx_traj['dv']
                correction_instant_pred['state_CVXMPC'] = cvxMPC_traj['state']
                correction_instant_pred['dv_CVXMPC'] = cvxMPC_traj['dv']
                correction_instant_pred['time'] = cvx_traj['time']
                correction_pred_history.append(correction_instant_pred)

                # Run the environment forward based on ART+SCP or CVX+SCP control
                if i in oracle_timesteps:
                    quad_env_art.load_prediction(cvx_traj, cvxMPC_traj)
                    _ = quad_env_art.step(cvxMPC_traj['dv'][:,0])
                else:
                    quad_env_art.load_prediction(art_traj, artMPC_traj)
                    _ = quad_env_art.step(artMPC_traj['dv'][:,0])

            out['J_artMPC'] = np.sum(la.norm(quad_env_art.dv, axis=0, ord=1))
            out['time_artMPC'] = time_artMPC
            
            out['context_state'] = observed_state[None]
            out['context_state_norm'] = artMPC.norm_state_context.cpu()
            out['context_action'] = quad_env_art.dv.T[None]
            out['context_action_norm'] = artMPC.norm_action_context.cpu()
            out['context_rtg'] = artMPC.rtgs_context.cpu()[0]
            out['context_ctg'] = artMPC.ctgs_context.cpu()[0]
            out['context_goal'] = observed_goal[None]
            out['context_goal_norm'] = artMPC.norm_goal_context.cpu()
            out['target_state'] = torch.tensor(np.array([correction_pred_history[t]['state_CVXMPC'][:, 1] for t in range(ff.n_time_rpod-1)]))
            out['target_action'] = torch.tensor(np.array([correction_pred_history[t]['dv_CVXMPC'][:, 0] for t in range(ff.n_time_rpod)]))

        except:
            out['feasible_artMPC'] = False
        
    else:
        out['feasible_cvx'] = False
        out['feasible_artMPC'] = False

    return out

if __name__ == '__main__':

    # OPEN LOOP INITIAL DATASET AND MODEL
    initial_transformer_name = 'checkpoint_ff_ctgrtg'
    import_config_ol = TTO_manager.transformer_import_config(initial_transformer_name)
    mdp_constr = import_config_ol['mdp_constr']
    timestep_norm = import_config_ol['timestep_norm']
    train_only_on_cl = False
    oracle_fixed_flag = True
    # Get the datasets and loaders from the torch data
    datasets_ol, dataloaders_ol = TTO_manager.get_train_val_test_data(mdp_constr, timestep_norm)
    train_loader_ol, eval_loader_ol, test_loader_ol = dataloaders_ol
    data_stats = copy.deepcopy(train_loader_ol.dataset.data_stats)
    
    # Pool creation --> Should automatically select the maximum number of processes
    set_start_method('spawn')
    num_processes = 20
    p = Pool(processes=num_processes)
    n_dagger_i = 0
    N_dagger_max = 20    
    # Select radomically the seed
    np.random.seed(int(datetime.now().timestamp()))
    for n_dagger in range(n_dagger_i, N_dagger_max):

        print('================== DAGGER ITERATION', str(n_dagger), '====================')
        ########## DAGGER GENERATION ##########
        if n_dagger == 0:
            current_transformer_model_name = initial_transformer_name
        else:
            previous_transformer_model_name = initial_transformer_name + '_cl_' + str(n_dagger - 2) if (n_dagger > 1) else initial_transformer_name
            current_transformer_model_name = initial_transformer_name + '_cl_' + str(n_dagger - 1)

        # Get the current model with the open loop dataloaders
        print('DAGGER ITERATION', n_dagger , ': Generating closed-loop dataset with ART model', current_transformer_model_name)
        model = TTO_manager.get_DT_model(current_transformer_model_name, train_loader_ol, eval_loader_ol)

        # Parallel for inputs
        N_data_test = 2000#4000
        other_args = {
            'model' : model,
            'train_loader' : train_loader_ol,
            'mdp_constr' : mdp_constr,
            'mpc_steps_min' : 10,
            'mpc_steps_max' : 100,
            'switch_window_min' : 15,
            'switch_window_max' : 25,
            'oracle_fixed' : oracle_fixed_flag
        }

        J_cvx = np.empty(shape=(N_data_test, ), dtype=float)
        J_artMPC = np.empty(shape=(N_data_test, ), dtype=float)
        time_artMPC = np.empty(shape=(N_data_test, ff.n_time_rpod), dtype=float)
        ctgs0_cvx = np.empty(shape=(N_data_test, ), dtype=float)
        cvx_problem = np.full(shape=(N_data_test, ), fill_value=False)
        dataset_ix = -np.ones(shape=(N_data_test, ), dtype=int)
        plan_steps = -np.ones(shape=(N_data_test, ), dtype=int)

        context_state = np.empty(shape=(N_data_test, ff.n_time_rpod, 6), dtype=float)
        context_state_norm = np.empty(shape=(N_data_test, ff.n_time_rpod, 6), dtype=float)
        context_action = np.empty(shape=(N_data_test, ff.n_time_rpod, 3), dtype=float)
        context_action_norm = np.empty(shape=(N_data_test, ff.n_time_rpod, 3), dtype=float)
        target_state = np.empty(shape=(N_data_test, ff.n_time_rpod-1, 6), dtype=float)
        target_action = np.empty(shape=(N_data_test, ff.n_time_rpod, 3), dtype=float)
        context_rtg = np.empty(shape=(N_data_test, ff.n_time_rpod, 1), dtype=float)
        context_ctg = np.empty(shape=(N_data_test, ff.n_time_rpod, 1), dtype=float)
        context_goal = np.empty(shape=(N_data_test, ff.n_time_rpod, 6), dtype=float)
        context_goal_norm = np.empty(shape=(N_data_test, ff.n_time_rpod, 6), dtype=float)

        i_unfeas_cvx = []
        i_unfeas_artMPC = []
        for i, res in enumerate(tqdm(p.imap(for_computation, zip(np.arange(N_data_test), itertools.repeat(other_args))), total=N_data_test)):
            # Save the input in the dataset
            dataset_ix[i] = res['dataset_ix']

            # If the solution is feasible save the optimization output
            if res['feasible_cvx']:
                J_cvx[i] = res['J_cvx']
                ctgs0_cvx[i] = res['ctgs0_cvx']
                cvx_problem[i] = res['cvx_problem']
            else:
                i_unfeas_cvx += [ i ]
            
            if res['feasible_artMPC']:
                plan_steps[i] = res['plan_steps']
                J_artMPC[i] = res['J_artMPC']
                time_artMPC[i,:] = res['time_artMPC']
                context_state[i,:,:] = res['context_state']
                context_state_norm[i,:,:] = res['context_state_norm']
                context_action[i,:,:] = res['context_action']
                context_action_norm[i,:,:] = res['context_action_norm']
                target_state[i,:,:] = res['target_state']
                target_action[i,:,:] = res['target_action']
                context_rtg[i,:,:] = res['context_rtg']
                context_ctg[i,:,:] = res['context_ctg']
                context_goal[i,:,:] = res['context_goal']
                context_goal_norm[i,:,:] = res['context_goal_norm']
            else:
                i_unfeas_artMPC += [ i ]

        #  Save dataset (local folder for the workstation)
        print('DAGGER ITERATION', n_dagger , ': Saving the dataset as', root_folder + '/optimization/saved_files/closed_loop/dagger_' + current_transformer_model_name + '.npz')
        np.savez_compressed(root_folder + '/optimization/saved_files/closed_loop/dagger_' + current_transformer_model_name,
                            J_cvx = J_cvx,
                            J_artMPC = J_artMPC,
                            time_artMPC = time_artMPC,
                            cvx_problem = cvx_problem,
                            dataset_ix = dataset_ix,
                            i_unfeas_cvx = i_unfeas_cvx,
                            i_unfeas_artMPC = i_unfeas_artMPC,
                            plan_steps = plan_steps,
                            context_state = context_state,
                            context_state_norm = context_state_norm,
                            context_action = context_action,
                            context_action_norm = context_action_norm,
                            target_state = target_state,
                            target_action = target_action,
                            context_rtg = context_rtg,
                            context_ctg = context_ctg,
                            context_goal = context_goal,
                            context_goal_norm = context_goal_norm
                            )
        
        ########################################

        ########## MODEL TRAINING ##########
        # Retrieve data from the previous dagger generation
        cl_data = np.load(root_folder + '/optimization/saved_files/closed_loop/dagger_' + current_transformer_model_name + '.npz')
        dataset_ix = cl_data['dataset_ix']
        i_unfeas_artMPC = cl_data['i_unfeas_artMPC']
        mask = (dataset_ix != -1) & (~(np.isin(np.arange(N_data_test), i_unfeas_artMPC)))
        context_state = torch.from_numpy(cl_data['context_state'][mask,:,:])
        context_action = torch.from_numpy(cl_data['context_action'][mask,:,:])
        context_rtg = torch.from_numpy(cl_data['context_rtg'][mask,:,:])
        context_ctg = torch.from_numpy(cl_data['context_ctg'][mask,:,:])
        context_goal = torch.from_numpy(cl_data['context_goal'][mask,:,:])
        context_state_norm = (context_state - data_stats['states_mean']) / (data_stats['states_std'] + 1e-6)
        context_action_norm = (context_action - data_stats['actions_mean']) / (data_stats['actions_std'] + 1e-6)
        context_goal_norm = (context_goal - data_stats['goal_mean']) / (data_stats['goal_std'] + 1e-6)
        target_state = (torch.from_numpy(cl_data['target_state'])[mask,:,:] - data_stats['states_mean'][:-1,:]) / (data_stats['states_std'][:-1,:] + 1e-6)
        target_action = (torch.from_numpy(cl_data['target_action'])[mask,:,:] - data_stats['actions_mean']) / (data_stats['actions_std'] + 1e-6)
        data_param_cl = {
            'time_discr' : train_loader_ol.dataset.data['data_param']['time_discr'][dataset_ix[mask]],
            'time_sec' : train_loader_ol.dataset.data['data_param']['time_sec'][dataset_ix[mask]]
        }
        # Get CURRENT closed-loop train and eval data
        n = int(0.9*context_state_norm.shape[0])
        cl_train_data = {
            'states' : context_state_norm[:n, :],
            'actions' : context_action_norm[:n, :],
            'rtgs' : context_rtg[:n, :],
            'ctgs' : context_ctg[:n, :],
            'target_states' : target_state[:n, :],
            'target_actions' : target_action[:n, :],
            'goal' : context_goal_norm[:n, :],
            'data_param' : {
                'time_discr' : data_param_cl['time_discr'][:n],
                'time_sec' : data_param_cl['time_sec'][:n, :]
                },
            'data_stats' : data_stats
            }

        cl_val_data = {
            'states' : context_state_norm[n:, :],
            'actions' : context_action_norm[n:, :],
            'rtgs' : context_rtg[n:, :],
            'ctgs' : context_ctg[n:, :],
            'target_states' : target_state[n:, :],
            'target_actions' : target_action[n:, :],
            'goal' : context_goal_norm[n:, :],
            'data_param' : {
                'time_discr' : data_param_cl['time_discr'][n:],
                'time_sec' : data_param_cl['time_sec'][n:, :]
                },
            'data_stats' : data_stats
            }

        # Compute the aggregation of the closed-loop dataset
        # If we are at the first iteration --> initialize cl_train_data_dagger with only open_loop data
        if n_dagger == 0:
            # Dataset that will be used for training at each cycle --> progressive
            cl_train_data_dagger = {
                'states' : cl_train_data['states'].clone().detach(),
                'actions' : cl_train_data['actions'].clone().detach(),
                'rtgs' : cl_train_data['rtgs'].squeeze(-1).clone().detach(),
                'ctgs' : cl_train_data['ctgs'].squeeze(-1).clone().detach(),
                'target_states' : cl_train_data['target_states'].clone().detach(),
                'target_actions' : cl_train_data['target_actions'].clone().detach(),
                'goal' : cl_train_data['goal'].clone().detach(),
                'data_param' : {
                    'time_discr' : cl_train_data['data_param']['time_discr'].copy(),
                    'time_sec' : cl_train_data['data_param']['time_sec'].copy()
                    },
                'data_stats' : data_stats
            }
        else:
            # Load the previous cl_train_data_dagger
            cl_train_data_dagger = torch.load(root_folder + '/optimization/saved_files/closed_loop/dagger_cl_dataset/cl_train_data_dagger_' + previous_transformer_model_name + '.pth')
            # Extend cl_train_data_dagger with current closed_loop data
            cl_train_data_dagger = {
                'states' : torch.concatenate((cl_train_data_dagger['states'], cl_train_data['states'])),
                'actions' : torch.concatenate((cl_train_data_dagger['actions'], cl_train_data['actions'])),
                'rtgs' : torch.concatenate((cl_train_data_dagger['rtgs'], cl_train_data['rtgs'].squeeze(-1))),
                'ctgs' : torch.concatenate((cl_train_data_dagger['ctgs'], cl_train_data['ctgs'].squeeze(-1))),
                'target_states' : torch.concatenate((cl_train_data_dagger['target_states'], cl_train_data['target_states'])),
                'target_actions' : torch.concatenate((cl_train_data_dagger['target_actions'], cl_train_data['target_actions'])),
                'goal' : torch.concatenate((cl_train_data_dagger['goal'], cl_train_data['goal'])),
                'data_param' : {
                    'time_discr' : np.concatenate((cl_train_data_dagger['data_param']['time_discr'], cl_train_data['data_param']['time_discr'])),
                    'time_sec' : np.concatenate((cl_train_data_dagger['data_param']['time_sec'], cl_train_data['data_param']['time_sec']))
                    },
                'data_stats' : data_stats
            }
        # Save the dataset (in case the dagger training crashes and has to be resumed)
        torch.save(cl_train_data_dagger, root_folder + '/optimization/saved_files/closed_loop/dagger_cl_dataset/cl_train_data_dagger_' + current_transformer_model_name + '.pth')

        # Part of the training dataset coming from open-loop dataset
        n_ol = n*9 if (not train_only_on_cl) else 0
        rand_ix = np.random.choice(np.arange(train_loader_ol.dataset.n_data), n_ol, replace=False)
        ol_train_data = {
            'states' : train_loader_ol.dataset.data['states'][rand_ix, :],
            'actions' : train_loader_ol.dataset.data['actions'][rand_ix, :],
            'rtgs' : train_loader_ol.dataset.data['rtgs'][rand_ix, :],
            'ctgs' : train_loader_ol.dataset.data['ctgs'][rand_ix, :],
            'target_states' : train_loader_ol.dataset.data['target_states'][rand_ix, :],
            'target_actions' : train_loader_ol.dataset.data['target_actions'][rand_ix, :],
            'goal' : train_loader_ol.dataset.data['goal'][rand_ix, :],
            'data_param' : {
                'time_discr' : train_loader_ol.dataset.data['data_param']['time_discr'][rand_ix],
                'time_sec' : train_loader_ol.dataset.data['data_param']['time_sec'][rand_ix, :]
                },
            'data_stats' : data_stats
        }
        # Extend train_data_dagger with current closed_loop data
        train_data_dagger = {
            'states' : torch.concatenate((ol_train_data['states'], cl_train_data_dagger['states'])),
            'actions' : torch.concatenate((ol_train_data['actions'], cl_train_data_dagger['actions'])),
            'rtgs' : torch.concatenate((ol_train_data['rtgs'], cl_train_data_dagger['rtgs'].squeeze(-1))),
            'ctgs' : torch.concatenate((ol_train_data['ctgs'], cl_train_data_dagger['ctgs'].squeeze(-1))),
            'target_states' : torch.concatenate((ol_train_data['target_states'], cl_train_data_dagger['target_states'])),
            'target_actions' : torch.concatenate((ol_train_data['target_actions'], cl_train_data_dagger['target_actions'])),
            'goal' : torch.concatenate((ol_train_data['goal'], cl_train_data_dagger['goal'])),
            'data_param' : {
                'time_discr' : np.concatenate((ol_train_data['data_param']['time_discr'], cl_train_data_dagger['data_param']['time_discr'])),
                'time_sec' : np.concatenate((ol_train_data['data_param']['time_sec'], cl_train_data_dagger['data_param']['time_sec']))
                },
            'data_stats' : data_stats
        }

        # Create Dataset and Dataloaders for the aggragated training
        train_dataset = TTO_manager.RpodDataset(train_data_dagger, mdp_constr, target=True)
        ol_val_dataset = TTO_manager.RpodDataset(eval_loader_ol.dataset.data, mdp_constr, target=True) # !!!!!!!!!!!!! CONSTANT?
        cl_val_dataset = TTO_manager.RpodDataset(cl_val_data, mdp_constr, target=True)
        train_loader = TTO_manager.DataLoader(
            train_dataset,
            sampler=torch.utils.data.RandomSampler(
                train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=4,
            num_workers=0,
        )
        ol_eval_loader = TTO_manager.DataLoader(
            ol_val_dataset,
            sampler=torch.utils.data.RandomSampler(
                ol_val_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=4,
            num_workers=0,
        )
        cl_eval_loader = TTO_manager.DataLoader(
            cl_val_dataset,
            sampler=torch.utils.data.RandomSampler(
                cl_val_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=4,
            num_workers=0,
        )

        # Create the transformer model with the model used for this iteration of dagger generation
        config = DecisionTransformerConfig(
            state_dim=train_loader.dataset.n_state, 
            act_dim=train_loader.dataset.n_action,
            hidden_size=384,
            max_ep_len=100,
            vocab_size=1,
            action_tanh=False,
            n_positions=1024,
            n_layer=6,
            n_head=6,
            n_inner=None,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            )
        model = AutonomousFreeflyerTransformer(config)
        model_size = sum(t.numel() for t in model.parameters())
        print(f"GPT size: {model_size/1000**2:.1f}M parameters")
        model.to(TTO_manager.device);
        optimizer = AdamW(model.parameters(), lr=3e-5)
        accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_loader, ol_eval_loader
        )
        num_training_steps = 10000000000
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=num_training_steps,
        )
        accelerator.load_state(root_folder + '/decision_transformer/saved_files/checkpoints/' + current_transformer_model_name)

        # Training loop
        print('DAGGER ITERATION', n_dagger , ': Starting the training of model loaded from', root_folder + '/decision_transformer/saved_files/checkpoints/' + current_transformer_model_name)
        eval_iters = 100
        @torch.no_grad()
        def evaluate():
            model.eval()
            losses_ol = []
            losses_cl = []
            losses_state_ol = []
            losses_state_cl = []
            losses_action_ol = []
            losses_action_cl = []
            for step in range(eval_iters):
                # Evaluate Open-loop data
                data_iter_ol = iter(ol_eval_loader)
                states_i, actions_i, rtgs_i, ctgs_i, goal_i, target_states_i, target_actions_i, timesteps_i, attention_mask_i, _, _, _ = next(data_iter_ol)
                with torch.no_grad():
                    state_preds, action_preds = model(
                        states=states_i.to(TTO_manager.device),
                        actions=actions_i.to(TTO_manager.device),
                        goal=goal_i.to(TTO_manager.device),
                        returns_to_go=rtgs_i.to(TTO_manager.device),
                        constraints_to_go=ctgs_i.to(TTO_manager.device),
                        timesteps=timesteps_i.to(TTO_manager.device),
                        attention_mask=attention_mask_i.to(TTO_manager.device),
                        return_dict=False,
                    )
                loss_i = torch.mean((action_preds - target_actions_i.to(TTO_manager.device)) ** 2)
                loss_i_state = torch.mean((state_preds[:,:-1,:] - target_states_i.to(TTO_manager.device)) ** 2)
                losses_ol.append(accelerator.gather(loss_i + loss_i_state))
                losses_state_ol.append(accelerator.gather(loss_i_state))
                losses_action_ol.append(accelerator.gather(loss_i))

                # Evaluate Open-loop data
                data_iter_cl = iter(cl_eval_loader)
                states_i, actions_i, rtgs_i, ctgs_i, goal_i, target_states_i, target_actions_i, timesteps_i, attention_mask_i, _, _, _ = next(data_iter_cl)
                with torch.no_grad():
                    state_preds, action_preds = model(
                        states=states_i.to(TTO_manager.device),
                        actions=actions_i.to(TTO_manager.device),
                        goal=goal_i.to(TTO_manager.device),
                        returns_to_go=rtgs_i.to(TTO_manager.device),
                        constraints_to_go=ctgs_i.to(TTO_manager.device),
                        timesteps=timesteps_i.to(TTO_manager.device),
                        attention_mask=attention_mask_i.to(TTO_manager.device),
                        return_dict=False,
                    )
                loss_i = torch.mean((action_preds - target_actions_i.to(TTO_manager.device)) ** 2)
                loss_i_state = torch.mean((state_preds[:,:-1,:] - target_states_i.to(TTO_manager.device)) ** 2)
                losses_cl.append(accelerator.gather(loss_i + loss_i_state))
                losses_state_cl.append(accelerator.gather(loss_i_state))
                losses_action_cl.append(accelerator.gather(loss_i))
                
            loss_ol = torch.mean(torch.tensor(losses_ol))
            loss_state_ol = torch.mean(torch.tensor(losses_state_ol))
            loss_action_ol = torch.mean(torch.tensor(losses_action_ol))

            loss_cl = torch.mean(torch.tensor(losses_cl))
            loss_state_cl = torch.mean(torch.tensor(losses_state_cl))
            loss_action_cl = torch.mean(torch.tensor(losses_action_cl))
            model.train()
            return (loss_ol.item(), loss_cl.item()), (loss_state_ol.item(), loss_state_cl.item()), (loss_action_ol.item(), loss_action_cl.item())

        eval_steps = 500
        n_eval_max = 50
        samples_per_step = accelerator.state.num_processes * train_loader.batch_size
        model.train()
        completed_steps = 0
        log = {
            'loss_ol':[],
            'loss_state_ol':[],
            'loss_action_ol':[],
            'loss_cl':[],
            'loss_state_cl':[],
            'loss_action_cl':[]
        }
        for step, batch in enumerate(train_loader, start=0):
            with accelerator.accumulate(model):
                states_i, actions_i, rtgs_i, ctgs_i, goal_i, target_states_i, target_actions_i, timesteps_i, attention_mask_i, _, _, _ = batch
                state_preds, action_preds = model(
                    states=states_i.to(TTO_manager.device),
                    actions=actions_i.to(TTO_manager.device),
                    goal=goal_i.to(TTO_manager.device),
                    returns_to_go=rtgs_i.to(TTO_manager.device),
                    constraints_to_go=ctgs_i.to(TTO_manager.device),
                    timesteps=timesteps_i.to(TTO_manager.device),
                    attention_mask=attention_mask_i.to(TTO_manager.device),
                    return_dict=False,
                )
                loss_i_action = torch.mean((action_preds - target_actions_i.to(TTO_manager.device)) ** 2)
                loss_i_state = torch.mean((state_preds[:,:-1,:] - target_states_i.to(TTO_manager.device)) ** 2)
                loss = loss_i_action + loss_i_state
                if step % 100 == 0:
                    accelerator.print(
                        {
                            "lr": lr_scheduler.get_lr(),
                            "samples": step * samples_per_step,
                            "steps": completed_steps,
                            "loss/train": loss.item(),
                        }
                    )
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                if (step % (eval_steps)) == 0:
                    (eval_loss_ol, eval_loss_cl), (loss_state_ol, loss_state_cl), (loss_action_ol, loss_action_cl) = evaluate()
                    accelerator.print({"OL EVAL:  loss": eval_loss_ol, "loss/state": loss_state_ol, "loss/action": loss_action_ol})
                    accelerator.print({"CL EVAL:  loss": eval_loss_cl, "loss/state": loss_state_cl, "loss/action": loss_action_cl})

                    log['loss_ol'].append(eval_loss_ol)
                    log['loss_cl'].append(eval_loss_cl)
                    log['loss_state_ol'].append(loss_state_ol)
                    log['loss_state_cl'].append(loss_state_cl)
                    log['loss_action_ol'].append(loss_action_ol)
                    log['loss_action_cl'].append(loss_action_cl)
                    model.train()
                    accelerator.wait_for_everyone()
                
                if step >= n_eval_max*eval_steps:
                    # At the end of the current training break the for cycle
                    break
        
        # Save model and log
        transformer_model_name_4_saving = initial_transformer_name + '_cl_' + str(n_dagger)
        print('DAGGER ITERATION', n_dagger , ': Saving the model and the training log in', root_folder + '/decision_transformer/saved_files/checkpoints/' + transformer_model_name_4_saving)
        accelerator.save_state(root_folder + '/decision_transformer/saved_files/checkpoints/' + transformer_model_name_4_saving)
        np.savez_compressed(root_folder + '/decision_transformer/saved_files/checkpoints/' + transformer_model_name_4_saving + '/log',
                            log = log
                            )

