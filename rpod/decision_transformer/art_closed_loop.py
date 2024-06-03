import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import torch
import cvxpy as cp
import time
import decision_transformer.manage as ART_manager
from dynamics.RpodEnv import RpodEnv
import optimization.ocp as ocp
import optimization.rpod_scenario as rpod
import copy

class AutonomousRendezvousTransformerMPC():
    '''
    Class to perform an rendezvous and docking in a closed-loop MPC fashion. The non-linear trajectory optimization problem is solved as an SCP problem
    warm-started using ART.

    Inputs:
        - model : the transformer model to be used for the warm-starting process;
        - test_loader : the dataloader of the dataset used to train the transformer model;
        - n_steps : the number of steps used for the horizon of the MPC;
        - transformer_mode : ('dyn'/'ol') string to select the desired mode to propagate the states in the transformer (default='dyn');
        - scp_mode : ('hard'/'soft') string to select whether to consider the waypoint constraints as hard or soft in the scp.
    
    Public methods:
        - warmstart : uses the transformer model of the object to compute the warm-start for the trajectory over the planning horizon;
        - solve_scp : uses the SCP to optimize the trajectory over the planning horizon starting from the warm-starting trajectory.
    '''

    # SCP data
    iter_max_SCP = rpod.iter_max_SCP # [-]
    trust_region0 = rpod.trust_region0 # [m]
    trust_regionf = rpod.trust_regionf # [m]
    J_tol = rpod.J_tol # [m/s]

    ########## CONSTRUCTOR ##########
    def __init__(self, model, test_loader, n_steps, transformer_mode='dyn', ctg_clipped=True, scp_mode='hard'):

        # Save the model, testing dataloader and the number of steps to use for MPC
        self.model = model
        self.test_loader = test_loader
        self.n_steps = n_steps
        self.transformer_mode = transformer_mode
        self.ctg_clipped = ctg_clipped
        self.scp_mode = scp_mode

        # Save data statistics
        self.data_stats = copy.deepcopy(test_loader.dataset.data_stats)
        self.data_stats['states_mean'] = self.data_stats['states_mean'].float().to(ART_manager.device)
        self.data_stats['states_std'] = self.data_stats['states_std'].float().to(ART_manager.device)
        self.data_stats['actions_mean'] = self.data_stats['actions_mean'].float().to(ART_manager.device)
        self.data_stats['actions_std'] = self.data_stats['actions_std'].float().to(ART_manager.device)
        self.data_stats['goal_mean'] = self.data_stats['goal_mean'].float().to(ART_manager.device)
        self.data_stats['goal_std'] = self.data_stats['goal_std'].float().to(ART_manager.device)
        if not test_loader.dataset.mdp_constr:
            self.data_stats['rtgs_mean'] = self.data_stats['rtgs_mean'].float().to(ART_manager.device)
            self.data_stats['rtgs_std'] = self.data_stats['rtgs_std'].float().to(ART_manager.device)

        # Initialize the history of all predictions
        self.__initialize_context_history()
    
    ########## CLOSED-LOOP METHODS ##########
    def warmstart(self, current_obs, current_env:RpodEnv, rtg0=None, rtgs_i=None, ctg0=None, return_dynamics=False):
        '''
        Function to use the ART model loaded in the class to predict the trajectory for the next self.n_steps, strating from the current environment.
        
        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : RpodEnv object containing the current information of the environment
            - rtg0 : initial rtg0 used for the conditioning of the maneuver, used in case the model performs offline-RL (default=None)
            - ctg0 : initial ctg0 used for the conditioning of the maneuver, used in case the model performs offline-RL (default=None)
            - rtgs_i : (1x100x1) true sequence of normalized rtgs, used only in case the model performs imitation learning (default=None)
            - return_dynamics : boolean flag to return the values of stm, cim and psi (default=False)
               
        Outputs:
            - ART_trajectory: dictionary contatining the predicted state and action trajectory over the next self.n_steps
            - stm : (6x6xself.n_steps) state transition matrices computed over the next self.n_steps time instants
            - cim : (6x3xself.n_steps) control input matrices computed over the next self.n_steps time instants
            - psi : (6x6xself.n_steps) mapping matrices from roe to rtn computed over the next self.n_steps time instants
        '''
        # Extract current information from the current_env
        t_i = current_env.timestep
        t_f = min(t_i + self.n_steps, self.test_loader.dataset.max_len)

        # Extract real state from environment observation
        current_state_rtn = current_obs['state_rtn']
        current_state_roe = current_obs['state_roe']
        current_state_oe = current_obs['oe']
        current_dock_param = current_obs['dock_param']

        if t_i == 0:            
            # Impose the initial state s0
            self.__extend_state_context(current_state_rtn, current_state_roe, t_i)

            # Impose the goal
            self.__extend_goal_context(current_dock_param['state_rtn_target'], t_i)

            # Impose the initial rtg0 and (eventually) ctg0
            if self.test_loader.dataset.mdp_constr:
                # For Offline-RL model impose unnormalized rtgs and ctgs
                self.__extend_rtgs_context(rtg0, t_i)
                self.__extend_ctgs_context(ctg0, t_i)
            else:
                # Use rtgs for the true history rtgs_i
                self.__extend_rtgs_context(rtgs_i[:,t_i,:].item(), t_i)
            
        else:
            # For time instants greater than 0
            # Extend the action
            latest_dv_rtn = current_obs['dv_rtn']
            self.__extend_action_context(latest_dv_rtn, t_i-1)

            # Extend the state
            self.__extend_state_context(current_state_rtn, current_state_roe, t_i)

            # Impose the goal
            self.__extend_goal_context(current_dock_param['state_rtn_target'], t_i)

            # Extend rtgs and (eventually) ctgs
            if self.test_loader.dataset.mdp_constr:
                # For Offline-RL model compute unnormalized unnormalized rtgs and ctgs
                past_reward = - np.linalg.norm(latest_dv_rtn)
                self.__extend_rtgs_context(self.rtgs_context[:,t_i-1,:].item() - past_reward, t_i)
                viol_dyn = ART_manager.torch_check_koz_constraint(self.rtn_context[:,t_i], t_i)
                self.__extend_ctgs_context(self.ctgs_context[:,t_i-1,:].item() - viol_dyn, t_i)
            else:
                # Use rtgs for the true history rtgs_i
                self.__extend_rtgs_context(rtgs_i[:,t_i,:].item(), t_i)

        # Precompute stm, cim and psi for the next n_steps
        stm, cim, psi = current_env.get_active_dynamic_constraints(current_state_oe, t_i, t_f)

        # Predict the trajectory for the next n_steps using the ART model and starting from the time and past context
        time_context = {
            'dt' : current_env.dt_hrz,
            't_i' : t_i,
            'n_steps' : self.n_steps,
            'period_ref' : current_env.period_ref
        }
        past_context = {
            'roe_context' : self.roe_context,
            'rtn_context' : self.rtn_context,
            'norm_state_context' : self.norm_state_context,
            'dv_context' : self.dv_context,
            'norm_action_context' : self.norm_action_context,
            'rtgs_context' : self.rtgs_context,
            'ctgs_context' : self.ctgs_context if self.test_loader.dataset.mdp_constr else None,
            'goal_context' : self.goal_context,
            'norm_goal_context' : self.norm_goal_context,
        }
        if self.test_loader.dataset.mdp_constr:
            if self.transformer_mode == 'dyn':
                ART_trajectory, ART_runtime = self.__torch_model_inference_dyn_closed_loop(self.model, self.test_loader, stm, cim, psi, time_context, past_context, ctg_clipped=self.ctg_clipped)
            else:
                ART_trajectory, ART_runtime = self.__torch_model_inference_ol_closed_loop(self.model, self.test_loader, stm, cim, psi, time_context, past_context, ctg_clipped=self.ctg_clipped)
        else:
            if self.transformer_mode == 'dyn':
                ART_trajectory, ART_runtime = self.__torch_model_inference_dyn_closed_loop(self.model, self.test_loader, stm, cim, psi, time_context, past_context, rtgs_i, ctg_clipped=self.ctg_clipped)
            else:
                ART_trajectory, ART_runtime = self.__torch_model_inference_ol_closed_loop(self.model, self.test_loader, stm, cim, psi, time_context, past_context, rtgs_i, ctg_clipped=self.ctg_clipped)

        if return_dynamics:
            return ART_trajectory, stm, cim, psi
        else:
            return ART_trajectory
    
    def solve_scp(self, current_obs, current_env:RpodEnv, stm, cim, psi, states_roe_ref, action_rtn_ref):
        '''
        Function to solve the scp optimization problem of MPC linearizing non-linear constraints with the reference trajectory provided in input.
        
        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : RpodEnv object containing the current information of the environment
            - stm : (6x6xself.n_steps) state transition matrices computed over the next self.n_steps time instants
            - cim : (6x3xself.n_steps) control input matrices computed over the next self.n_steps time instants
            - psi : (6x6xself.n_steps) mapping matrices from roe to rtn computed over the next self.n_steps time instants
            - state_roe_ref : (6xself.n_steps) roe state trajectory over the next self.n_steps time instants to be used for linearization
            - action_rtn_ref : (3xself.n_steps) rtn action trajectory over the next self.n_steps time instants to be used for linearization
               
        Outputs:
            - SCPMPC_trajectory : dictionary contatining the predicted state and action trajectory over the next self.n_steps
            - scp_dict : dictionary containing useful information on the scp solution:
                         feas : feasibility flag ('optimal'/'optimal_inaccurate'/'infeasible')
                         iter_SCP : number of scp iterations required to achieve convergence
                         J_vect : history of the cost along the scp iterations
                         runtime_scp : computational time required to solve the scp problem
        '''
        # Initial state and constraints extraction from the current environment
        t_i = current_env.timestep
        t_f = min(t_i + self.n_steps, self.test_loader.dataset.max_len)
        n_time = t_f - t_i
        # Extract real state from environment observation
        state_roe_0 = current_obs['state_roe'].copy()
        current_dock_param = current_obs['dock_param']
        constr_range = current_env.get_active_constraints(t_i, t_f)
        #print(constr_range['docking_port'])

        # SCP parameters initialization
        beta_SCP = (self.trust_regionf/self.trust_region0)**(1/self.iter_max_SCP)
        iter_SCP = 0
        DELTA_J = 10
        J_vect = np.ones(shape=(self.iter_max_SCP,), dtype=float)*1e12
        diff = self.trust_region0
        trust_region = self.trust_region0
        s_end = states_roe_ref[:,-1].copy()
        
        runtime0_scp = time.time()
        while (iter_SCP < self.iter_max_SCP) and ((diff > self.trust_regionf) or (DELTA_J > self.J_tol)):
            
            # Solve OCP (safe)
            try:
                [states_roe, actions, feas, cost] = self.__ocp_scp_closed_loop(stm, cim, psi, state_roe_0, current_dock_param, states_roe_ref, action_rtn_ref, trust_region, n_time, constr_range, current_env, s_end, self.scp_mode)
            except:
                states_roe = None
                actions = None
                feas = 'infeasible'

            if not np.char.equal(feas,'infeasible'):#'optimal'):
                if iter_SCP == 0:
                    states_roe_vect = states_roe[None,:,:].copy()
                    actions_vect = actions[None,:,:].copy()
                else:
                    states_roe_vect = np.vstack((states_roe_vect, states_roe[None,:,:]))
                    actions_vect = np.vstack((actions_vect, actions[None,:,:]))

                # Compute performances
                diff = np.max(la.norm(states_roe - states_roe_ref, axis=0))
                # print('scp gap:', diff)
                J = cost# sum(la.norm(actions,axis=0));#2,1
                J_vect[iter_SCP] = J
                #print(J)

                # Update iteration
                iter_SCP += 1
                
                if iter_SCP > 1:
                    DELTA_J = J_old - J
                J_old = J
                #print('-----')
                #print(DELTA_J)
                #print('*****')
                #print(feas)

                #  Update trust region
                trust_region = beta_SCP * trust_region
                
                #  Update reference
                states_roe_ref = states_roe
                action_rtn_ref = actions
            else:
                print(feas)
                print('unfeasible scp')
                break;
        
        runtime1_scp = time.time()
        runtime_scp = runtime1_scp - runtime0_scp
        
        ind_J_min = iter_SCP-1#np.argmin(J_vect)#
        if not np.char.equal(feas,'infeasible'):#np.char.equal(feas,'optimal'):
            states_roe = states_roe_vect[ind_J_min,:,:]
            states_rtn = (psi.transpose(2,0,1) @ states_roe[None,:,:].transpose(2,1,0))[:,:,0].transpose(1,0)
            actions = actions_vect[ind_J_min,:,:]
        else:
            states_roe = None
            states_rtn = None
            actions = None

        SCPMPC_trajectory = {
            'state_roe' : states_roe,
            'state_rtn' : states_rtn,
            'dv_rtn' : actions
        }
        scp_dict = {
            'feas' : feas,
            'iter_scp' : iter_SCP,
            'J_vect' : J_vect,
            'runtime_scp' : runtime_scp
        }

        return SCPMPC_trajectory, scp_dict

    ########## HISTORY/CONTEXT METHODS ##########
    def __initialize_context_history(self):
        '''
        Function to initialize the context and history vectors of the predictions requested to the ART, each of which will be characterized by n_steps.
        '''
        # Context initialization (N_batch x N_time x N_state/N_action/N_rtgs/N_ctgs) -> saved as torch to use them with transformer
        self.norm_state_context = torch.zeros((1, self.test_loader.dataset.max_len, 6)).float().to(ART_manager.device)
        self.roe_context = torch.zeros((6, self.test_loader.dataset.max_len)).float().to(ART_manager.device)
        self.rtn_context = torch.zeros((6, self.test_loader.dataset.max_len)).float().to(ART_manager.device)
        self.norm_action_context = torch.zeros((1, self.test_loader.dataset.max_len, 3)).float().to(ART_manager.device)
        self.dv_context = torch.zeros((3, self.test_loader.dataset.max_len)).float().to(ART_manager.device)
        self.norm_goal_context = torch.zeros((1, self.test_loader.dataset.max_len, 6)).float().to(ART_manager.device)
        self.goal_context = torch.zeros((6, self.test_loader.dataset.max_len)).float().to(ART_manager.device)
        self.rtgs_context = torch.zeros((1, self.test_loader.dataset.max_len, 1)).float().to(ART_manager.device)
        if self.test_loader.dataset.mdp_constr:
            self.ctgs_context = torch.zeros((1, self.test_loader.dataset.max_len, 1)).float().to(ART_manager.device)
    
    def __extend_state_context(self, state_rtn, state_roe, t):
        '''
        Function to extend the current state context with the information provided in input.
        '''
        # Update state context
        self.rtn_context[:,t] = torch.tensor(state_rtn).float().to(ART_manager.device)
        self.roe_context[:,t] = torch.tensor(state_roe).float().to(ART_manager.device)
        self.norm_state_context[:,t,:] = (self.rtn_context[:,t] - self.data_stats['states_mean'][t]) / (self.data_stats['states_std'][t] + 1e-6)
    
    def __extend_action_context(self, action_dv, t):
        '''
        Function to extend the current action context with the information provided in input.
        '''
        # Update action context
        self.dv_context[:,t] = torch.tensor(action_dv).float().to(ART_manager.device)
        self.norm_action_context[:,t,:] = (self.dv_context[:,t] - self.data_stats['actions_mean'][t]) / (self.data_stats['actions_std'][t] + 1e-6)
    
    def __extend_rtgs_context(self, rtg, t):
        '''
        Function to extend the current reward to go context with the information provided in input.
        '''
        # Update rtgs context
        self.rtgs_context[:,t,:] = torch.tensor(rtg).float().to(ART_manager.device)
    
    def __extend_ctgs_context(self, ctg, t):
        '''
        Function to extend the current constraint to go context with the information provided in input.
        '''
        # Update ctgs context
        self.ctgs_context[:,t,:] = torch.tensor(ctg).float().to(ART_manager.device)

    def __extend_goal_context(self, goal, t):
        '''
        Function to extend the current goal context with the information provided in input.
        '''
        # Update state context
        self.goal_context[:,t] = torch.tensor(goal).float().to(ART_manager.device)
        self.norm_goal_context[:,t,:] = (self.goal_context[:,t] - self.data_stats['goal_mean'][t]) / (self.data_stats['goal_std'][t] + 1e-6)

    ########## STATIC METHODS ##########
    @staticmethod
    def __torch_model_inference_dyn_closed_loop(model, test_loader, stm, cim, psi, time_context, past_context, rtgs_i=None, ctg_clipped=True):
    
        # Get dimensions and statistics from the dataset
        data_stats = copy.deepcopy(test_loader.dataset.data_stats)
        data_stats['states_mean'] = data_stats['states_mean'].float().to(ART_manager.device)
        data_stats['states_std'] = data_stats['states_std'].float().to(ART_manager.device)
        data_stats['actions_mean'] = data_stats['actions_mean'].float().to(ART_manager.device)
        data_stats['actions_std'] = data_stats['actions_std'].float().to(ART_manager.device)
        data_stats['goal_mean'] = data_stats['goal_mean'].float().to(ART_manager.device)
        data_stats['goal_std'] = data_stats['goal_std'].float().to(ART_manager.device)

        # Extract the time information over the horizon
        dt = time_context['dt']
        t_i = time_context['t_i']
        n_steps = time_context['n_steps']
        t_f = min(t_i + n_steps, test_loader.dataset.max_len)

        # Extract the orbital dynamics over the horizon
        stm = torch.from_numpy(stm).float().to(ART_manager.device)
        cim = torch.from_numpy(cim).float().to(ART_manager.device)
        psi = torch.from_numpy(psi).float().to(ART_manager.device)

        # Extract the past context for states, actions, rtgs and ctg -> !!!MAKE SURE TO COPY THEM!!!
        roe_dyn = past_context['roe_context'].clone().detach()
        rtn_dyn = past_context['rtn_context'].clone().detach()
        dv_dyn = past_context['dv_context'].clone().detach()
        states_dyn = past_context['norm_state_context'].clone().detach()
        actions_dyn = past_context['norm_action_context'].clone().detach()
        rtgs_dyn = past_context['rtgs_context'].clone().detach()
        norm_goal_dyn = past_context['norm_goal_context'].clone().detach()
        if test_loader.dataset.mdp_constr:
            ctgs_dyn = past_context['ctgs_context'].clone().detach()
        timesteps_i = torch.arange(0,t_f)[None,:].long().to(ART_manager.device)
        attention_mask_i = torch.ones(timesteps_i.shape).long().to(ART_manager.device)

        runtime0_DT = time.time()
        # For loop trajectory generation
        for t in np.arange(t_i, t_f):
            
            ##### Dynamics inference  
            # Compute action pred for dynamics model
            step = t - t_i
            with torch.no_grad():
                if test_loader.dataset.mdp_constr:
                    output_dyn = model(
                        states=states_dyn[:,:t+1,:],
                        actions=actions_dyn[:,:t+1,:],
                        goal=norm_goal_dyn[:,:t+1,:],
                        returns_to_go=rtgs_dyn[:,:t+1,:],
                        constraints_to_go=ctgs_dyn[:,:t+1,:],
                        timesteps=timesteps_i[:,:t+1],
                        attention_mask=attention_mask_i[:,:t+1],
                        return_dict=False,
                    )
                    (_, action_preds_dyn) = output_dyn
                else:
                    output_dyn = model(
                        states=states_dyn[:,:t+1,:],
                        actions=actions_dyn[:,:t+1,:],
                        goal=norm_goal_dyn[:,:t+1,:],
                        returns_to_go=rtgs_dyn[:,:t+1,:],
                        timesteps=timesteps_i[:,:t+1],
                        attention_mask=attention_mask_i[:,:t+1],
                        return_dict=False,
                    )
                    (_, action_preds_dyn, _) = output_dyn

            action_dyn_t = action_preds_dyn[0,t]
            actions_dyn[:,t,:] = action_dyn_t
            dv_dyn[:, t] = (action_dyn_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

            # Dynamics propagation of state variable 
            if t != t_f-1:
                roe_dyn[:,t+1] = stm[:,:,step] @ (roe_dyn[:,t] + cim[:,:,step] @ dv_dyn[:,t])
                rtn_dyn[:,t+1] = psi[:,:,step+1] @ roe_dyn[:,t+1]
                states_dyn_norm = (rtn_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
                states_dyn[:,t+1,:] = states_dyn_norm
                
                if test_loader.dataset.mdp_constr:
                    reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t])
                    rtgs_dyn[:,t+1,:] = rtgs_dyn[0,t] - reward_dyn_t
                    viol_dyn = ART_manager.torch_check_koz_constraint(rtn_dyn[:,t+1], t+1)
                    ctgs_dyn[:,t+1,:] = ctgs_dyn[0,t] - (viol_dyn if (not ctg_clipped) else 0)
                else:
                    rtgs_dyn[:,t+1,:] = rtgs_i[0,t+1]
                actions_dyn[:,t+1,:] = 0
                norm_goal_dyn[:,t+1,:] = norm_goal_dyn[0,t,:]
            
        time_sec = timesteps_i*dt
        time_orb = time_sec/time_context['period_ref']

        # Pack trajectory's data in a dictionary and compute runtime
        runtime1_DT = time.time()
        runtime_DT = runtime1_DT - runtime0_DT
        DT_trajectory = {
            'state_rtn' : rtn_dyn[:,t_i:t_f].cpu().numpy(),
            'state_roe' : roe_dyn[:,t_i:t_f].cpu().numpy(),
            'dv_rtn' : dv_dyn[:,t_i:t_f].cpu().numpy(),
            'time' : time_sec[0,t_i:t_f].cpu().numpy(),
            'time_orb' : time_orb[0,t_i:t_f].cpu().numpy()
        }

        return DT_trajectory, runtime_DT
    
    @staticmethod
    def __torch_model_inference_ol_closed_loop(model, test_loader, stm, cim, psi, time_context, past_context, rtgs_i=None, ctg_clipped=True):
    
        # Get dimensions and statistics from the dataset
        data_stats = copy.deepcopy(test_loader.dataset.data_stats)
        data_stats['states_mean'] = data_stats['states_mean'].float().to(ART_manager.device)
        data_stats['states_std'] = data_stats['states_std'].float().to(ART_manager.device)
        data_stats['actions_mean'] = data_stats['actions_mean'].float().to(ART_manager.device)
        data_stats['actions_std'] = data_stats['actions_std'].float().to(ART_manager.device)
        data_stats['goal_mean'] = data_stats['goal_mean'].float().to(ART_manager.device)
        data_stats['goal_std'] = data_stats['goal_std'].float().to(ART_manager.device)

        # Extract the time information over the horizon
        dt = time_context['dt']
        t_i = time_context['t_i']
        n_steps = time_context['n_steps']
        t_f = min(t_i + n_steps, test_loader.dataset.max_len)

        # Extract the orbital dynamics over the horizon
        stm = torch.from_numpy(stm).float().to(ART_manager.device)
        cim = torch.from_numpy(cim).float().to(ART_manager.device)
        psi = torch.from_numpy(psi).float().to(ART_manager.device)
        psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=ART_manager.device)[None,:,:]).permute(1,2,0).to(ART_manager.device)

        # Extract the past context for states, actions, rtgs and ctg -> !!!MAKE SURE TO COPY THEM!!!
        roe_ol = past_context['roe_context'].clone().detach()
        rtn_ol = past_context['rtn_context'].clone().detach()
        dv_ol = past_context['dv_context'].clone().detach()
        states_ol = past_context['norm_state_context'].clone().detach()
        actions_ol = past_context['norm_action_context'].clone().detach()
        rtgs_ol = past_context['rtgs_context'].clone().detach()
        norm_goal_ol = past_context['norm_goal_context'].clone().detach()
        if test_loader.dataset.mdp_constr:
            ctgs_ol = past_context['ctgs_context'].clone().detach()
        timesteps_i = torch.arange(0,t_f)[None,:].long().to(ART_manager.device)
        attention_mask_i = torch.ones(timesteps_i.shape).long().to(ART_manager.device)

        runtime0_DT = time.time()
        # For loop trajectory generation
        for t in np.arange(t_i, t_f):
            
            ##### Open-loop inference
            # Compute action pred for open-loop model
            step = t - t_i
            with torch.no_grad():
                if test_loader.dataset.mdp_constr:
                    output_ol = model(
                        states=states_ol[:,:t+1,:],
                        actions=actions_ol[:,:t+1,:],
                        goal=norm_goal_ol[:,:t+1,:],
                        returns_to_go=rtgs_ol[:,:t+1,:],
                        constraints_to_go=ctgs_ol[:,:t+1,:],
                        timesteps=timesteps_i[:,:t+1],
                        attention_mask=attention_mask_i[:,:t+1],
                        return_dict=False,
                    )
                    (_, action_preds_ol) = output_ol
                else:
                    output_ol = model(
                        states=states_ol[:,:t+1,:],
                        actions=actions_ol[:,:t+1,:],
                        goal=norm_goal_ol[:,:t+1,:],
                        returns_to_go=rtgs_ol[:,:t+1,:],
                        timesteps=timesteps_i[:,:t+1],
                        attention_mask=attention_mask_i[:,:t+1],
                        return_dict=False,
                    )
                    (_, action_preds_ol, _) = output_ol

            action_ol_t = action_preds_ol[0,t]
            actions_ol[:,t,:] = action_ol_t
            dv_ol[:, t] = (action_ol_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

            # Compute states pred for open-loop model
            with torch.no_grad():
                if test_loader.dataset.mdp_constr:
                    output_ol = model(
                        states=states_ol[:,:t+1,:],
                        actions=actions_ol[:,:t+1,:],
                        goal=norm_goal_ol[:,:t+1,:],
                        returns_to_go=rtgs_ol[:,:t+1,:],
                        constraints_to_go=ctgs_ol[:,:t+1,:],
                        timesteps=timesteps_i[:,:t+1],
                        attention_mask=attention_mask_i[:,:t+1],
                        return_dict=False,
                    )
                    (state_preds_ol, _) = output_ol
                else:
                    output_ol = model(
                        states=states_ol[:,:t+1,:],
                        actions=actions_ol[:,:t+1,:],
                        goal=norm_goal_ol[:,:t+1,:],
                        returns_to_go=rtgs_ol[:,:t+1,:],
                        timesteps=timesteps_i[:,:t+1],
                        attention_mask=attention_mask_i[:,:t+1],
                        return_dict=False,
                    )
                    (state_preds_ol, _, _) = output_ol
            
            state_ol_t = state_preds_ol[0,t]
            
            # Open-loop propagation of state variable
            if t != t_f-1:
                states_ol[:,t+1,:] = state_ol_t
                rtn_ol[:,t+1] = (state_ol_t * data_stats['states_std'][t+1]) + data_stats['states_mean'][t+1]
                roe_ol[:,t+1] = psi_inv[:,:,step+1] @ rtn_ol[:,t+1]

                if test_loader.dataset.mdp_constr:
                    reward_ol_t = - torch.linalg.norm(dv_ol[:, t])
                    rtgs_ol[:,t+1,:] = rtgs_ol[0,t] - reward_ol_t
                    viol_ol = ART_manager.torch_check_koz_constraint(rtn_ol[:,t+1], t+1)
                    ctgs_ol[:,t+1,:] = ctgs_ol[0,t] - (viol_ol if (not ctg_clipped) else 0)
                else:
                    rtgs_ol[:,t+1,:] = rtgs_i[0,t+1]
                actions_ol[:,t+1,:] = 0
                norm_goal_ol[:,t+1,:] = norm_goal_ol[0,t,:]

        time_sec = timesteps_i*dt
        time_orb = time_sec/time_context['period_ref']

        # Pack trajectory's data in a dictionary and compute runtime
        runtime1_DT = time.time()
        runtime_DT = runtime1_DT - runtime0_DT
        DT_trajectory = {
            'state_rtn' : rtn_ol[:,t_i:t_f].cpu().numpy(),
            'state_roe' : roe_ol[:,t_i:t_f].cpu().numpy(),
            'dv_rtn' : dv_ol[:,t_i:t_f].cpu().numpy(),
            'time' : time_sec[0,t_i:t_f].cpu().numpy(),
            'time_orb' : time_orb[0,t_i:t_f].cpu().numpy()
        }

        return DT_trajectory, runtime_DT

    @staticmethod
    def __ocp_scp_closed_loop(stm, cim, psi, s_0, dock_param, s_ref, u_ref, trust_region, n_time, constr_range, rpod_env:RpodEnv, s_end, scp_mode):

        s = cp.Variable((6, n_time))
        a = cp.Variable((3, n_time))

        # Docking port parameter extraction
        state_roe_target, dock_axis, dock_port, dock_cone_angle, dock_wyp = dock_param['state_roe_target'], dock_param['dock_axis'], dock_param['dock_port'], dock_param['dock_cone_angle'], dock_param['dock_wyp']
        
        # Compute parameters
        s_f = s_end
        d_soc = -np.transpose(dock_axis).dot(dock_port)/np.cos(dock_cone_angle)

        if scp_mode == 'hard':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == s_0]
            # Dynamics
            constraints += [s[:,h+1] == stm[:,:,h] @ (s[:,h] + cim[:,:,h] @ a[:,h]) for h in range(n_time-1)]
            # Trust region
            n_trust_region = constr_range['approach_cone'][0] if len(constr_range['approach_cone']) > 0 else n_time
            for i in range(n_trust_region):
                b_soc_i = -s_ref[:,i]
                constraints += [cp.SOC(trust_region, s[:,i] + b_soc_i)]
            # Docking port
            for j in constr_range['docking_port']:
                constraints += [s[:,j] + cim[:,:,j] @ a[:,j] == state_roe_target]

            # Additional constraints based on constr_range dictionary
            # Docking waypoint
            if len(constr_range['docking_waypoint']) > 0:
                constraints += [psi[:,:,k] @ s[:,k] == dock_wyp for k in constr_range['docking_waypoint']]
            # Approach cone
            for l in constr_range['approach_cone']:
                c_soc_l = np.transpose(dock_axis).dot(np.matmul(rpod_env.D_pos, psi[:,:,l]))/np.cos(dock_cone_angle)
                A_soc_l = np.matmul(rpod_env.D_pos, psi[:,:,l])
                b_soc_l = -dock_port
                constraints += [cp.SOC(c_soc_l @ s[:,l] + d_soc, A_soc_l @ s[:,l] + b_soc_l)]
            # Plume impingement
            for m in constr_range['plume_impingement']:
                A_plum_m = (dock_axis.T - (np.cos(rpod_env.plume_imp_angle)*u_ref[:,m].T)/np.sqrt(u_ref[:,m].T @ u_ref[:,m]))
                constraints += [A_plum_m @ a[:,m] <= 0]
            # Keep-out-zone plus trust region
            for n in constr_range['keep_out_zone']:
                c_koz_n = np.transpose(s_ref[:,n]).dot(np.matmul(np.transpose(psi[:,:,n]), np.matmul(rpod_env.DEED_koz, psi[:,:,n])))
                b_koz_n = np.sqrt(c_koz_n.dot(s_ref[:,n]))
                constraints += [c_koz_n @ s[:,n] >= b_koz_n]
            # Thrusters bounding box
            if len(constr_range['actions_bounding_box']) > 0:
                # If target poiting is active, then compute boudning box constraint in body RF
                if len(constr_range['target_pointing']) > 0:
                    b_bb_o = (rpod_env.dv_max*np.ones((3,)))**2
                    for o in constr_range['target_pointing']:
                        As_bb_o, Au_bb_o, c_bb_o = ocp.bounding_box_body_linearization(psi[:,:,o] @ s_ref[:,o], u_ref[:,o])
                        constraints += [As_bb_o @ (psi[:,:,o] @ s[:,o]) + Au_bb_o @ a[:,o] + c_bb_o <= b_bb_o]
                # Otherwise assume body RF == rtn RF
                else:
                    upper_bb_o = rpod_env.dv_max*np.ones((3,))
                    lower_bb_o = -rpod_env.dv_max*np.ones((3,))
                    for o in constr_range['actions_bounding_box']:
                        constraints += [a[:,o] <= upper_bb_o]
                        constraints += [a[:,o] >= lower_bb_o]

            # Compute Cost = action integral cost + terminal cost : sum(u'*R*u) + x_f'*P*x_f
            rho = 1.
            cost = cp.sum(cp.norm(a, 2, axis=0))
            if len(constr_range['docking_port']) == 0:
                cost = cost + rho*cp.norm(s[:,-1] - s_f, 2)
        
        elif scp_mode == 'soft':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == s_0]
            # Dynamics
            constraints += [s[:,h+1] == stm[:,:,h] @ (s[:,h] + cim[:,:,h] @ a[:,h]) for h in range(n_time-1)]
            # Trust region
            n_trust_region = constr_range['approach_cone'][0] if len(constr_range['approach_cone']) > 0 else n_time
            for i in range(n_trust_region):
                b_soc_i = -s_ref[:,i]
                constraints += [cp.SOC(trust_region, s[:,i] + b_soc_i)]

            # Additional constraints based on constr_range dictionary
            # Approach cone
            for l in np.delete(constr_range['approach_cone'], constr_range['approach_cone']==0):
                c_soc_l = np.transpose(dock_axis).dot(np.matmul(rpod_env.D_pos, psi[:,:,l]))/np.cos(dock_cone_angle)
                A_soc_l = np.matmul(rpod_env.D_pos, psi[:,:,l])
                b_soc_l = -dock_port
                constraints += [cp.SOC(c_soc_l @ s[:,l] + d_soc, A_soc_l @ s[:,l] + b_soc_l)]
            # Plume impingement
            for m in constr_range['plume_impingement']:
                A_plum_m = (dock_axis.T - (np.cos(rpod_env.plume_imp_angle)*u_ref[:,m].T)/np.sqrt(u_ref[:,m].T @ u_ref[:,m]))
                constraints += [A_plum_m @ a[:,m] <= 0]
            # Keep-out-zone plus trust region
            for n in np.delete(constr_range['keep_out_zone'], constr_range['keep_out_zone']==0):
                c_koz_n = np.transpose(s_ref[:,n]).dot(np.matmul(np.transpose(psi[:,:,n]), np.matmul(rpod_env.DEED_koz, psi[:,:,n])))
                b_koz_n = np.sqrt(c_koz_n.dot(s_ref[:,n]))
                constraints += [c_koz_n @ s[:,n] >= b_koz_n]
            # Thrusters bounding box
            if len(constr_range['actions_bounding_box']) > 0:
                # If target poiting is active, then compute boudning box constraint in body RF
                if len(constr_range['target_pointing']) > 0:
                    b_bb_o = (rpod_env.dv_max*np.ones((3,)))**2
                    for o in constr_range['target_pointing']:
                        As_bb_o, Au_bb_o, c_bb_o = ocp.bounding_box_body_linearization(psi[:,:,o] @ s_ref[:,o], u_ref[:,o])
                        constraints += [As_bb_o @ (psi[:,:,o] @ s[:,o]) + Au_bb_o @ a[:,o] + c_bb_o <= b_bb_o]
                # Otherwise assume body RF == rtn RF
                else:
                    upper_bb_o = rpod_env.dv_max*np.ones((3,))
                    lower_bb_o = -rpod_env.dv_max*np.ones((3,))
                    for o in constr_range['actions_bounding_box']:
                        constraints += [a[:,o] <= upper_bb_o]
                        constraints += [a[:,o] >= lower_bb_o]

            # Compute Cost = action integral cost + terminal cost : sum(u'*R*u) + x_f'*P*x_f
            rho = 0.1
            cost = 0.1*cp.sum(cp.norm(a, 2, axis=0))
            # Docking waypoint and docking port penalizing terms
            if len(constr_range['docking_waypoint']) > 0:
                j = constr_range['docking_waypoint'][0]
                cost = cost + 1*cp.norm((psi[:,:,j] @ s[:,j]) - dock_wyp, 2)
            if len(constr_range['docking_port']) > 0:
                k = constr_range['docking_port'][0]
                cost = cost + 1*cp.norm((s[:,k] + cim[:,:,k] @ a[:,k]) - state_roe_target, 2)
            else:
                cost = cost + rho*cp.norm(s[:,-1] - s_f, 2)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value

        return s_opt, a_opt, prob.status, prob.value


class ConvexMPC():
    '''
    Class to perform an rendezvous and docking in a closed-loop MPC fashion. The non-linear trajectory optimization problem is solved as an SCP problem
    warm-started using a convexified version of the true problem.

    Inputs:
        - n_steps : the number of steps used for the horizon of the MPC
        - scp_mode : ('hard'/'soft') string to select whether to consider the waypoint constraints as hard or soft in the scp.
    
    Public methods:
        - warmstart : compute the warm-start for the trajectory over the planning horizon using the convexified version of the problem;
        - solve_scp : uses the SCP to optimize the trajectory over the planning horizon starting from the warm-starting trajectory.
    '''
    # SCP data
    iter_max_SCP = rpod.iter_max_SCP # [-]
    trust_region0 = rpod.trust_region0 # [m]
    trust_regionf = rpod.trust_regionf # [m]
    J_tol = rpod.J_tol # [m/s]

    ########## CONSTRUCTOR ##########
    def __init__(self, n_steps, scp_mode='hard'):

        # Save the number of steps to use for MPC
        self.n_steps = n_steps
        self.scp_mode = scp_mode
    
    ########## CLOSED-LOOP METHODS ##########
    def warmstart(self, current_obs, current_env:RpodEnv, return_dynamics=False):
        '''
        Function to use the covexified optimization problem to predict the trajectory for the next self.n_steps, starting from the current environment.

        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : RpodEnv object containing the current information of the environment
            - return_dynamics : boolean flag to return the values of stm, cim and psi (default=False)
               
        Outputs:
            - CVX_trajectory: dictionary contatining the predicted state and action trajectory over the next self.n_steps
            - stm : (6x6xself.n_steps) state transition matrices computed over the next self.n_steps time instants
            - cim : (6x3xself.n_steps) control input matrices computed over the next self.n_steps time instants
            - psi : (6x6xself.n_steps) mapping matrices from roe to rtn computed over the next self.n_steps time instants
        '''
        # Extract current information from the current_env
        t_i = current_env.timestep
        t_f = current_env.n_time_rpod
        t_cut = min(t_i + self.n_steps, current_env.n_time_rpod)
        n_time_remaining = t_f - t_i
        # Extract real state from environment observation
        state_roe_0 = current_obs['state_roe']
        state_oe_0 = current_obs['oe']
        current_dock_param = current_obs['dock_param']

        # Precompute stm, cim and psi for the next n_steps
        stm, cim, psi = current_env.get_active_dynamic_constraints(state_oe_0, t_i, t_f)
        constr_range = current_env.get_active_constraints(t_i, t_f)

        # Predict the trajectory for the next n_steps using the covexified optimization problem and starting from the current time and state until the end of the maneuver
        runtime_cvx1 = time.time()
        [states_roe, actions, feas, cost] = self.__ocp_cvx_predict(stm, cim, psi, state_roe_0, current_dock_param, n_time_remaining, constr_range, current_env, self.scp_mode)
        runtime_cvx = time.time() - runtime_cvx1
        states_rtn = (psi.transpose(2,0,1) @ states_roe[None,:,:].transpose(2,1,0))[:,:,0].transpose(1,0)

        # Keep only the next self.n_steps for the output
        CVX_trajectory = {
            'state_roe' : states_roe[:,:(t_cut - t_i)],
            'state_rtn' : states_rtn[:,:(t_cut - t_i)],
            'dv_rtn' : actions[:,:(t_cut - t_i)],
            'time' : np.arange(t_i, t_cut)*current_env.dt_hrz,
            'time_orb' : np.arange(t_i, t_cut)*current_env.dt_hrz/current_env.period_ref
        }
        cvx_dict = {
            'feas' : feas,
            'runtime_scp' : runtime_cvx
        }
        if return_dynamics:
            return CVX_trajectory, stm[:,:,:(t_cut - t_i) - 1], cim[:,:,:(t_cut - t_i)], psi[:,:,:(t_cut - t_i)]
        else:
            return CVX_trajectory
    
    def solve_scp(self, current_obs, current_env:RpodEnv, stm, cim, psi, states_roe_ref, action_rtn_ref):
        '''
        Function to solve the scp optimization problem of MPC linearizing non-linear constraints with the reference trajectory provided in input.

        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : RpodEnv object containing the current information of the environment
            - stm : (6x6xself.n_steps) state transition matrices computed over the next self.n_steps time instants
            - cim : (6x3xself.n_steps) control input matrices computed over the next self.n_steps time instants
            - psi : (6x6xself.n_steps) mapping matrices from roe to rtn computed over the next self.n_steps time instants
            - state_roe_ref : (6xself.n_steps) roe state trajectory over the next self.n_steps time instants to be used for linearization
            - action_rtn_ref : (3xself.n_steps) rtn action trajectory over the next self.n_steps time instants to be used for linearization
               
        Outputs:
            - SCPMPC_trajectory : dictionary contatining the predicted state and action trajectory over the next self.n_steps
            - scp_dict : dictionary containing useful information on the scp solution:
                         feas : feasibility flag ('optimal'/'optimal_inaccurate'/'infeasible')
                         iter_SCP : number of scp iterations required to achieve convergence
                         J_vect : history of the cost along the scp iterations
                         runtime_scp : computational time required to solve the scp problem
        '''
        # Initial state and constraints extraction from the current environment
        t_i = current_env.timestep
        t_f = min(t_i + self.n_steps, current_env.n_time_rpod)
        n_time = t_f - t_i
        # Extract real state from environment observation
        state_roe_0 = current_obs['state_roe'].copy()
        current_dock_param = current_obs['dock_param']
        constr_range = current_env.get_active_constraints(t_i, t_f)
        #print(constr_range['docking_port'])

        # SCP parameters initialization
        beta_SCP = (self.trust_regionf/self.trust_region0)**(1/self.iter_max_SCP)
        iter_SCP = 0
        DELTA_J = 10
        J_vect = np.ones(shape=(self.iter_max_SCP,), dtype=float)*1e12
        diff = self.trust_region0
        trust_region = self.trust_region0
        s_end = states_roe_ref[:,-1].copy()
        
        runtime0_scp = time.time()
        while (iter_SCP < self.iter_max_SCP) and ((diff > self.trust_regionf) or (DELTA_J > self.J_tol)):
            
            # Solve OCP (safe)
            try:
                [states_roe, actions, feas, cost] = self.__ocp_scp_closed_loop(stm, cim, psi, state_roe_0, current_dock_param, states_roe_ref, action_rtn_ref, trust_region, n_time, constr_range, current_env, s_end, self.scp_mode)
            except:
                states_roe = None
                actions = None
                feas = 'infeasible'

            if not np.char.equal(feas,'infeasible'):#'optimal'):
                if iter_SCP == 0:
                    states_roe_vect = states_roe[None,:,:].copy()
                    actions_vect = actions[None,:,:].copy()
                else:
                    states_roe_vect = np.vstack((states_roe_vect, states_roe[None,:,:]))
                    actions_vect = np.vstack((actions_vect, actions[None,:,:]))

                # Compute performances
                diff = np.max(la.norm(states_roe - states_roe_ref, axis=0))
                # print('scp gap:', diff)
                J = cost# sum(la.norm(actions,axis=0));#2,1
                J_vect[iter_SCP] = J
                #print(J)

                # Update iteration
                iter_SCP += 1
                
                if iter_SCP > 1:
                    DELTA_J = J_old - J
                J_old = J
                #print('-----')
                #print(DELTA_J)
                #print('*****')
                #print(feas)

                #  Update trust region
                trust_region = beta_SCP * trust_region
                
                #  Update reference
                states_roe_ref = states_roe
                action_rtn_ref = actions
            else:
                print(feas)
                print('unfeasible scp')
                break;
        
        runtime1_scp = time.time()
        runtime_scp = runtime1_scp - runtime0_scp
        
        ind_J_min = iter_SCP-1#np.argmin(J_vect)
        if not np.char.equal(feas,'infeasible'):#np.char.equal(feas,'optimal'):
            states_roe = states_roe_vect[ind_J_min,:,:]
            states_rtn = (psi.transpose(2,0,1) @ states_roe[None,:,:].transpose(2,1,0))[:,:,0].transpose(1,0)
            actions = actions_vect[ind_J_min,:,:]
        else:
            states_roe = None
            states_rtn = None
            actions = None

        SCPMPC_trajectory = {
            'state_roe' : states_roe,
            'state_rtn' : states_rtn,
            'dv_rtn' : actions
        }
        scp_dict = {
            'feas' : feas,
            'iter_scp' : iter_SCP,
            'J_vect' : J_vect,
            'runtime_scp' : runtime_scp
        }

        return SCPMPC_trajectory, scp_dict

    ########## STATIC METHODS ##########
    @staticmethod
    def __ocp_cvx_predict(stm, cim, psi, s_0, dock_param, n_time, constr_range, rpod_env:RpodEnv, scp_mode):

        s = cp.Variable((6, n_time))
        a = cp.Variable((3, n_time))

        # Docking port parameter extraction
        state_roe_target, dock_axis, dock_port, dock_cone_angle, dock_wyp = dock_param['state_roe_target'], dock_param['dock_axis'], dock_param['dock_port'], dock_param['dock_cone_angle'], dock_param['dock_wyp']

        # Compute parameters
        s_f = state_roe_target
        d_soc = -np.transpose(dock_axis).dot(dock_port)/np.cos(dock_cone_angle)

        if scp_mode == 'hard':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == s_0]
            # Dynamics
            constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim[:,:,i] @ a[:,i]) for i in range(n_time-1)]
            # Terminal Condition
            constraints += [s[:,-1] + cim[:,:,-1] @ a[:,-1] == s_f]
            # Docking waypoint
            if len(constr_range['docking_waypoint']) > 0:
                constraints += [psi[:,:,k] @ s[:,k] == dock_wyp for k in constr_range['docking_waypoint']]
            # Approach cone
            for l in constr_range['approach_cone']:
                c_soc_l = np.transpose(dock_axis).dot(np.matmul(rpod_env.D_pos, psi[:,:,l]))/np.cos(dock_cone_angle)
                A_soc_l = np.matmul(rpod_env.D_pos, psi[:,:,l])
                b_soc_l = -dock_port
                constraints += [cp.SOC(c_soc_l @ s[:,l] + d_soc, A_soc_l @ s[:,l] + b_soc_l)]

            # Compute Cost
            cost = cp.sum(cp.norm(a, 2, axis=0))
        
        elif scp_mode == 'soft':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == s_0]
            # Dynamics
            constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim[:,:,i] @ a[:,i]) for i in range(n_time-1)]
            
            # Approach cone
            for l in np.delete(constr_range['approach_cone'], constr_range['approach_cone']==0):
                c_soc_l = np.transpose(dock_axis).dot(np.matmul(rpod_env.D_pos, psi[:,:,l]))/np.cos(dock_cone_angle)
                A_soc_l = np.matmul(rpod_env.D_pos, psi[:,:,l])
                b_soc_l = -dock_port
                constraints += [cp.SOC(c_soc_l @ s[:,l] + d_soc, A_soc_l @ s[:,l] + b_soc_l)]

            # Compute Cost
            cost = 0.1*cp.sum(cp.norm(a, 2, axis=0))
            # Docking waypoint and docking port penalizing terms
            if len(constr_range['docking_waypoint']) > 0:
                j = constr_range['docking_waypoint'][0]
                cost = cost + 1*cp.norm((psi[:,:,j] @ s[:,j]) - dock_wyp, 2)
            if len(constr_range['docking_port']) > 0:
                k = constr_range['docking_port'][0]
                cost = cost + 1*cp.norm((s[:,k] + cim[:,:,k] @ a[:,k]) - state_roe_target, 2)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value

        return s_opt, a_opt, prob.status, prob.value
    
    @staticmethod
    def __ocp_scp_closed_loop(stm, cim, psi, s_0, dock_param, s_ref, u_ref, trust_region, n_time, constr_range, rpod_env:RpodEnv, s_end, scp_mode):

        s = cp.Variable((6, n_time))
        a = cp.Variable((3, n_time))

        # Docking port parameter extraction
        state_roe_target, dock_axis, dock_port, dock_cone_angle, dock_wyp = dock_param['state_roe_target'], dock_param['dock_axis'], dock_param['dock_port'], dock_param['dock_cone_angle'], dock_param['dock_wyp']

        # Compute parameters
        s_f = s_end
        d_soc = -np.transpose(dock_axis).dot(dock_port)/np.cos(dock_cone_angle)

        if scp_mode == 'hard':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == s_0]
            # Dynamics
            constraints += [s[:,h+1] == stm[:,:,h] @ (s[:,h] + cim[:,:,h] @ a[:,h]) for h in range(n_time-1)]
            # Trust region
            n_trust_region = constr_range['approach_cone'][0] if len(constr_range['approach_cone']) > 0 else n_time
            for i in range(n_trust_region):
                b_soc_i = -s_ref[:,i]
                constraints += [cp.SOC(trust_region, s[:,i] + b_soc_i)]
            # Docking port
            for j in constr_range['docking_port']:
                constraints += [s[:,j] + cim[:,:,j] @ a[:,j] == state_roe_target]

            # Additional constraints based on constr_range dictionary
            # Docking waypoint
            if len(constr_range['docking_waypoint']) > 0:
                constraints += [psi[:,:,k] @ s[:,k] == dock_wyp for k in constr_range['docking_waypoint']]
            # Approach cone
            for l in constr_range['approach_cone']:
                c_soc_l = np.transpose(dock_axis).dot(np.matmul(rpod_env.D_pos, psi[:,:,l]))/np.cos(dock_cone_angle)
                A_soc_l = np.matmul(rpod_env.D_pos, psi[:,:,l])
                b_soc_l = -dock_port
                constraints += [cp.SOC(c_soc_l @ s[:,l] + d_soc, A_soc_l @ s[:,l] + b_soc_l)]
            # Plume impingement
            for m in constr_range['plume_impingement']:
                A_plum_m = (dock_axis.T - (np.cos(rpod_env.plume_imp_angle)*u_ref[:,m].T)/np.sqrt(u_ref[:,m].T @ u_ref[:,m]))
                constraints += [A_plum_m @ a[:,m] <= 0]
            # Keep-out-zone plus trust region
            for n in constr_range['keep_out_zone']:
                c_koz_n = np.transpose(s_ref[:,n]).dot(np.matmul(np.transpose(psi[:,:,n]), np.matmul(rpod_env.DEED_koz, psi[:,:,n])))
                b_koz_n = np.sqrt(c_koz_n.dot(s_ref[:,n]))
                constraints += [c_koz_n @ s[:,n] >= b_koz_n]
            # Thrusters bounding box
            if len(constr_range['actions_bounding_box']) > 0:
                # If target poiting is active, then compute boudning box constraint in body RF
                if len(constr_range['target_pointing']) > 0:
                    b_bb_o = (rpod_env.dv_max*np.ones((3,)))**2
                    for o in constr_range['target_pointing']:
                        As_bb_o, Au_bb_o, c_bb_o = ocp.bounding_box_body_linearization(psi[:,:,o] @ s_ref[:,o], u_ref[:,o])
                        constraints += [As_bb_o @ (psi[:,:,o] @ s[:,o]) + Au_bb_o @ a[:,o] + c_bb_o <= b_bb_o]
                # Otherwise assume body RF == rtn RF
                else:
                    upper_bb_o = rpod_env.dv_max*np.ones((3,))
                    lower_bb_o = -rpod_env.dv_max*np.ones((3,))
                    for o in constr_range['actions_bounding_box']:
                        constraints += [a[:,o] <= upper_bb_o]
                        constraints += [a[:,o] >= lower_bb_o]

            # Compute Cost = action integral cost + terminal cost : sum(u'*R*u) + x_f'*P*x_f
            rho = 1.
            cost = cp.sum(cp.norm(a, 2, axis=0))
            if len(constr_range['docking_port']) == 0:
                cost = cost + rho*cp.norm(s[:,-1] - s_f, 2)
        
        elif scp_mode == 'soft':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == s_0]
            # Dynamics
            constraints += [s[:,h+1] == stm[:,:,h] @ (s[:,h] + cim[:,:,h] @ a[:,h]) for h in range(n_time-1)]
            # Trust region
            n_trust_region = constr_range['approach_cone'][0] if len(constr_range['approach_cone']) > 0 else n_time
            for i in range(n_trust_region):
                b_soc_i = -s_ref[:,i]
                constraints += [cp.SOC(trust_region, s[:,i] + b_soc_i)]

            # Additional constraints based on constr_range dictionary
            # Approach cone
            for l in np.delete(constr_range['approach_cone'], constr_range['approach_cone']==0):
                c_soc_l = np.transpose(dock_axis).dot(np.matmul(rpod_env.D_pos, psi[:,:,l]))/np.cos(dock_cone_angle)
                A_soc_l = np.matmul(rpod_env.D_pos, psi[:,:,l])
                b_soc_l = -dock_port
                constraints += [cp.SOC(c_soc_l @ s[:,l] + d_soc, A_soc_l @ s[:,l] + b_soc_l)]
            # Plume impingement
            for m in constr_range['plume_impingement']:
                A_plum_m = (dock_axis.T - (np.cos(rpod_env.plume_imp_angle)*u_ref[:,m].T)/np.sqrt(u_ref[:,m].T @ u_ref[:,m]))
                constraints += [A_plum_m @ a[:,m] <= 0]
            # Keep-out-zone plus trust region
            for n in np.delete(constr_range['keep_out_zone'], constr_range['keep_out_zone']==0):
                c_koz_n = np.transpose(s_ref[:,n]).dot(np.matmul(np.transpose(psi[:,:,n]), np.matmul(rpod_env.DEED_koz, psi[:,:,n])))
                b_koz_n = np.sqrt(c_koz_n.dot(s_ref[:,n]))
                constraints += [c_koz_n @ s[:,n] >= b_koz_n]
            # Thrusters bounding box
            if len(constr_range['actions_bounding_box']) > 0:
                # If target poiting is active, then compute boudning box constraint in body RF
                if len(constr_range['target_pointing']) > 0:
                    b_bb_o = (rpod_env.dv_max*np.ones((3,)))**2
                    for o in constr_range['target_pointing']:
                        As_bb_o, Au_bb_o, c_bb_o = ocp.bounding_box_body_linearization(psi[:,:,o] @ s_ref[:,o], u_ref[:,o])
                        constraints += [As_bb_o @ (psi[:,:,o] @ s[:,o]) + Au_bb_o @ a[:,o] + c_bb_o <= b_bb_o]
                # Otherwise assume body RF == rtn RF
                else:
                    upper_bb_o = rpod_env.dv_max*np.ones((3,))
                    lower_bb_o = -rpod_env.dv_max*np.ones((3,))
                    for o in constr_range['actions_bounding_box']:
                        constraints += [a[:,o] <= upper_bb_o]
                        constraints += [a[:,o] >= lower_bb_o]

            # Compute Cost = action integral cost + terminal cost : sum(u'*R*u) + x_f'*P*x_f
            rho = 0.1
            cost = 0.1*cp.sum(cp.norm(a, 2, axis=0))
            # Docking waypoint and docking port penalizing terms
            if len(constr_range['docking_waypoint']) > 0:
                j = constr_range['docking_waypoint'][0]
                cost = cost + 1*cp.norm((psi[:,:,j] @ s[:,j]) - dock_wyp, 2)
            if len(constr_range['docking_port']) > 0:
                k = constr_range['docking_port'][0]
                cost = cost + 1*cp.norm((s[:,k] + cim[:,:,k] @ a[:,k]) - state_roe_target, 2)
            else:
                cost = cost + rho*cp.norm(s[:,-1] - s_f, 2)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value

        return s_opt, a_opt, prob.status, prob.value


class MyopicConvexMPC():
    '''
    Class to perform an rendezvous and docking in a closed-loop MPC fashion. The non-linear trajectory optimization problem is solved as an SCP problem
    warm-started using a myopic and local convexified version of the true problem.

    Inputs:
        - n_steps : the number of steps used for the horizon of the MPC;
        - scp_mode : ('hard'/'soft') string to select whether to consider the waypoint constraints as hard or soft in the scp.
    
    Public methods:
        - warmstart : compute the warm-start for the trajectory over the planning horizon using the myopic and local convexified version of the problem;
        - solve_scp : uses the SCP to optimize the trajectory over the planning horizon starting from the warm-starting trajectory.
    '''
    # CVX cost parameters
    # TBD
    # SCP data
    iter_max_SCP = rpod.iter_max_SCP # [-]
    trust_region0 = rpod.trust_region0 # [m]
    trust_regionf = rpod.trust_regionf # [m]
    J_tol = rpod.J_tol # [m/s]

    ########## CONSTRUCTOR ##########
    def __init__(self, n_steps, scp_mode='hard'):

        # Save the number of steps to use for MPC
        self.n_steps = n_steps
        self.scp_mode = scp_mode
    
    ########## CLOSED-LOOP METHODS ##########
    def warmstart(self, current_obs, current_env:RpodEnv, return_dynamics=False):
        '''
        Function to use the myopic and local covexified optimization problem to predict the trajectory for the next self.n_steps, starting from the current environment.
        The terminal cost used in the cvx problem is computed following Riccati's equation.

        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : RpodEnv object containing the current information of the environment
            - return_dynamics : boolean flag to return the values of stm, cim and psi (default=False)
               
        Outputs:
            - CVX_trajectory: dictionary contatining the predicted state and action trajectory over the next self.n_steps
            - stm : (6x6xself.n_steps) state transition matrices computed over the next self.n_steps time instants
            - cim : (6x3xself.n_steps) control input matrices computed over the next self.n_steps time instants
            - psi : (6x6xself.n_steps) mapping matrices from roe to rtn computed over the next self.n_steps time instants
        '''
        # Extract current information from the current_env
        t_i = current_env.timestep
        t_f = min(t_i + self.n_steps, current_env.n_time_rpod)
        n_time_remaining = t_f - t_i
        # Extract real state from environment observation
        state_roe_0 = current_obs['state_roe']
        state_oe_0 = current_obs['oe']
        current_dock_param = current_obs['dock_param']

        # Precompute stm, cim and psi for the next n_steps
        stm, cim, psi = current_env.get_active_dynamic_constraints(state_oe_0, t_i, t_f)
        constr_range = current_env.get_active_constraints(t_i, t_f)

        # Predict the trajectory for the next n_steps using the covexified optimization problem and starting from the current time and state until the end of the maneuver
        runtime_cvx1 = time.time()
        [states_roe, actions, feas, cost] = self.__ocp_cvx_myopic_predict(stm, cim, psi, state_roe_0, current_dock_param, n_time_remaining, constr_range, current_env, self.scp_mode)
        runtime_cvx = time.time() - runtime_cvx1
        states_rtn = (psi.transpose(2,0,1) @ states_roe[None,:,:].transpose(2,1,0))[:,:,0].transpose(1,0)

        # Keep only the next self.n_steps for the output
        CVX_trajectory = {
            'state_roe' : states_roe,
            'state_rtn' : states_rtn,
            'dv_rtn' : actions,
            'time' : np.arange(t_i, t_f)*current_env.dt_hrz,
            'time_orb' : np.arange(t_i, t_f)*current_env.dt_hrz/current_env.period_ref
        }
        cvx_dict = {
            'feas' : feas,
            'runtime_scp' : runtime_cvx
        }
        if return_dynamics:
            return CVX_trajectory, stm, cim, psi
        else:
            return CVX_trajectory
    
    def solve_scp(self, current_obs, current_env:RpodEnv, stm, cim, psi, states_roe_ref, action_rtn_ref):
        '''
        Function to solve the scp optimization problem of MPC linearizing non-linear constraints with the reference trajectory provided in input.

        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : RpodEnv object containing the current information of the environment
            - stm : (6x6xself.n_steps) state transition matrices computed over the next self.n_steps time instants
            - cim : (6x3xself.n_steps) control input matrices computed over the next self.n_steps time instants
            - psi : (6x6xself.n_steps) mapping matrices from roe to rtn computed over the next self.n_steps time instants
            - state_roe_ref : (6xself.n_steps) roe state trajectory over the next self.n_steps time instants to be used for linearization
            - action_rtn_ref : (3xself.n_steps) rtn action trajectory over the next self.n_steps time instants to be used for linearization
               
        Outputs:
            - SCPMPC_trajectory : dictionary contatining the predicted state and action trajectory over the next self.n_steps
            - scp_dict : dictionary containing useful information on the scp solution:
                         feas : feasibility flag ('optimal'/'optimal_inaccurate'/'infeasible')
                         iter_SCP : number of scp iterations required to achieve convergence
                         J_vect : history of the cost along the scp iterations
                         runtime_scp : computational time required to solve the scp problem
        '''
        # Initial state and constraints extraction from the current environment
        t_i = current_env.timestep
        t_f = min(t_i + self.n_steps, current_env.n_time_rpod)
        n_time = t_f - t_i
        # Extract real state from environment observation
        state_roe_0 = current_obs['state_roe'].copy()
        current_dock_param = current_obs['dock_param']
        constr_range = current_env.get_active_constraints(t_i, t_f)
        #print(constr_range['docking_port'])

        # SCP parameters initialization
        beta_SCP = (self.trust_regionf/self.trust_region0)**(1/self.iter_max_SCP)
        iter_SCP = 0
        DELTA_J = 10
        J_vect = np.ones(shape=(self.iter_max_SCP,), dtype=float)*1e12
        diff = self.trust_region0
        trust_region = self.trust_region0
        s_end = states_roe_ref[:,-1].copy()
        
        runtime0_scp = time.time()
        while (iter_SCP < self.iter_max_SCP) and ((diff > self.trust_regionf) or (DELTA_J > self.J_tol)):
            
            # Solve OCP (safe)
            try:
                [states_roe, actions, feas, cost] = self.__ocp_scp_closed_loop(stm, cim, psi, state_roe_0, current_dock_param, states_roe_ref, action_rtn_ref, trust_region, n_time, constr_range, current_env, s_end, self.scp_mode)
            except:
                states_roe = None
                actions = None
                feas = 'infeasible'

            if not np.char.equal(feas,'infeasible'):#'optimal'):
                if iter_SCP == 0:
                    states_roe_vect = states_roe[None,:,:].copy()
                    actions_vect = actions[None,:,:].copy()
                else:
                    states_roe_vect = np.vstack((states_roe_vect, states_roe[None,:,:]))
                    actions_vect = np.vstack((actions_vect, actions[None,:,:]))

                # Compute performances
                diff = np.max(la.norm(states_roe - states_roe_ref, axis=0))
                # print('scp gap:', diff)
                J = cost# sum(la.norm(actions,axis=0));#2,1
                J_vect[iter_SCP] = J
                #print(J)

                # Update iteration
                iter_SCP += 1
                
                if iter_SCP > 1:
                    DELTA_J = J_old - J
                J_old = J
                #print('-----')
                #print(DELTA_J)
                #print('*****')
                #print(feas)

                #  Update trust region
                trust_region = beta_SCP * trust_region
                
                #  Update reference
                states_roe_ref = states_roe
                action_rtn_ref = actions
            else:
                print(feas)
                print('unfeasible scp')
                break;
        
        runtime1_scp = time.time()
        runtime_scp = runtime1_scp - runtime0_scp
        
        ind_J_min = iter_SCP-1#np.argmin(J_vect)
        if not np.char.equal(feas,'infeasible'):#np.char.equal(feas,'optimal'):
            states_roe = states_roe_vect[ind_J_min,:,:]
            states_rtn = (psi.transpose(2,0,1) @ states_roe[None,:,:].transpose(2,1,0))[:,:,0].transpose(1,0)
            actions = actions_vect[ind_J_min,:,:]
        else:
            states_roe = None
            states_rtn = None
            actions = None

        SCPMPC_trajectory = {
            'state_roe' : states_roe,
            'state_rtn' : states_rtn,
            'dv_rtn' : actions
        }
        scp_dict = {
            'feas' : feas,
            'iter_scp' : iter_SCP,
            'J_vect' : J_vect,
            'runtime_scp' : runtime_scp
        }

        return SCPMPC_trajectory, scp_dict

    ########## STATIC METHODS ##########
    @staticmethod
    def __ocp_cvx_myopic_predict(stm, cim, psi, s_0, dock_param, n_time, constr_range, rpod_env:RpodEnv, scp_mode):

        s = cp.Variable((6, n_time))
        a = cp.Variable((3, n_time))

        # Docking port parameter extraction
        state_roe_target, dock_axis, dock_port, dock_cone_angle, dock_wyp = dock_param['state_roe_target'], dock_param['dock_axis'], dock_param['dock_port'], dock_param['dock_cone_angle'], dock_param['dock_wyp']

        # Compute parameters
        s_f = state_roe_target if (len(constr_range['docking_port']) > 0) else psi[:,:,-1]@dock_wyp
        d_soc = -np.transpose(dock_axis).dot(dock_port)/np.cos(dock_cone_angle)

        if scp_mode == 'hard':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == s_0]
            # Dynamics
            constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim[:,:,i] @ a[:,i]) for i in range(n_time-1)]
            # Docking port
            for j in constr_range['docking_port']:
                constraints += [s[:,j] + cim[:,:,j] @ a[:,j] == s_f]

            # Additional constraints based on constr_range dictionary
            # Docking waypoint
            if len(constr_range['docking_waypoint']) > 0:
                constraints += [psi[:,:,k] @ s[:,k] == dock_wyp for k in constr_range['docking_waypoint']]
            # Approach cone
            for l in constr_range['approach_cone']:
                c_soc_l = np.transpose(dock_axis).dot(np.matmul(rpod_env.D_pos, psi[:,:,l]))/np.cos(dock_cone_angle)
                A_soc_l = np.matmul(rpod_env.D_pos, psi[:,:,l])
                b_soc_l = -dock_port
                constraints += [cp.SOC(c_soc_l @ s[:,l] + d_soc, A_soc_l @ s[:,l] + b_soc_l)]

            # Compute Cost = action integral cost + terminal cost : sum(u'*R*u) + x_f'*P*x_f
            rho = 1.
            cost = cp.sum(cp.norm(a, 2, axis=0))
            if len(constr_range['docking_port']) == 0:
                cost = cost + rho*cp.norm(s[:,-1] - s_f, 2)
        
        elif scp_mode == 'soft':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == s_0]
            # Dynamics
            constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim[:,:,i] @ a[:,i]) for i in range(n_time-1)]
            
            # Approach cone
            for l in np.delete(constr_range['approach_cone'], constr_range['approach_cone']==0):
                c_soc_l = np.transpose(dock_axis).dot(np.matmul(rpod_env.D_pos, psi[:,:,l]))/np.cos(dock_cone_angle)
                A_soc_l = np.matmul(rpod_env.D_pos, psi[:,:,l])
                b_soc_l = -dock_port
                constraints += [cp.SOC(c_soc_l @ s[:,l] + d_soc, A_soc_l @ s[:,l] + b_soc_l)]

            # Compute Cost = action integral cost + terminal cost : sum(u'*R*u) + x_f'*P*x_f
            rho = 0.1
            cost = 0.1*cp.sum(cp.norm(a, 2, axis=0))
            # Docking waypoint and docking port penalizing terms
            if len(constr_range['docking_waypoint']) > 0:
                j = constr_range['docking_waypoint'][0]
                cost = cost + 1*cp.norm((psi[:,:,j] @ s[:,j]) - dock_wyp, 2)
            if len(constr_range['docking_port']) > 0:
                k = constr_range['docking_port'][0]
                cost = cost + 1*cp.norm((s[:,k] + cim[:,:,k] @ a[:,k]) - state_roe_target, 2)
            else:
                cost = cost + rho*cp.norm(s[:,-1] - s_f, 2)
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value

        return s_opt, a_opt, prob.status, prob.value
    
    @staticmethod
    def __ocp_scp_closed_loop(stm, cim, psi, s_0, dock_param, s_ref, u_ref, trust_region, n_time, constr_range, rpod_env:RpodEnv, s_end, scp_mode):

        s = cp.Variable((6, n_time))
        a = cp.Variable((3, n_time))

        # Docking port parameter extraction
        state_roe_target, dock_axis, dock_port, dock_cone_angle, dock_wyp = dock_param['state_roe_target'], dock_param['dock_axis'], dock_param['dock_port'], dock_param['dock_cone_angle'], dock_param['dock_wyp']

        # Compute parameters
        s_f = state_roe_target if (len(constr_range['docking_port']) > 0) else psi[:,:,-1]@dock_wyp
        d_soc = -np.transpose(dock_axis).dot(dock_port)/np.cos(dock_cone_angle)

        if scp_mode == 'hard':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == s_0]
            # Dynamics
            constraints += [s[:,h+1] == stm[:,:,h] @ (s[:,h] + cim[:,:,h] @ a[:,h]) for h in range(n_time-1)]
            # Trust region
            n_trust_region = constr_range['approach_cone'][0] if len(constr_range['approach_cone']) > 0 else n_time
            for i in range(n_trust_region):
                b_soc_i = -s_ref[:,i]
                constraints += [cp.SOC(trust_region, s[:,i] + b_soc_i)]
            # Docking port
            for j in constr_range['docking_port']:
                constraints += [s[:,j] + cim[:,:,j] @ a[:,j] == state_roe_target]

            # Additional constraints based on constr_range dictionary
            # Docking waypoint
            if len(constr_range['docking_waypoint']) > 0:
                constraints += [psi[:,:,k] @ s[:,k] == dock_wyp for k in constr_range['docking_waypoint']]
            # Approach cone
            for l in constr_range['approach_cone']:
                c_soc_l = np.transpose(dock_axis).dot(np.matmul(rpod_env.D_pos, psi[:,:,l]))/np.cos(dock_cone_angle)
                A_soc_l = np.matmul(rpod_env.D_pos, psi[:,:,l])
                b_soc_l = -dock_port
                constraints += [cp.SOC(c_soc_l @ s[:,l] + d_soc, A_soc_l @ s[:,l] + b_soc_l)]
            # Plume impingement
            for m in constr_range['plume_impingement']:
                A_plum_m = (dock_axis.T - (np.cos(rpod_env.plume_imp_angle)*u_ref[:,m].T)/np.sqrt(u_ref[:,m].T @ u_ref[:,m]))
                constraints += [A_plum_m @ a[:,m] <= 0]
            # Keep-out-zone plus trust region
            for n in constr_range['keep_out_zone']:
                c_koz_n = np.transpose(s_ref[:,n]).dot(np.matmul(np.transpose(psi[:,:,n]), np.matmul(rpod_env.DEED_koz, psi[:,:,n])))
                b_koz_n = np.sqrt(c_koz_n.dot(s_ref[:,n]))
                constraints += [c_koz_n @ s[:,n] >= b_koz_n]
            # Thrusters bounding box
            if len(constr_range['actions_bounding_box']) > 0:
                # If target poiting is active, then compute boudning box constraint in body RF
                if len(constr_range['target_pointing']) > 0:
                    b_bb_o = (rpod_env.dv_max*np.ones((3,)))**2
                    for o in constr_range['target_pointing']:
                        As_bb_o, Au_bb_o, c_bb_o = ocp.bounding_box_body_linearization(psi[:,:,o] @ s_ref[:,o], u_ref[:,o])
                        constraints += [As_bb_o @ (psi[:,:,o] @ s[:,o]) + Au_bb_o @ a[:,o] + c_bb_o <= b_bb_o]
                # Otherwise assume body RF == rtn RF
                else:
                    upper_bb_o = rpod_env.dv_max*np.ones((3,))
                    lower_bb_o = -rpod_env.dv_max*np.ones((3,))
                    for o in constr_range['actions_bounding_box']:
                        constraints += [a[:,o] <= upper_bb_o]
                        constraints += [a[:,o] >= lower_bb_o]

            # Compute Cost = action integral cost + terminal cost : sum(u'*R*u) + x_f'*P*x_f
            rho = 1.
            cost = cp.sum(cp.norm(a, 2, axis=0))
            if len(constr_range['docking_port']) == 0:
                cost = cost + rho*cp.norm(s[:,-1] - s_f, 2)
        
        elif scp_mode == 'soft':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == s_0]
            # Dynamics
            constraints += [s[:,h+1] == stm[:,:,h] @ (s[:,h] + cim[:,:,h] @ a[:,h]) for h in range(n_time-1)]
            # Trust region
            n_trust_region = constr_range['approach_cone'][0] if len(constr_range['approach_cone']) > 0 else n_time
            for i in range(n_trust_region):
                b_soc_i = -s_ref[:,i]
                constraints += [cp.SOC(trust_region, s[:,i] + b_soc_i)]

            # Additional constraints based on constr_range dictionary
            # Approach cone
            for l in np.delete(constr_range['approach_cone'], constr_range['approach_cone']==0):
                c_soc_l = np.transpose(dock_axis).dot(np.matmul(rpod_env.D_pos, psi[:,:,l]))/np.cos(dock_cone_angle)
                A_soc_l = np.matmul(rpod_env.D_pos, psi[:,:,l])
                b_soc_l = -dock_port
                constraints += [cp.SOC(c_soc_l @ s[:,l] + d_soc, A_soc_l @ s[:,l] + b_soc_l)]
            # Plume impingement
            for m in constr_range['plume_impingement']:
                A_plum_m = (dock_axis.T - (np.cos(rpod_env.plume_imp_angle)*u_ref[:,m].T)/np.sqrt(u_ref[:,m].T @ u_ref[:,m]))
                constraints += [A_plum_m @ a[:,m] <= 0]
            # Keep-out-zone plus trust region
            for n in np.delete(constr_range['keep_out_zone'], constr_range['keep_out_zone']==0):
                c_koz_n = np.transpose(s_ref[:,n]).dot(np.matmul(np.transpose(psi[:,:,n]), np.matmul(rpod_env.DEED_koz, psi[:,:,n])))
                b_koz_n = np.sqrt(c_koz_n.dot(s_ref[:,n]))
                constraints += [c_koz_n @ s[:,n] >= b_koz_n]
            # Thrusters bounding box
            if len(constr_range['actions_bounding_box']) > 0:
                # If target poiting is active, then compute boudning box constraint in body RF
                if len(constr_range['target_pointing']) > 0:
                    b_bb_o = (rpod_env.dv_max*np.ones((3,)))**2
                    for o in constr_range['target_pointing']:
                        As_bb_o, Au_bb_o, c_bb_o = ocp.bounding_box_body_linearization(psi[:,:,o] @ s_ref[:,o], u_ref[:,o])
                        constraints += [As_bb_o @ (psi[:,:,o] @ s[:,o]) + Au_bb_o @ a[:,o] + c_bb_o <= b_bb_o]
                # Otherwise assume body RF == rtn RF
                else:
                    upper_bb_o = rpod_env.dv_max*np.ones((3,))
                    lower_bb_o = -rpod_env.dv_max*np.ones((3,))
                    for o in constr_range['actions_bounding_box']:
                        constraints += [a[:,o] <= upper_bb_o]
                        constraints += [a[:,o] >= lower_bb_o]

            # Compute Cost = action integral cost + terminal cost : sum(u'*R*u) + x_f'*P*x_f
            rho = 0.1
            cost = 0.1*cp.sum(cp.norm(a, 2, axis=0))
            # Docking waypoint and docking port penalizing terms
            if len(constr_range['docking_waypoint']) > 0:
                j = constr_range['docking_waypoint'][0]
                cost = cost + 1*cp.norm((psi[:,:,j] @ s[:,j]) - dock_wyp, 2)
            if len(constr_range['docking_port']) > 0:
                k = constr_range['docking_port'][0]
                cost = cost + 1*cp.norm((s[:,k] + cim[:,:,k] @ a[:,k]) - state_roe_target, 2)
            else:
                cost = cost + rho*cp.norm(s[:,-1] - s_f, 2)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value

        return s_opt, a_opt, prob.status, prob.value

