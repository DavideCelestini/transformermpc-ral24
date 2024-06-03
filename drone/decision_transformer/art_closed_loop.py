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
from dynamics.QuadEnv import QuadEnv, QuadModel
from dynamics.quadrotor import trust_region_diff
import optimization.quad_scenario as quad
import copy

class AutonomousQuadrotorTransformerMPC():
    '''
    Class to perform trajectory optimization in a closed-loop MPC fashion. The non-linear trajectory optimization problem is solved as an SCP problem
    warm-started using AQT.

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
    iter_max_SCP = quad.iter_max_SCP # [-]
    trust_region0 = quad.trust_region0 # [m]
    trust_regionf = quad.trust_regionf # [m]
    J_tol = quad.J_tol # [N]

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
    def warmstart(self, current_obs, current_env:QuadEnv, rtg0=None, rtgs_i=None, ctg0=None):
        '''
        Function to use the ART model loaded in the class to predict the trajectory for the next self.n_steps, strating from the current environment.
        
        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : QuadEnv object containing the current information of the environment
            - rtg0 : initial rtg0 used for the conditioning of the maneuver, used in case the model performs offline-RL (default=None)
            - ctg0 : initial ctg0 used for the conditioning of the maneuver, used in case the model performs offline-RL (default=None)
            - rtgs_i : (1x100x1) true sequence of normalized rtgs, used only in case the model performs imitation learning (default=None)
               
        Outputs:
            - ART_trajectory: dictionary contatining the predicted state and action trajectory over the next self.n_steps
        '''
        # Extract current information from the current_env
        t_i = current_env.timestep
        t_f = min(t_i + self.n_steps, self.test_loader.dataset.max_len)

        # Extract real state from environment observation
        current_state = current_obs['state']
        current_goal = current_obs['goal']
        obs_positions = torch.tensor(current_env.obs_positions).to(ART_manager.device)
        obs_radii = torch.tensor(current_env.obs_radii).to(ART_manager.device)

        if t_i == 0:            
            # Impose the initial state s0
            self.__extend_state_context(current_state, t_i)

            # Impose the goal
            self.__extend_goal_context(current_goal, t_i)

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
            latest_dv = current_obs['dv']
            self.__extend_action_context(latest_dv, t_i-1)

            # Extend the state
            self.__extend_state_context(current_state, t_i)

            # Extend the goal
            self.__extend_goal_context(current_goal, t_i)

            # Extend rtgs and (eventually) ctgs
            if self.test_loader.dataset.mdp_constr:
                # For Offline-RL model compute unnormalized unnormalized rtgs and ctgs
                past_reward = - (np.linalg.norm(latest_dv)**2)/2
                self.__extend_rtgs_context(self.rtgs_context[:,t_i-1,:].item() - past_reward, t_i)
                viol_dyn = ART_manager.torch_check_koz_constraint(self.state_context[:,t_i], obs_positions, obs_radii)
                self.__extend_ctgs_context(self.ctgs_context[:,t_i-1,:].item() - viol_dyn, t_i)
            else:
                # Use rtgs for the true history rtgs_i
                self.__extend_rtgs_context(rtgs_i[:,t_i,:].item(), t_i)

        # Predict the trajectory for the next n_steps using the ART model and starting from the time and past context
        time_context = {
            'dt' : current_env.dt,
            't_i' : t_i,
            'n_steps' : self.n_steps
        }
        past_context = {
            'state_context' : self.state_context,
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
                ART_trajectory, ART_runtime = self.__torch_model_inference_dyn_closed_loop(self.model, self.test_loader, time_context, past_context, obs_positions, obs_radii, ctg_clipped=self.ctg_clipped)
            else:
                ART_trajectory, ART_runtime = self.__torch_model_inference_ol_closed_loop(self.model, self.test_loader, time_context, past_context, obs_positions, obs_radii, ctg_clipped=self.ctg_clipped)
        else:
            if self.transformer_mode == 'dyn':
                ART_trajectory, ART_runtime = self.__torch_model_inference_dyn_closed_loop(self.model, self.test_loader, time_context, past_context, obs_positions, obs_radii, rtgs_i, ctg_clipped=self.ctg_clipped)
            else:
                ART_trajectory, ART_runtime = self.__torch_model_inference_ol_closed_loop(self.model, self.test_loader, time_context, past_context, obs_positions, obs_radii, rtgs_i, ctg_clipped=self.ctg_clipped)

        return ART_trajectory
    
    def solve_scp(self, current_obs, current_env:QuadEnv, states_ref, actions_ref):
        '''
        Function to solve the scp optimization problem of MPC linearizing non-linear constraints with the reference trajectory provided in input.
        
        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : RpodEnv object containing the current information of the environment
            - states_ref : (6xself.n_steps) state trajectory over the next self.n_steps time instants to be used for linearization
            - actions_ref : (3xself.n_steps) action trajectory over the next self.n_steps time instants to be used for linearization
               
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
        current_state = current_obs['state']
        current_goal = current_obs['goal']

        # Makse sure the inputs have the correct dimensions (n_steps, n_state) and (n_steps, n_actions)
        states_ref = states_ref.T
        state_end_ref = states_ref[-1,:]
        actions_ref = actions_ref.T
        J_vect = np.ones(shape=(self.iter_max_SCP,), dtype=float)*1e12
        
        # Initial condition for the scp
        DELTA_J = 10
        trust_region = self.trust_region0
        beta_SCP = (self.trust_regionf/self.trust_region0)**(1/self.iter_max_SCP)
        runtime0_scp = time.time()
        for scp_iter in range(self.iter_max_SCP):
            '''print("scp_iter =", scp_iter)'''
            # Solve OCP (safe)
            try:
                states, actions, cost, feas = self.__ocp_scp_closed_loop(states_ref, actions_ref, current_state, current_goal, state_end_ref, t_i, t_f, current_env, trust_region, self.scp_mode)
            except:
                states = None
                actions = None
                feas = 'infeasible'
            
            if not np.char.equal(feas,'infeasible'):
                J_vect[scp_iter] = cost
                
                # compute error
                L2_error = trust_region_diff(states, states_ref) #L2_error_us(us, us_prev)
                if scp_iter > 0:
                    DELTA_J = cost_prev - cost

                # Update iterations
                states_ref = states
                actions_ref = actions
                cost_prev = cost
                trust_region = beta_SCP*trust_region
                if scp_iter >= 1 and (L2_error <= self.trust_regionf and abs(DELTA_J) < self.J_tol):
                    break
            else:
                print(feas)
                print('unfeasible scp') 
                break
        runtime1_scp = time.time()
        runtime_scp = runtime1_scp - runtime0_scp

        SCPMPC_trajectory = {
            'state' : states.T if not np.char.equal(feas,'infeasible') else None,
            'dv' : actions.T if not np.char.equal(feas,'infeasible') else None
        }
        scp_dict = {
            'feas' : feas,
            'iter_scp' : scp_iter,
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
        self.state_context = torch.zeros((6, self.test_loader.dataset.max_len)).float().to(ART_manager.device)
        self.norm_action_context = torch.zeros((1, self.test_loader.dataset.max_len, 3)).float().to(ART_manager.device)
        self.dv_context = torch.zeros((3, self.test_loader.dataset.max_len)).float().to(ART_manager.device)
        self.norm_goal_context = torch.zeros((1, self.test_loader.dataset.max_len, 6)).float().to(ART_manager.device)
        self.goal_context = torch.zeros((6, self.test_loader.dataset.max_len)).float().to(ART_manager.device)
        self.rtgs_context = torch.zeros((1, self.test_loader.dataset.max_len, 1)).float().to(ART_manager.device)
        if self.test_loader.dataset.mdp_constr:
            self.ctgs_context = torch.zeros((1, self.test_loader.dataset.max_len, 1)).float().to(ART_manager.device)
    
    def __extend_state_context(self, state, t):
        '''
        Function to extend the current state context with the information provided in input.
        '''
        # Update state context
        self.state_context[:,t] = torch.tensor(state).float().to(ART_manager.device)
        self.norm_state_context[:,t,:] = (self.state_context[:,t] - self.data_stats['states_mean'][t]) / (self.data_stats['states_std'][t] + 1e-6)
    
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
    def __torch_model_inference_dyn_closed_loop(model, test_loader, time_context, past_context, obs_positions, obs_radii, rtgs_i=None, ctg_clipped=True):
    
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

        # Extract the past context for states, actions, rtgs and ctg -> !!!MAKE SURE TO COPY THEM!!!
        xyz_dyn = past_context['state_context'].clone().detach()
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
                xyz_dyn[:,t+1] = xyz_dyn[:,t] + ART_manager.torch_quad_propagator(xyz_dyn[:, t], dv_dyn[:, t])*dt
                states_dyn_norm = (xyz_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
                states_dyn[:,t+1,:] = states_dyn_norm
                
                if test_loader.dataset.mdp_constr:
                    reward_dyn_t = - (torch.linalg.norm(dv_dyn[:, t])**2)/2
                    rtgs_dyn[:,t+1,:] = rtgs_dyn[0,t] - reward_dyn_t
                    viol_dyn = ART_manager.torch_check_koz_constraint(xyz_dyn[:,t+1], obs_positions, obs_radii)
                    ctgs_dyn[:,t+1,:] = ctgs_dyn[0,t] - (viol_dyn if (not ctg_clipped) else 0)
                else:
                    rtgs_dyn[:,t+1,:] = rtgs_i[0,t+1]
                actions_dyn[:,t+1,:] = 0
                norm_goal_dyn[:,t+1,:] = norm_goal_dyn[0,t,:]
            
        time_sec = timesteps_i*dt

        # Pack trajectory's data in a dictionary and compute runtime
        runtime1_DT = time.time()
        runtime_DT = runtime1_DT - runtime0_DT
        DT_trajectory = {
            'state' : xyz_dyn[:,t_i:t_f].cpu().numpy(),
            'dv' : dv_dyn[:,t_i:t_f].cpu().numpy(),
            'time' : time_sec[0,t_i:t_f].cpu().numpy()
        }

        return DT_trajectory, runtime_DT
    
    @staticmethod
    def __torch_model_inference_ol_closed_loop(model, test_loader, time_context, past_context, obs_positions, obs_radii, rtgs_i=None, ctg_clipped=True):
    
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

        # Extract the past context for states, actions, rtgs and ctg -> !!!MAKE SURE TO COPY THEM!!!
        xyz_ol = past_context['roe_context'].clone().detach()
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
                xyz_ol[:,t+1] = (state_ol_t * data_stats['states_std'][t+1]) + data_stats['states_mean'][t+1]

                if test_loader.dataset.mdp_constr:
                    reward_ol_t = - (torch.linalg.norm(dv_ol[:, t])**2)/2
                    rtgs_ol[:,t+1,:] = rtgs_ol[0,t] - reward_ol_t
                    viol_ol = ART_manager.torch_check_koz_constraint(xyz_ol[:,t+1], obs_positions, obs_radii)
                    ctgs_ol[:,t+1,:] = ctgs_ol[0,t] - (viol_ol if (not ctg_clipped) else 0)
                else:
                    rtgs_ol[:,t+1,:] = rtgs_i[0,t+1]
                actions_ol[:,t+1,:] = 0
                norm_goal_ol[:,t+1,:] = norm_goal_ol[0,t,:]

        time_sec = timesteps_i*dt

        # Pack trajectory's data in a dictionary and compute runtime
        runtime1_DT = time.time()
        runtime_DT = runtime1_DT - runtime0_DT
        DT_trajectory = {
            'state' : xyz_ol[:,t_i:t_f].cpu().numpy(),
            'dv_rtn' : dv_ol[:,t_i:t_f].cpu().numpy(),
            'time' : time_sec[0,t_i:t_f].cpu().numpy()
        }

        return DT_trajectory, runtime_DT

    @staticmethod
    def __ocp_scp_closed_loop(xs, us, x_init, x_final, xs_end_ref, t_i, t_f, env:QuadEnv, trust_region, scp_mode):
        # IMPORTANT: xs and us are the references and must be of shape (n_steps,n_state) and (n_steps,n_actions)
        # Setup SQP problem
        xs, us = xs.T, us.T
        n_time = xs.shape[1]
        qm = env.quad_model

        s = cp.Variable((6, n_time))
        a = cp.Variable((3, n_time))

        if scp_mode == 'hard':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == x_init]
            # Dynamics
            A = np.array([[0, 0, 0, 1, 0, 0.],
                        [0, 0, 0, 0, 1, 0.],
                        [0, 0, 0, 0, 0, 1.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.]])
            B = 1/env.mass*np.vstack((np.zeros((3,3)),np.eye(3)))
            constraints += [s[:,i+1] == (s[:,i] + ((A + qm.A_d(xs[:,i])) @ s[:,i] + qm.b_d(xs[:,i]) + B @ a[:,i])*env.dt) for i in range(n_time-1)]
            # Terminal Condition
            if t_f == env.n_time_rpod:
                constraints += [s[:,-1] + ((A + qm.A_d(xs[:,-1])) @ s[:,-1] + qm.b_d(xs[:,-1]) + B @ a[:,-1])*env.dt == x_final]
            # Control contraints
            constraints += [a >= -env.u_max]
            constraints += [a <= env.u_max]
            # Keep-out-zone plus trust region
            for k in range(n_time):
                # KOZ
                for n_obs in range(len(env.obs_radii)):
                    c_koz_k = np.transpose(xs[:3,k] - env.obs_positions[n_obs,:]).dot(np.eye(3)/(env.obs_radii[n_obs]**2))
                    b_koz_k = np.sqrt(c_koz_k.dot(xs[:3,k] - env.obs_positions[n_obs,:]))
                    constraints += [c_koz_k @ (s[:3,k] - env.obs_positions[n_obs,:]) >= b_koz_k]
                # Trust region
                b_soc_k = -xs[:,k]
                constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
            # Compute Cost
            rho = 1.
            cost = 0.5*cp.sum(cp.norm(a, 2, axis=0)**2)
            if t_f < env.n_time_rpod:
                cost = cost + rho*cp.norm(s[:,-1] - xs_end_ref, 2)

        else:
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == x_init]
            # Dynamics
            A = np.array([[0, 0, 0, 1, 0, 0.],
                        [0, 0, 0, 0, 1, 0.],
                        [0, 0, 0, 0, 0, 1.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.]])
            B = 1/env.mass*np.vstack((np.zeros((3,3)),np.eye(3)))
            constraints += [s[:,i+1] == (s[:,i] + ((A + qm.A_d(xs[:,i])) @ s[:,i] + qm.b_d(xs[:,i]) + B @ a[:,i])*env.dt) for i in range(n_time-1)]
            
            # Control contraints
            constraints += [a >= -env.u_max]
            constraints += [a <= env.u_max]
            # Keep-out-zone plus trust region --> Imposed for all the timestep except the current one (since we can't modify the current state)
            for k in range(1, n_time):
                #KOZ
                for n_obs in range(len(env.obs_radii)):
                    c_koz_k = np.transpose(xs[:3,k] - env.obs_positions[n_obs,:]).dot(np.eye(3)/(env.obs_radii[n_obs]**2))
                    b_koz_k = np.sqrt(c_koz_k.dot(xs[:3,k] - env.obs_positions[n_obs,:]))
                    constraints += [c_koz_k @ (s[:3,k] - env.obs_positions[n_obs,:]) >= b_koz_k]
                # Trust region
                b_soc_k = -xs[:,k]
                constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
            
            # Compute Cost
            rho = 1.
            cost = 0.5*cp.sum(cp.norm(a, 2, axis=0)**2)
            # Goal reaching penalizing term: if the end of the maneuver is already in the planning horizon aim for the goal
            if t_f == env.n_time_rpod:
                cost = cost + 10*cp.norm((s[:,-1] + ((A + qm.A_d(xs[:,-1])) @ s[:,-1] + qm.b_d(xs[:,-1]) +  B @ a[:,-1])*env.dt) - x_final, 2)
            
            # Otherwise follow the warmstarting reference
            else:
                cost = cost + rho*cp.norm(s[:,-1] - xs_end_ref, 2)
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # SolveOSQP problem
        prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status == 'infeasible':
            print("[solve]: Problem infeasible.")
            s_opt = None
            a_opt = None
            J = None
        else:
            s_opt = s.value.T
            a_opt = a.value.T
            J = prob.value

        return s_opt, a_opt, J, prob.status


class ConvexMPC():
    '''
    Class to perform trajectory optimization in a closed-loop MPC fashion. The non-linear trajectory optimization problem is solved as an SCP problem
    warm-started using a convexified version of the true problem.

    Inputs:
        - n_steps : the number of steps used for the horizon of the MPC
        - scp_mode : ('hard'/'soft') string to select whether to consider the waypoint constraints as hard or soft in the scp.
    
    Public methods:
        - warmstart : compute the warm-start for the trajectory over the planning horizon using the convexified version of the problem;
        - solve_scp : uses the SCP to optimize the trajectory over the planning horizon starting from the warm-starting trajectory.
    '''
    # SCP data
    iter_max_SCP = quad.iter_max_SCP # [-]
    trust_region0 = quad.trust_region0 # [m]
    trust_regionf = quad.trust_regionf # [m]
    J_tol = quad.J_tol # [N]

    ########## CONSTRUCTOR ##########
    def __init__(self, n_steps, scp_mode='hard'):

        # Save the number of steps to use for MPC
        self.n_steps = n_steps
        self.scp_mode = scp_mode
    
    ########## CLOSED-LOOP METHODS ##########
    def warmstart(self, current_obs, current_env:QuadEnv):
        '''
        Function to use the covexified optimization problem to predict the trajectory for the next self.n_steps, starting from the current environment.

        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : RpodEnv object containing the current information of the environment
               
        Outputs:
            - CVX_trajectory: dictionary contatining the predicted state and action trajectory over the next self.n_steps
        '''
        # Extract current information from the current_env
        t_i = current_env.timestep
        t_f = current_env.n_time_rpod
        t_cut = min(t_i + self.n_steps, current_env.n_time_rpod)
        n_time_remaining = t_f - t_i
        T_rem = n_time_remaining*current_env.dt
        # Extract real state from environment observation
        current_state = current_obs['state']
        current_goal = current_obs['goal']

        # Line warmstart from the quadrotor model -> Make sure the inputs have the correct dimensions (n_steps, n_state) and (n_steps, n_actions)
        states_ref = (current_state + ((current_goal - current_state)/T_rem)*np.arange(0,T_rem,current_env.dt)[:,None])
        state_end_ref = None
        actions_ref = np.zeros((n_time_remaining,3))
        J_vect = np.ones(shape=(self.iter_max_SCP,), dtype=float)*1e12

        # Initial condition for the scp
        DELTA_J = 10
        trust_region = self.trust_region0
        beta_SCP = (self.trust_regionf/self.trust_region0)**(1/self.iter_max_SCP)
        runtime0_scp = time.time()
        runtime0_cvx = time.time()
        for scp_iter in range(self.iter_max_SCP):
            # Solve OCP (safe)
            try:
                states, actions, cost, feas = self.__ocp_scp_closed_loop(states_ref, actions_ref, current_state, current_goal, state_end_ref, t_i, t_f, current_env, trust_region, self.scp_mode, obs_av=False)
            except:
                states = None
                actions = None
                feas = 'infeasible'
            
            if not np.char.equal(feas,'infeasible'):
                J_vect[scp_iter] = cost
                # compute error
                L2_error = trust_region_diff(states, states_ref) #L2_error_us(us, us_prev)
                if scp_iter > 0:
                    DELTA_J = cost_prev - cost

                # Update iterations
                states_ref = states
                actions_ref = actions
                cost_prev = cost
                trust_region = beta_SCP*trust_region
                if scp_iter >= 1 and (L2_error <= self.trust_regionf or abs(DELTA_J) < self.J_tol):
                    break
            else:
                print(feas)
                print('unfeasible scp') 
                break
        runtime1_cvx = time.time()
        runtime_cvx = runtime1_cvx - runtime0_cvx

        # Keep only the next self.n_steps for the output
        CVX_trajectory = {
            'state' : (states.T)[:,:(t_cut - t_i)],
            'dv' : (actions.T)[:,:(t_cut - t_i)],
            'time' : np.arange(t_i, t_cut)*current_env.dt,
        }
        cvx_dict = {
            'feas' : feas,
            'runtime_scp' : runtime_cvx
        }

        return CVX_trajectory
    
    def solve_scp(self, current_obs, current_env:QuadEnv, states_ref, actions_ref):
        '''
        Function to solve the scp optimization problem of MPC linearizing non-linear constraints with the reference trajectory provided in input.
        
        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : RpodEnv object containing the current information of the environment
            - states_ref : (6xself.n_steps) state trajectory over the next self.n_steps time instants to be used for linearization
            - actions_ref : (3xself.n_steps) action trajectory over the next self.n_steps time instants to be used for linearization
               
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
        current_state = current_obs['state']
        current_goal = current_obs['goal']

        # Makse sure the inputs have the correct dimensions (n_steps, n_state) and (n_steps, n_actions)
        states_ref = states_ref.T
        state_end_ref = states_ref[-1,:]
        actions_ref = actions_ref.T
        J_vect = np.ones(shape=(self.iter_max_SCP,), dtype=float)*1e12
        
        # Initial condition for the scp
        DELTA_J = 10
        trust_region = self.trust_region0
        beta_SCP = (self.trust_regionf/self.trust_region0)**(1/self.iter_max_SCP)
        runtime0_scp = time.time()
        for scp_iter in range(self.iter_max_SCP):
            '''print("scp_iter =", scp_iter)'''
            # Solve OCP (safe)
            try:
                states, actions, cost, feas = self.__ocp_scp_closed_loop(states_ref, actions_ref, current_state, current_goal, state_end_ref, t_i, t_f, current_env, trust_region, self.scp_mode)
            except:
                states = None
                actions = None
                feas = 'infeasible'
            
            if not np.char.equal(feas,'infeasible'):
                J_vect[scp_iter] = cost
                # compute error
                L2_error = trust_region_diff(states, states_ref) #L2_error_us(us, us_prev)
                if scp_iter > 0:
                    DELTA_J = cost_prev - cost

                # Update iterations
                states_ref = states
                actions_ref = actions
                cost_prev = cost
                trust_region = beta_SCP*trust_region
                if scp_iter >= 1 and (L2_error <= self.trust_regionf and abs(DELTA_J) < self.J_tol):
                    break
            else:
                print(feas)
                print('unfeasible scp') 
                break
        runtime1_scp = time.time()
        runtime_scp = runtime1_scp - runtime0_scp

        SCPMPC_trajectory = {
            'state' : states.T if not np.char.equal(feas,'infeasible') else None,
            'dv' : actions.T if not np.char.equal(feas,'infeasible') else None
        }
        scp_dict = {
            'feas' : feas,
            'iter_scp' : scp_iter,
            'J_vect' : J_vect,
            'runtime_scp' : runtime_scp
        }

        return SCPMPC_trajectory, scp_dict

    ########## STATIC METHODS ##########
    @staticmethod
    def __ocp_scp_closed_loop(xs, us, x_init, x_final, xs_end_ref, t_i, t_f, env:QuadEnv, trust_region, scp_mode, obs_av=True):
        # IMPORTANT: xs and us are the references and must be of shape (n_steps,n_state) and (n_steps,n_actions)
        # Setup SQP problem
        xs, us = xs.T, us.T
        n_time = xs.shape[1]
        qm = env.quad_model

        s = cp.Variable((6, n_time))
        a = cp.Variable((3, n_time))

        if scp_mode == 'hard':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == x_init]
            # Dynamics
            A = np.array([[0, 0, 0, 1, 0, 0.],
                        [0, 0, 0, 0, 1, 0.],
                        [0, 0, 0, 0, 0, 1.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.]])
            B = 1/env.mass*np.vstack((np.zeros((3,3)),np.eye(3)))
            constraints += [s[:,i+1] == (s[:,i] + ((A + qm.A_d(xs[:,i])) @ s[:,i] + qm.b_d(xs[:,i]) + B @ a[:,i])*env.dt) for i in range(n_time-1)]
            # Terminal Condition
            if t_f == env.n_time_rpod:
                constraints += [s[:,-1] + ((A + qm.A_d(xs[:,-1])) @ s[:,-1] + qm.b_d(xs[:,-1]) + B @ a[:,-1])*env.dt == x_final]
            # Control contraints
            constraints += [a >= -env.u_max]
            constraints += [a <= env.u_max]
            # Keep-out-zone plus trust region
            if obs_av:
                for k in range(n_time):
                    # KOZ
                    for n_obs in range(len(env.obs_radii)):
                        c_koz_k = np.transpose(xs[:3,k] - env.obs_positions[n_obs,:]).dot(np.eye(3)/(env.obs_radii[n_obs]**2))
                        b_koz_k = np.sqrt(c_koz_k.dot(xs[:3,k] - env.obs_positions[n_obs,:]))
                        constraints += [c_koz_k @ (s[:3,k] - env.obs_positions[n_obs,:]) >= b_koz_k]
                    # Trust region
                    b_soc_k = -xs[:,k]
                    constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
            else:
                for k in range(n_time):
                    # Trust region
                    b_soc_k = -xs[:,k]
                    constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
            
            # Compute Cost
            rho = 1.
            cost = 0.5*cp.sum(cp.norm(a, 2, axis=0)**2)
            if t_f < env.n_time_rpod:
                cost = cost + rho*cp.norm(s[:,-1] - xs_end_ref, 2)

        else:
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == x_init]
            # Dynamics
            A = np.array([[0, 0, 0, 1, 0, 0.],
                        [0, 0, 0, 0, 1, 0.],
                        [0, 0, 0, 0, 0, 1.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.]])
            B = 1/env.mass*np.vstack((np.zeros((3,3)),np.eye(3)))
            constraints += [s[:,i+1] == (s[:,i] + ((A + qm.A_d(xs[:,i])) @ s[:,i] + qm.b_d(xs[:,i]) + B @ a[:,i])*env.dt) for i in range(n_time-1)]
            
            # Control contraints
            constraints += [a >= -env.u_max]
            constraints += [a <= env.u_max]
            # Keep-out-zone plus trust region --> Imposed for all the timestep except the current one (since we can't modify the current state)
            if obs_av:
                for k in range(1,n_time):
                    # KOZ
                    for n_obs in range(len(env.obs_radii)):
                        c_koz_k = np.transpose(xs[:3,k] - env.obs_positions[n_obs,:]).dot(np.eye(3)/(env.obs_radii[n_obs]**2))
                        b_koz_k = np.sqrt(c_koz_k.dot(xs[:3,k] - env.obs_positions[n_obs,:]))
                        constraints += [c_koz_k @ (s[:3,k] - env.obs_positions[n_obs,:]) >= b_koz_k]
                    # Trust region
                    b_soc_k = -xs[:,k]
                    constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
            else:
                for k in range(1,n_time):
                    # Trust region
                    b_soc_k = -xs[:,k]
                    constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
            
            # Compute Cost
            rho = 1.
            cost = 0.5*cp.sum(cp.norm(a, 2, axis=0)**2)
            # Goal reaching penalizing term: if the end of the maneuver is already in the planning horizon aim for the goal
            if t_f == env.n_time_rpod:
                cost = cost + 10*cp.norm((s[:,-1] + ((A + qm.A_d(xs[:,-1])) @ s[:,-1] + qm.b_d(xs[:,-1]) +  B @ a[:,-1])*env.dt) - x_final, 2)
            
            # Otherwise follow the warmstarting reference
            else:
                cost = cost + rho*cp.norm(s[:,-1] - xs_end_ref, 2)
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # SolveOSQP problem
        prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status == 'infeasible':
            print("[solve]: Problem infeasible.")
            s_opt = None
            a_opt = None
            J = None
        else:
            s_opt = s.value.T
            a_opt = a.value.T
            J = prob.value

        return s_opt, a_opt, J, prob.status


class MyopicConvexMPC():
    '''
    Class to perform trajectory optimization in a closed-loop MPC fashion. The non-linear trajectory optimization problem is solved as an SCP problem
    warm-started using a myopic and local version of the true problem.

    Inputs:
        - n_steps : the number of steps used for the horizon of the MPC
        - scp_mode : ('hard'/'soft') string to select whether to consider the waypoint constraints as hard or soft in the scp.
    
    Public methods:
        - warmstart : compute the warm-start for the trajectory over the planning horizon using the convexified version of the problem;
        - solve_scp : uses the SCP to optimize the trajectory over the planning horizon starting from the warm-starting trajectory.
    '''
    # SCP data
    iter_max_SCP = quad.iter_max_SCP # [-]
    trust_region0 = quad.trust_region0 # [m]
    trust_regionf = quad.trust_regionf # [m]
    J_tol = quad.J_tol # [N]

    ########## CONSTRUCTOR ##########
    def __init__(self, n_steps, scp_mode='hard'):

        # Save the number of steps to use for MPC
        self.n_steps = n_steps
        self.scp_mode = scp_mode
    
    ########## CLOSED-LOOP METHODS ##########
    def warmstart(self, current_obs, current_env:QuadEnv):
        '''
        Function to use the myopic and local optimization problem to predict the trajectory for the next self.n_steps, starting from the current environment.
        The terminal cost used in the cvx problem is computed following Riccati's equation.

        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : RpodEnv object containing the current information of the environment
               
        Outputs:
            - CVX_trajectory: dictionary contatining the predicted state and action trajectory over the next self.n_steps
        '''
        # Extract current information from the current_env
        t_i = current_env.timestep
        t_f = min(t_i + self.n_steps, current_env.n_time_rpod)
        t_cut = t_f - t_i
        n_time_remaining = current_env.n_time_rpod - t_i
        T_rem = n_time_remaining*current_env.dt
        # Extract real state from environment observation
        current_state = current_obs['state']
        current_goal = current_obs['goal']

        # Line warmstart from the quadrotor model -> Make sure the inputs have the correct dimensions (n_steps, n_state) and (n_steps, n_actions)
        states_ref = (current_state + ((current_goal - current_state)/T_rem)*np.arange(0,T_rem,current_env.dt)[:,None])[:t_cut,:]
        state_end_ref = current_goal
        actions_ref = np.zeros((t_cut,3))
        J_vect = np.ones(shape=(self.iter_max_SCP,), dtype=float)*1e12

        # Initial condition for the scp
        DELTA_J = 10
        trust_region = self.trust_region0
        beta_SCP = (self.trust_regionf/self.trust_region0)**(1/self.iter_max_SCP)
        runtime0_scp = time.time()
        runtime0_cvx = time.time()
        for scp_iter in range(self.iter_max_SCP):
            # Solve OCP (safe)
            try:
                states, actions, cost, feas = self.__ocp_scp_closed_loop(states_ref, actions_ref, current_state, current_goal, state_end_ref, t_i, t_f, current_env, trust_region, self.scp_mode, obs_av=False)
            except:
                states = None
                actions = None
                feas = 'infeasible'
            
            if not np.char.equal(feas,'infeasible'):
                J_vect[scp_iter] = cost
                # compute error
                L2_error = trust_region_diff(states, states_ref) #L2_error_us(us, us_prev)
                if scp_iter > 0:
                    DELTA_J = cost_prev - cost

                # Update iterations
                states_ref = states
                actions_ref = actions
                cost_prev = cost
                trust_region = beta_SCP*trust_region
                if scp_iter >= 1 and (L2_error <= self.trust_regionf or abs(DELTA_J) < self.J_tol):
                    break
            else:
                print(feas)
                print('unfeasible scp') 
                break
        runtime1_cvx = time.time()
        runtime_cvx = runtime1_cvx - runtime0_cvx

        # Keep only the next self.n_steps for the output
        CVX_trajectory = {
            'state' : states.T,
            'dv' : actions.T,
            'time' : np.arange(t_i, t_f)*current_env.dt,
        }
        cvx_dict = {
            'feas' : feas,
            'runtime_scp' : runtime_cvx
        }

        return CVX_trajectory
    
    def solve_scp(self, current_obs, current_env:QuadEnv, states_ref, actions_ref):
        '''
        Function to solve the scp optimization problem of MPC linearizing non-linear constraints with the reference trajectory provided in input.
        
        Inputs:
            - current_obs: dictionary containing the current observation taken from the current environment, used to initialize the prediction and the context
            - current_env : RpodEnv object containing the current information of the environment
            - states_ref : (6xself.n_steps) state trajectory over the next self.n_steps time instants to be used for linearization
            - actions_ref : (3xself.n_steps) action trajectory over the next self.n_steps time instants to be used for linearization
               
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
        current_state = current_obs['state']
        current_goal = current_obs['goal']

        # Makse sure the inputs have the correct dimensions (n_steps, n_state) and (n_steps, n_actions)
        states_ref = states_ref.T
        state_end_ref = current_goal
        actions_ref = actions_ref.T
        J_vect = np.ones(shape=(self.iter_max_SCP,), dtype=float)*1e12
        
        # Initial condition for the scp
        DELTA_J = 10
        trust_region = self.trust_region0
        beta_SCP = (self.trust_regionf/self.trust_region0)**(1/self.iter_max_SCP)
        runtime0_scp = time.time()
        for scp_iter in range(self.iter_max_SCP):
            '''print("scp_iter =", scp_iter)'''
            # Solve OCP (safe)
            try:
                states, actions, cost, feas = self.__ocp_scp_closed_loop(states_ref, actions_ref, current_state, current_goal, state_end_ref, t_i, t_f, current_env, trust_region, self.scp_mode)
            except:
                states = None
                actions = None
                feas = 'infeasible'
            
            if not np.char.equal(feas,'infeasible'):
                J_vect[scp_iter] = cost
                # compute error
                L2_error = trust_region_diff(states, states_ref) #L2_error_us(us, us_prev)
                if scp_iter > 0:
                    DELTA_J = cost_prev - cost

                # Update iterations
                states_ref = states
                actions_ref = actions
                cost_prev = cost
                trust_region = beta_SCP*trust_region
                if scp_iter >= 1 and (L2_error <= self.trust_regionf and abs(DELTA_J) < self.J_tol):
                    break
            else:
                print(feas)
                print('unfeasible scp') 
                break
        runtime1_scp = time.time()
        runtime_scp = runtime1_scp - runtime0_scp

        SCPMPC_trajectory = {
            'state' : states.T if not np.char.equal(feas,'infeasible') else None,
            'dv' : actions.T if not np.char.equal(feas,'infeasible') else None
        }
        scp_dict = {
            'feas' : feas,
            'iter_scp' : scp_iter,
            'J_vect' : J_vect,
            'runtime_scp' : runtime_scp
        }

        return SCPMPC_trajectory, scp_dict

    ########## STATIC METHODS ##########
    @staticmethod
    def __ocp_scp_closed_loop(xs, us, x_init, x_final, xs_end_ref, t_i, t_f, env:QuadEnv, trust_region, scp_mode, obs_av=True):
        # IMPORTANT: xs and us are the references and must be of shape (n_steps,n_state) and (n_steps,n_actions)
        # Setup SQP problem
        xs, us = xs.T, us.T
        n_time = xs.shape[1]
        qm = env.quad_model

        s = cp.Variable((6, n_time))
        a = cp.Variable((3, n_time))

        if scp_mode == 'hard':
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == x_init]
            # Dynamics
            A = np.array([[0, 0, 0, 1, 0, 0.],
                        [0, 0, 0, 0, 1, 0.],
                        [0, 0, 0, 0, 0, 1.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.]])
            B = 1/env.mass*np.vstack((np.zeros((3,3)),np.eye(3)))
            constraints += [s[:,i+1] == (s[:,i] + ((A + qm.A_d(xs[:,i])) @ s[:,i] + qm.b_d(xs[:,i]) + B @ a[:,i])*env.dt) for i in range(n_time-1)]
            # Terminal Condition
            if t_f == env.n_time_rpod:
                constraints += [s[:,-1] + ((A + qm.A_d(xs[:,-1])) @ s[:,-1] + qm.b_d(xs[:,-1]) + B @ a[:,-1])*env.dt == x_final]
            # Control contraints
            constraints += [a >= -env.u_max]
            constraints += [a <= env.u_max]
            # Keep-out-zone plus trust region
            if obs_av:
                for k in range(n_time):
                    # KOZ
                    for n_obs in range(len(env.obs_radii)):
                        c_koz_k = np.transpose(xs[:3,k] - env.obs_positions[n_obs,:]).dot(np.eye(3)/(env.obs_radii[n_obs]**2))
                        b_koz_k = np.sqrt(c_koz_k.dot(xs[:3,k] - env.obs_positions[n_obs,:]))
                        constraints += [c_koz_k @ (s[:3,k] - env.obs_positions[n_obs,:]) >= b_koz_k]
                    # Trust region
                    b_soc_k = -xs[:,k]
                    constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
            else:
                for k in range(n_time):
                    # Trust region
                    b_soc_k = -xs[:,k]
                    constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
            
            # Compute Cost
            rho = 0.08
            cost = 0.5*cp.sum(cp.norm(a, 2, axis=0)**2)
            if t_f < env.n_time_rpod:
                cost = cost + rho*cp.norm(s[:,-1] - xs_end_ref, 2)

        else:
            # Compute Constraints
            constraints = []
            # Initial Condition
            constraints += [s[:,0] == x_init]
            # Dynamics
            A = np.array([[0, 0, 0, 1, 0, 0.],
                        [0, 0, 0, 0, 1, 0.],
                        [0, 0, 0, 0, 0, 1.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.],
                        [0, 0, 0, 0, 0, 0.]])
            B = 1/env.mass*np.vstack((np.zeros((3,3)),np.eye(3)))
            constraints += [s[:,i+1] == (s[:,i] + ((A + qm.A_d(xs[:,i])) @ s[:,i] + qm.b_d(xs[:,i]) + B @ a[:,i])*env.dt) for i in range(n_time-1)]
            
            # Control contraints
            constraints += [a >= -env.u_max]
            constraints += [a <= env.u_max]
            # Keep-out-zone plus trust region --> Imposed for all the timestep except the current one (since we can't modify the current state)
            if obs_av:
                for k in range(1,n_time):
                    # KOZ
                    for n_obs in range(len(env.obs_radii)):
                        c_koz_k = np.transpose(xs[:3,k] - env.obs_positions[n_obs,:]).dot(np.eye(3)/(env.obs_radii[n_obs]**2))
                        b_koz_k = np.sqrt(c_koz_k.dot(xs[:3,k] - env.obs_positions[n_obs,:]))
                        constraints += [c_koz_k @ (s[:3,k] - env.obs_positions[n_obs,:]) >= b_koz_k]
                    # Trust region
                    b_soc_k = -xs[:,k]
                    constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
            else:
                for k in range(1,n_time):
                    # Trust region
                    b_soc_k = -xs[:,k]
                    constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
            
            # Compute Cost
            rho = 0.08
            cost = 0.5*cp.sum(cp.norm(a, 2, axis=0)**2)
            # Goal reaching penalizing term: if the end of the maneuver is already in the planning horizon aim for the goal
            if t_f == env.n_time_rpod:
                cost = cost + 10*cp.norm((s[:,-1] + ((A + qm.A_d(xs[:,-1])) @ s[:,-1] + qm.b_d(xs[:,-1]) +  B @ a[:,-1])*env.dt) - x_final, 2)
            
            # Otherwise follow the warmstarting reference
            else:
                cost = cost + rho*cp.norm(s[:,-1] - xs_end_ref, 2)
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # SolveOSQP problem
        prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status == 'infeasible':
            print("[solve]: Problem infeasible.")
            s_opt = None
            a_opt = None
            J = None
        else:
            s_opt = s.value.T
            a_opt = a.value.T
            J = prob.value

        return s_opt, a_opt, J, prob.status

