import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_folder)

import numpy as np
import copy
import dynamics.orbit_dynamics as dyn
import matplotlib.pyplot as plt
from optimization.rpod_scenario import dock_param_maker

'''
    TODO:
        - step():
            1) should we keep an history of the constraints violations too? Like checking at each step if each of the constraint is violated or not
            2) How to handle the last step of the trajectory (t==n_time_ropd)?
            3) Is the reward time index correct?
'''

class RpodEnv():

    ########## CONSTRUCTOR ##########
    def __init__(self, rpod_label='ISS',
                 constraints={
                     'docking_waypoint' : True,
                     'approach_cone' : True,
                     'keep_out_zone' : True,
                     'plume_impingement' : False,
                     'actions_bounding_box' : False,
                     'target_pointing' : False
                 }):    
        
        # Import the correct RPOD from the catalogue
        self.__get_rpod_scenario(rpod_label, constraints)

        # Save the contraints to be used
        self.constraints = constraints

        # Initialize history vectors
        self.__initialize_history()


    ########## RESET METHODS ##########
    def reset(self, reset_mode, reset_condition=None, dataloader=None, idx_sample=None, return_sample=False):
        '''
        This function is used to set the initial condition of the deputy and the maneuver parameters. It is expected to be used BEFORE any call to the step method.
        Different kind of reset may be executed through on the following innputs:
            - reset_mode: 'det' -> deterministic reset which uses reset_condition to reset
                          'rsamp' -> random sample performed from the dataloader to reset
                          'dsamp' -> deterministic sample of the idx_sample from the dataloader to reset
            - reset_condtion: tuple (horizon, roe_0, dock_param) containing the deterministic condition to reset the environment, TO BE PROVIDED for 'det' mode
            - dataloader: dataloader containing the dataset from which the reset condition should be sample, TO BE PROVIDED for 'rsamp' AND 'dsamp' modes
            - idx_sample: index of the sample to be extracted from the dataloader to reset the environment, TO BE PROVIDED for 'dsamp' mode
            - return_sample: boolean input, set to True if the data sample used for the reset should be returned (in case of 'rsamp' and 'dsamp' modes)
        '''

        # Reset the condition
        self.__initialize_history()

        # Select the correct reset mode
        if reset_mode[1:] == 'samp':
            # Random sample or deterministic extraction from the dataloader provided
            if reset_mode == 'rsamp':
                traj_sample = next(iter(dataloader))
            elif reset_mode == 'dsamp':
                traj_sample = dataloader.dataset.getix(idx_sample)
            
            # Extract the values form the sample
            if dataloader.dataset.mdp_constr:
                states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = traj_sample
            else:
                states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = traj_sample
            data_stats = dataloader.dataset.data_stats
            dock_param, self.cone_plotting_param = dock_param_maker(np.array((goal_i[0, 0, :] * data_stats['goal_std'][0]) + data_stats['goal_mean'][0]))

            # Time characteristics and discretization of the manuever
            self.hrz = horizons.item()
            self.dt_hrz = self.hrz*self.period_ref/(self.n_time_rpod - 1)

            # Fill initial conditions
            self.__load_oe(self.oe_0_ref)
            self.__load_state(np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0]), dataloader.dataset.state_representation)
            self.__load_goal(dock_param)
            self.timestep = 0
            self.time = np.array([0.])

            # Evntually return the data sampled
            if return_sample:
                return traj_sample

        elif reset_mode == 'det':
            # Time characteristics and discretization of the manuever
            self.hrz = reset_condition[0]
            self.dt_hrz = self.hrz*self.period_ref/(self.n_time_rpod - 1)

            # Fill initial conditions
            self.__load_oe(self.oe_0_ref)
            self.__load_state(reset_condition[1], state_representation='roe')
            self.__load_goal(reset_condition[2])
            _, self.cone_plotting_param = dock_param_maker(reset_condition[2]['state_rtn_target'])
            self.timestep = 0
            self.time = np.array([0.])

        else:
            raise NameError('Reset mode not found.')
    
    def __initialize_history(self):
        '''
        Function to initialize the history vectors of the environment storing reference orbital elements, states, actions, DCMs, rewards and time.
        '''
        self.timestep = -1 # <0 means no reset has been done
        self.time = np.empty((0, ))
        self.oe = np.empty((6, 0)) # true absolute state
        self.state_roe = np.empty((6, 0)) # true relative state
        self.state_rtn = np.empty((6, 0)) # true relative state
        self.dock_param = np.empty((0,), dtype=object) # docking parameters dictionary
        self.dv_rtn = np.empty((3, 0)) # true applied action
        self.dv_body = np.empty((3, 0)) # true applied action
        self.dv_com_rtn = np.empty((3, 0)) # commanded action
        self.dv_com_body = np.empty((3, 0)) # commanded action
        self.DCM_rtn_b = np.empty((3, 3, 0))
        self.reward = np.empty((0, ))

        self.__initialize_predictions()

    def __initialize_predictions(self):
        '''
        Function to initialize the history of the environment for plotting purposes.
        '''
        self.pred_history = []


    ########## PROPAGATION METHODS ##########
    def step(self, action, action_RF = 'rtn', dock_param=None):
        '''
        Function to compute one step forward in the environment and return the corresponding observation and reward.
        Inputs:
            - actions: np.array with shape (3,) containing the action to execute at the current timestep
            - actions_RF: label identifying the RF for the actions vector provided ('rtn'/'body')
            - dock_param : dictionary containing the current information about the docking port (if None -> maintain the current)
        '''

        # Check that the current timestep
        if self.timestep < 0:
            raise RuntimeError('The environment has never been reset to an initial condition!')
        
        elif self.timestep > self.n_time_rpod-1:
            raise RuntimeError('The environment has reached the end of the time horizon considered!')
        
        else:
            # Load the actions in the history
            self.__load_action(action, action_RF)
            
            # Load the goal in the history
            if dock_param == None:
                self.__load_goal(copy.deepcopy(self.dock_param[-1]))
            else:
                self.__load_goal(dock_param)
            
            # Propagate the dynamics
            self.__propagate_dynamics()
           
            # Get the reward and load it into the history vector
            current_reward = self.__get_reward()
            self.__load_reward(current_reward)

            # Propagate time and index
            self.__propagate_time()
            self.timestep += 1

            # Get the observation
            #current_observation = self.get_observation()

            return current_reward#current_observation, 
    
    def __propagate_time(self):
        '''
        Function to propagate the time.
        '''
        # Update the time
        self.time = np.hstack((self.time, self.time[-1] + self.dt_hrz))

    def __propagate_dynamics(self):
        '''
        Function that computes the propagation of the dynamics associated to the rpod environment for the timestep required. If the timestep is the last one, the final position of the maneuver is computed.
        '''
        # Check the correct length of the timeseries
        time_state = self.state_roe.shape[1]
        time_action = self.dv_body.shape[1]
        time_oe = self.oe.shape[1]

        if (time_state == time_action) and (time_state == time_oe):
            
            # Propagate relative orbital elements and reference orbital elements
            new_state_roe = dyn.dynamics(self.state_roe[:, -1], self.dv_rtn[:, -1], self.oe[:, -1], self.dt_hrz)
            new_oe = dyn.kepler_dyanmics(self.oe[:,-1], self.dt_hrz)
            
            # Update
            self.__load_oe(new_oe)
            self.__load_state(new_state_roe, state_representation='roe')
        
        else:
            raise RuntimeError('Trying to propagate dyanmics with state at time index', time_state, ', action at time index', time_action, 'and reference orbital element at time index', time_oe)


    ########## ACTIVE CONSTRAINTS METHODS ##########
    def get_active_constraints(self, timestep_i=None, timestep_f=None):
        '''
        Function to get the list of constraints active for a specific timesteps interval. In case no timesteps are provided, the function returns the active constraints from the current timestep to the end of the maneuver.
        '''
        # Initialize the dictionary
        constraints_active_range = {
            'docking_waypoint' : np.zeros((0),int),
            'approach_cone' : np.zeros((0),int),
            'keep_out_zone' : np.zeros((0),int),
            'plume_impingement' : np.zeros((0),int),
            'actions_bounding_box' : np.zeros((0),int),
            'target_pointing' : np.zeros((0),int),
            'docking_port' : np.zeros((0),int)
        }
        # If no timesteps are provided, compute the active constraints to the current timestep to the end of the maneuver
        if timestep_i == None:
            timestep_i = self.timestep
        if timestep_f == None:
            timestep_f = self.n_time_rpod

        # Fill the values with the inital and final timestep in which each constraint is valid for the timestep interval provided
        if self.constraints['docking_waypoint'] and timestep_i != self.dock_wyp_sample:
            constraints_active_range['docking_waypoint'] = self.__check_constraints_interval(self.dock_wyp_sample, self.dock_wyp_sample+1, timestep_i, timestep_f)
        
        if self.constraints['approach_cone']:
            constraints_active_range['approach_cone'] = self.__check_constraints_interval(self.dock_wyp_sample, self.n_time_rpod, timestep_i, timestep_f)
        
        if self.constraints['keep_out_zone']:
            constraints_active_range['keep_out_zone'] = self.__check_constraints_interval(0, self.dock_wyp_sample, timestep_i, timestep_f)
        
        if self.constraints['plume_impingement']:
            constraints_active_range['plume_impingement'] = self.__check_constraints_interval(self.dock_wyp_sample, self.n_time_rpod, timestep_i, timestep_f)
        
        if self.constraints['actions_bounding_box']:
            constraints_active_range['actions_bounding_box'] = np.arange(0, timestep_f - timestep_i)
        
        if self.constraints['target_pointing']:
            constraints_active_range['target_pointing'] = np.arange(0, timestep_f - timestep_i)
        
        if timestep_f >= self.n_time_rpod:
            constraints_active_range['docking_port'] = np.array([self.n_time_rpod - timestep_i - 1])
        
        return constraints_active_range

    def __check_constraints_interval(self, constr_i, constr_f, time_i, time_f):
        '''
        Function to check whether a constraints (active in constr_i->constr_f) is active during the time interval time_i->time_f. Void list is returned if no overlapping time interval is detected.
        '''
        if time_i <= constr_i <= time_f and time_i <= constr_f <= time_f:
            t_active_i = constr_i - time_i
            t_active_f = constr_f - time_i
            active_range = np.arange(t_active_i, t_active_f)
        elif time_i <= constr_i <= time_f and constr_f > time_f:
            t_active_i = constr_i - time_i
            t_active_f = time_f - time_i
            active_range = np.arange(t_active_i, t_active_f)
        elif constr_i < time_i and time_i <= constr_f <= time_f:
            t_active_i = time_i - time_i
            t_active_f = constr_f - time_i
            active_range = np.arange(t_active_i, t_active_f)
        elif constr_i < time_i and constr_f > time_f:
            t_active_i = time_i - time_i
            t_active_f = time_f - time_i
            active_range = np.arange(t_active_i, t_active_f)
        else:
            active_range = np.zeros((0),int)
        
        return active_range

    def get_active_dynamic_constraints(self, oe_obs_i, timestep_i=None, timestep_f=None):
        '''
        Function to get the dyanmics constraints (stm, cim, psi) for a specific timesteps interval. In case no timesteps are provided, the function returns the matrices from the current timestep to the end of the maneuver. 
        '''
        # If no timesteps are provided, compute the active constraints to the current timestep to the end of the maneuver
        if timestep_i == None:
            timestep_i = self.timestep
        if timestep_f == None:
            timestep_f = self.n_time_rpod

        # Compute the horizon interval an the timestep interval
        time_i = self.dt_hrz*timestep_i
        time_f = self.dt_hrz*timestep_f
        timestep_int = timestep_f - timestep_i
        hrz_int = (timestep_int - 1)*self.dt_hrz/self.period_ref

        # Dynamics Matrices Precomputations
        stm_int, cim_int, psi_int, oe_int, time_int, dt_int = dyn.dynamics_roe_optimization(oe_obs_i, time_i, hrz_int, timestep_int)

        return stm_int, cim_int, psi_int


    ########## GET METHODS ##########
    def __get_rpod_scenario(self,
                            rpod_label='ISS',
                            constraints= {
                                'docking_waypoint' : True,
                                'approach_cone' : True,
                                'keep_out_zone' : True,
                                'plume_impingement' : False,
                                'actions_bounding_box' : False,
                                'target_pointing' : False
                            }):
        '''
        This functiong loads into the environment the requested rpod scenario and maneuver constraints.
        '''

        if rpod_label == 'ISS':
            from optimization.rpod_scenario import oe_0_ref, period_ref, n_time_rpod, D_pos, DEED_koz, x_ell, y_ell, z_ell, plume_imp_angle, dv_max, dock_wyp_sample
            # ISS Reference orbit parameters
            self.oe_0_ref = oe_0_ref
            self.period_ref = period_ref

            # Transfer Horizons Set
            self.n_time_rpod = n_time_rpod # Number of time sample in transfer horizon
            self.dock_wyp_sample = dock_wyp_sample

            # Save keep-out-zone parameters only if requested
            if constraints['keep_out_zone']:
                self.D_pos = D_pos
                self.DEED_koz = DEED_koz
                # Keep-out-zone Ellipse (for plotting)
                self.x_ell = x_ell
                self.y_ell = y_ell
                self.z_ell = z_ell

            # Save plume impingement parameters only if requested
            if constraints['plume_impingement']:
                self.plume_imp_angle = plume_imp_angle

            # Save actions bounding box parameters only if requested
            if constraints['actions_bounding_box']:
                self.dv_max = dv_max
        
        else:
            raise NameError('RPOD scenario not identified.')

    def __get_reward(self):
        '''
        Function to provide the reward obtained at the current timestep.
        '''
        # Check the correct length of the timeseries
        time_action = self.dv_body.shape[1]
        time_reward = self.reward.shape[0] + 1
        if time_reward == time_action:
            # Compute reward      
            new_reward = -np.linalg.norm(self.dv_body[:, -1])
            return new_reward
        
        else:
            raise RuntimeError('Trying to compute reward at time index', time_reward, 'with action at time index', time_action)
    
    def get_observation(self):
        '''
        Function to get the current observation vector from the environment. When the function is called, the action should have already been performed, hence: len(action_history) == len(state_history) - 1.
        The output is a dictionary containing: 
            - 'time' : current time,
            - 'state_roe'/'state_rtn' : current value of the state in both representation
            - 'dv_rtn'/'dv_body' : latest action performed (so the action corresponding to the previous timestep)
            - 'DCM_rtn_b' : current rotation matrix from body to rtn RF
            - 'dock_param' : dictionary containing the current information about the docking port
        '''
        # Check the correct length of the timeseries
        time_action = self.dv_com_body.shape[1]
        time_state = self.state_roe.shape[1]

        if time_action == time_state - 1:
            observation = {
                'time' : self.time[-1],
                'state_roe' : (self.state_roe[:, -1]).copy(),
                'state_rtn' : (self.state_rtn[:, -1]).copy(),
                'dv_rtn' : (self.dv_com_rtn[:, -1] if time_action > 0 else np.array([])).copy(),
                'dv_body' : (self.dv_com_body[:, -1] if time_action > 0 else np.array([])).copy(),
                'DCM_rtn_b' : (self.DCM_rtn_b[:, :, -1]).copy(),
                'oe' : (self.oe[:, -1]).copy(),
                'dock_param' : copy.deepcopy(self.dock_param[-1])
            }
            return observation
        
        else:
            raise RuntimeError('Trying to get observation with state time index', time_state, 'and action time index', time_action)


    ########## LOAD METHODS ##########
    def __load_oe(self, oe):
        '''
        Function to load into the orbital elements history vector the orbital elements provided.
        '''
        # Update the orbital elements vector
        self.oe = np.hstack((self.oe, oe.reshape(6,1)))

    def __load_state(self, state, state_representation):
        '''
        Function to load into the state history vectors (both as rtn and roe) the provided state, using the provided orbital elements as reference.
        '''
        # Check the correct length of the timeseries
        time_oe = self.oe.shape[1]
        time_state = self.state_roe.shape[1] + 1
        if time_oe == time_state:
            # Update the state history
            if state_representation == 'roe':
                self.state_roe = np.hstack((self.state_roe, state.reshape(6,1)))
                self.state_rtn = np.hstack((self.state_rtn, dyn.map_roe_to_rtn(self.state_roe[:,-1], self.oe[:,-1]).reshape(6,1)))
            elif state_representation == 'rtn':
                self.state_rtn = np.hstack((self.state_rtn, state.reshape(6,1)))
                self.state_roe = np.hstack((self.state_roe, dyn.map_rtn_to_roe(self.state_rtn[:,-1], self.oe[:,-1]).reshape(6,1)))
            
            # Update the rotation matrix
            if self.constraints['target_pointing']:
                self.DCM_rtn_b = np.concatenate((self.DCM_rtn_b, dyn.map_body_to_rtn(self.state_rtn[:, -1], self.dock_port)[:,:,None]), axis=2)
            else:
                self.DCM_rtn_b = np.concatenate((self.DCM_rtn_b, np.eye(3)[:,:,None]), axis=2)
        
        else:
            raise RuntimeError('Trying to update state and Direct Cosine Matrix at time index', time_state, 'with reference orbital element at time index', time_oe)

    def __load_goal(self, dock_param):
        '''
        Function to load into the dock_param history vectors the provided dock_param.
        '''
        # Check the correct length of the timeseries
        time_steps = self.time.shape[0]
        time_goal = self.dock_param.shape[0]
        if time_steps == time_goal:
            # Update the dock_param history
            self.dock_param = np.hstack((self.dock_param, dock_param))
        
        else:
            raise RuntimeError('Trying to update dock_param at time index', time_goal, 'with time at time index', time_steps)

    def __load_action(self, action, action_RF):
        '''
        Function the load the action into the action history vectors, computed in rtn RF and body RF according to the presence of target poiting requirements.
        '''
        # Check the correct length of the timeseries
        time_DCM = self.DCM_rtn_b.shape[2]
        time_action = self.dv_body.shape[1] + 1

        if time_DCM == time_action:
            # Compute action in both RFs        
            if action_RF == 'body':
                action_body = action.reshape(3,1)
                action_rtn = self.DCM_rtn_b[:, :, -1] @ action_body
                action_com_body = action.reshape(3,1)
                action_com_rtn = self.DCM_rtn_b[:, :, -1] @ action_com_body
            elif action_RF == 'rtn':
                action_rtn = action.reshape(3,1)
                action_body = self.DCM_rtn_b[:, :, -1].T @ action_rtn
                action_com_rtn = action.reshape(3,1)
                action_com_body = self.DCM_rtn_b[:, :, -1].T @ action_com_rtn

            # Update the actiojn history
            self.dv_body = np.hstack((self.dv_body, action_body))
            self.dv_rtn = np.hstack((self.dv_rtn, action_rtn))
            self.dv_com_body = np.hstack((self.dv_com_body, action_com_body))
            self.dv_com_rtn = np.hstack((self.dv_com_rtn, action_com_rtn))
        
        else:
            raise RuntimeError('Trying to update action at time index', time_action, 'with Direct Cosine Matrix at time index', time_DCM)

    def __load_reward(self, reward):
        '''
        Function to load into the reward history vector the provided reward.
        '''
        # Update the rewards vector
        self.reward = np.hstack((self.reward, reward))

    def load_prediction(self, ART_trajectory, ARTMPC_trajectory):
        instant_pred = {}
        instant_pred['state_roe_ART'] = ART_trajectory['state_roe']
        instant_pred['state_rtn_ART'] = ART_trajectory['state_rtn']
        instant_pred['dv_rtn_ART'] = ART_trajectory['dv_rtn']
        instant_pred['state_roe_ARTMPC'] = ARTMPC_trajectory['state_roe']
        instant_pred['state_rtn_ARTMPC'] = ARTMPC_trajectory['state_rtn']
        instant_pred['dv_rtn_ARTMPC'] = ARTMPC_trajectory['dv_rtn']
        instant_pred['time_orb'] = ART_trajectory['time_orb']
        self.pred_history.append(instant_pred)


    ########## PLOT METHODS ##########
    def plot(self, plan=None, history=None, maneuver=None, ax=None, mpc_label='ART'):
        '''current_state_rtn = np.full((6,), np.nan) if current is None else current['rtn']
        current_time = np.full((1,), np.nan) if current is None else current['time']'''
        plan_rtn_ART = np.full((6, 1), np.nan) if plan is None else plan['state_rtn_ART']
        plan_rtn_ARTMPC = np.full((6, 1), np.nan) if plan is None else plan['state_rtn_ARTMPC']
        plan_time = np.full((1,), np.nan) if plan is None else plan['time_orb']
        history_rtn = np.full((6, 1), np.nan) if history is None else history['state_rtn']
        history_time = np.full((1,), np.nan) if history is None else history['time_orb']
        maneuver_rtn_ART = np.full((6, 1), np.nan) if maneuver is None else maneuver['state_rtn_ART']
        maneuver_rtn_scpART = np.full((6, 1), np.nan) if maneuver is None else maneuver['state_rtn_scpART']
        maneuver_time = np.full((1,), np.nan) if maneuver is None else maneuver['time_orb']

        if ax is None:
            fig, ax = plt.subplots(2,3,figsize=(20, 5))
            ART_lines, ARTMPC_lines, history_lines, maneuver_ART_lines, maneuver_ARTscp_lines = [], [], [], [], []
            for i in range(2):
                maneuver_ART_lines.extend([ax[i,j].plot(maneuver_time, maneuver_rtn_ART[3*i+j,:], color=[0.2,0.2,0.2], linewidth=0.5, label=mpc_label+'$_{total} $')[0] for j in range(3)])
                maneuver_ARTscp_lines.extend([ax[i,j].plot(maneuver_time, maneuver_rtn_scpART[3*i+j,:], color='r', linewidth=0.5, label='scp'+mpc_label+'$_{total} $')[0] for j in range(3)])
                ART_lines.extend([ax[i,j].plot(plan_time, plan_rtn_ART[3*i+j,:], color="b", linewidth=1.5, label=mpc_label)[0] for j in range(3)])
                ARTMPC_lines.extend([ax[i,j].plot(plan_time, plan_rtn_ARTMPC[3*i+j,:], color="g", linewidth=1.5, label=mpc_label+'-MPC')[0] for j in range(3)])
                history_lines.extend([ax[i,j].plot(history_time, history_rtn[3*i+j,:], color="k", linewidth=1.5, label='env')[0] for j in range(3)])
                if i == 0:
                    ylabels = ['$ \delta r_r$ [m]', '$ \delta r_t$ [m]', '$ \delta r_n$ [m]']
                elif i == 1:
                    ylabels = ['$ \delta v_r$ [m/s]', '$ \delta v_t$ [m/s]', '$ \delta v_n$ [m/s]']
                elif i == 2:
                    ylabels = ['$ \Delta \delta v_r$ [mm/s]', '$ \Delta \delta v_t$ [mm/s]', '$ \Delta \delta v_n$ [mm/s]']
                for j in range(3):
                    ax[i,j].grid(True)
                    ax[i,j].set_xlabel('time [orbits]', fontsize=10)
                    ax[i,j].set_ylabel(ylabels[j], fontsize=10)
                    ax[i,j].legend(loc='best', fontsize=10)
                    ax[i,j].set_xlim((0, self.hrz*1.1))
                    ax[i,j] = self.__dyn_set_ylim(ax[i,j])

        else:
            fig = ax[0,0].figure
            ART_lines = [ax[i,j].lines[2+0] for i in range(2) for j in range(3)]
            ARTMPC_lines = [ax[i,j].lines[2+1] for i in range(2) for j in range(3)]
            history_lines = [ax[i,j].lines[2+2] for i in range(2) for j in range(3)]

        [ART_lines[3*i+j].set_data(plan_time, plan_rtn_ART[3*i+j,:]) for i in range(2) for j in range(3)]
        [ARTMPC_lines[3*i+j].set_data(plan_time, plan_rtn_ARTMPC[3*i+j,:]) for i in range(2) for j in range(3)]
        [history_lines[3*i+j].set_data(history_time, history_rtn[3*i+j,:]) for i in range(2) for j in range(3)]

        return fig, ax

    def __dyn_set_ylim(self,ax):
        lines = ax.lines
        maximum_values = np.array([np.max(lines[i].get_data()[1]) if not np.isnan(np.max(lines[i].get_data()[1])) else -10**(12) for i in range(len(lines))])
        minimum_values = np.array([np.min(lines[i].get_data()[1]) if not np.isnan(np.min(lines[i].get_data()[1])) else 10**(12)for i in range(len(lines))])
        abs_max = np.max(maximum_values)
        abs_min = np.min(minimum_values)
        abs_mean = (abs_max + abs_min)/2
        abs_diff = (abs_max - abs_min)/2
        ax.set_ylim((abs_mean - 1.2*abs_diff, abs_mean + 1.2*abs_diff))
        return ax

    def plot3D(self, plan=None, history=None, maneuver=None, ax=None, mpc_label='ART'):
        '''current_state_rtn = np.full((6,), np.nan) if current is None else current['rtn']
        current_time = np.full((1,), np.nan) if current is None else current['time']'''
        plan_rtn_ART = np.full((6, 1), np.nan) if plan is None else plan['state_rtn_ART']
        plan_rtn_ARTMPC = np.full((6, 1), np.nan) if plan is None else plan['state_rtn_ARTMPC']
        plan_time = np.full((1,), np.nan) if plan is None else plan['time_orb']
        history_rtn = np.full((6, 1), np.nan) if history is None else history['state_rtn']
        history_time = np.full((1,), np.nan) if history is None else history['time_orb']
        maneuver_rtn_ART = np.full((6, 1), np.nan) if maneuver is None else maneuver['state_rtn_ART']
        maneuver_rtn_scpART = np.full((6, 1), np.nan) if maneuver is None else maneuver['state_rtn_scpART']
        maneuver_time = np.full((1,), np.nan) if maneuver is None else maneuver['time_orb']

        if ax is None:
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(projection='3d')
            ART_lines, ARTMPC_lines, history_lines, maneuver_ART_lines, maneuver_ARTscp_lines = [], [], [], [], []
            
            maneuver_ART_lines = ax.plot3D(maneuver_rtn_ART[1,:], maneuver_rtn_ART[2,:], maneuver_rtn_ART[0,:], color=[0.2,0.2,0.2], linewidth=0.5, label=mpc_label+'$_{total} $')[0]
            maneuver_ARTscp_lines = ax.plot3D(maneuver_rtn_scpART[1,:], maneuver_rtn_scpART[2,:], maneuver_rtn_scpART[0,:], color='r', linewidth=0.5, label='scp'+mpc_label+'$_{total} $')[0]
            pell = ax.plot_surface(self.y_ell, self.z_ell, self.x_ell, rstride=1, cstride=1, color='r', linewidth=0, alpha=0.3, label='keep-out-zone')
            pell._facecolors2d = pell._facecolor3d
            pell._edgecolors2d = pell._edgecolor3d
            pcone = ax.plot_surface(self.cone_plotting_param['t_cone'], self.cone_plotting_param['n_cone'], self.cone_plotting_param['r_cone'], rstride=1, cstride=1, color='g', linewidth=0, alpha=0.3, label='approach cone')
            pcone._facecolors2d = pcone._facecolor3d
            pcone._edgecolors2d = pcone._edgecolor3d
            ART_lines = ax.plot3D(plan_rtn_ART[1,:], plan_rtn_ART[2,:], plan_rtn_ART[0,:], color="b", linewidth=1.5, label=mpc_label)[0]
            ARTMPC_lines = ax.plot3D(plan_rtn_ARTMPC[1,:], plan_rtn_ARTMPC[2,:], plan_rtn_ARTMPC[0,:], color="g", linewidth=1.5, label=mpc_label+'-MPC')[0]
            history_lines = ax.plot3D(history_rtn[1,:], history_rtn[2,:], history_rtn[0,:], color="k", linewidth=1.5, label='env')[0]
            ax.set_xlabel('$\delta r_T$ [m]', fontsize=10)
            ax.set_ylabel('$\delta r_N$ [m]', fontsize=10)
            ax.set_zlabel('$\delta r_R$ [m]', fontsize=10)
            ax.grid(True)
            ax.legend(loc='best', fontsize=10)

        else:
            fig = ax.figure
            ART_lines = ax.lines[2+0]
            ARTMPC_lines = ax.lines[2+1]
            history_lines = ax.lines[2+2]

        ART_lines.set_data_3d(plan_rtn_ART[1,:], plan_rtn_ART[2,:], plan_rtn_ART[0,:])
        ARTMPC_lines.set_data_3d(plan_rtn_ARTMPC[1,:], plan_rtn_ARTMPC[2,:], plan_rtn_ARTMPC[0,:])
        history_lines.set_data_3d(history_rtn[1,:], history_rtn[2,:], history_rtn[0,:])

        return fig, ax