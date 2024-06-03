import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_folder)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from dynamics.freeflyer import FreeflyerModel, sample_init_target

'''
    TODO:
        - modify from ACTIVE CONSTRAINTS METHODS on
'''

class FreeflyerEnv():

    ########## CONSTRUCTOR ##########
    def __init__(self, PID=False):    
        
        # Import the correct RPOD from the catalogue
        self.__get_ff_scenario()
        self.PID = PID
 
        # Initialize history vectors
        self.__initialize_history()

    ########## RESET METHODS ##########
    def reset(self, reset_mode, reset_condition=None, dataloader=None, idx_sample=None, return_sample=False):
        '''
        This function is used to set the initial condition of the freeflyers and the maneuver parameters. It is expected to be used BEFORE any call to the step method.
        Different kind of reset may be executed through on the following innputs:
            - reset_mode: 'det' -> deterministic reset which uses reset_condition to reset
                          'rsamp' -> random sample performed from the dataloader to reset
                          'dsamp' -> deterministic sample of the idx_sample from the dataloader to reset
            - reset_condtion: tuple (dt, state_init, state_final) containing the deterministic condition to reset the environment, TO BE PROVIDED for 'det' mode
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
                states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = traj_sample
            else:
                states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = traj_sample
            data_stats = dataloader.dataset.data_stats

            # Time characteristics and discretization of the manuever
            self.dt = dt.item()

            # Fill initial conditions
            self.__load_state(np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0]))
            self.__load_goal(np.array((goal_i[0, 0, :] * data_stats['goal_std'][0]) + data_stats['goal_mean'][0]))
            self.timestep = 0
            self.time = np.array([0.])
            
            # Evntually return the data sampled
            if return_sample:
                return traj_sample

        elif reset_mode == 'det':
            # Time characteristics and discretization of the manuever
            self.dt = reset_condition[0]

            # Fill initial conditions
            self.__load_state(reset_condition[1])
            self.__load_goal(reset_condition[2])
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
        self.state = np.empty((self.N_STATE, 0)) # true relative state
        self.dv = np.empty((self.N_ACTION, 0)) # true applied action
        self.dv_t = np.empty((self.N_CLUSTERS,0)) # thursters deltaV
        self.goal = np.empty((self.N_STATE, 0)) # true goal
        self.reward = np.empty((0, ))

        self.__initialize_predictions()

    def __initialize_predictions(self):
        '''
        Function to initialize the history of the environment for plotting purposes.
        '''
        self.pred_history = []

    ########## PROPAGATION METHODS ##########
    def step(self, action=None, goal=None, state_desired=None):
        '''
        Function to compute one step forward in the environment and return the corresponding observation and reward.
        Inputs:
            - actions: np.array with shape (3,) containing the action to execute at the current timestep
            - goal : np.array with shape (6,) containing the current goal (if None -> maintain the current)
            - state_desired : step used for the feedback control term
        '''

        # Check that the current timestep
        if self.timestep < 0:
            raise RuntimeError('The environment has never been reset to an initial condition!')
        
        elif self.timestep > self.n_time_rpod-1:
            raise RuntimeError('The environment has reached the end of the time horizon considered!')
        
        else:
            # Load the actions in the history
            self.__load_action(action)

            # Load the goal in the history
            if goal == None:
                self.__load_goal(self.goal[:,-1].copy())
            else:
                self.__load_goal(goal)

            # Propagate the dynamics
            if self.PID:
                self.__propagate_dynamics_with_PID(state_desired)
            else:
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
        self.time = np.hstack((self.time, self.time[-1] + self.dt))

    def __propagate_dynamics(self):
        '''
        Function that computes the propagation of the dynamics associated to the freeflyers environment for the timestep required. If the timestep is the last one, the final position of the maneuver is computed.
        '''
        # Check the correct length of the timeseries
        time_step = self.time.shape[0]
        time_state = self.state.shape[1]
        time_action = self.dv.shape[1]

        if (time_state == time_action) and (time_state == time_step):
            
            # Propagate the quadrotor state
            new_state = self.ff_model.f_imp(self.state[:, -1], self.dv[:, -1])
            
            # Update
            self.__load_state(new_state)
        
        else:
            raise RuntimeError('Trying to propagate dyanmics with state at time index', time_state, ', action at time index', time_action, 'and time at time index', time_step)
    
    def __propagate_dynamics_with_PID(self, state_desired):
        '''
        Function that computes the propagation of the dynamics associated to the freeflyers environment for the timestep required. If the timestep is the last one, the final position of the maneuver is computed.
        Dynamics are propagated using a PID controller for trajectory tracking.
        '''
        # Check the correct length of the timeseries
        time_step = self.time.shape[0]
        time_state = self.state.shape[1]
        time_action = self.dv.shape[1]

        if (time_state == time_action) and (time_state == time_step):
            
            # Propagate the quadrotor state
            new_state = self.ff_model.f_PID(self.state[:, -1], state_desired)
            
            # Update
            self.__load_state(new_state)
        
        else:
            raise RuntimeError('Trying to propagate dyanmics with state at time index', time_state, ', action at time index', time_action, 'and time at time index', time_step)


    ########## GET METHODS ##########
    def __get_ff_scenario(self):
        '''
        This functiong loads into the environment the ff scenario and maneuver constraints.
        '''

        self.ff_model = FreeflyerModel()
        from optimization.ff_scenario import n_time_rpod, dt, obs, mass, inertia, robot_radius, safety_margin, table, F_max_per_thruster, T, Lambda, Lambda_inv, N_STATE, N_ACTION, N_CLUSTERS
        self.N_STATE = N_STATE
        self.N_ACTION = N_ACTION
        self.N_CLUSTERS = N_CLUSTERS
        self.mass = mass
        self.J = inertia
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.table = table
        self.F_t_M = F_max_per_thruster
        self.n_time_rpod = n_time_rpod
        self.dt = dt
        self.T = T
        self.obs = obs
        self.Lambda = Lambda
        self.Lambda_inv = Lambda_inv

    def __get_reward(self):
        '''
        Function to provide the reward obtained at the current timestep.
        '''
        # Check the correct length of the timeseries
        time_action = self.dv.shape[1]
        time_reward = self.reward.shape[0] + 1
        if time_reward == time_action:
            # Compute reward      
            new_reward = -np.linalg.norm(self.dv[:, -1], ord=1)
            return new_reward
        
        else:
            raise RuntimeError('Trying to compute reward at time index', time_reward, 'with action at time index', time_action)
    
    def get_observation(self):
        '''
        Function to get the current observation vector from the environment. When the function is called, the action should have already been performed, hence: len(action_history) == len(state_history) - 1.
        The output is a dictionary containing: 
            - 'time' : current time,
            - 'state' : current value of the state
            - 'dv' : latest action performed (so the action corresponding to the previous timestep)
        '''
        # Check the correct length of the timeseries
        time_action = self.dv.shape[1]
        time_action_t = self.dv_t.shape[1]
        time_state = self.state.shape[1]

        if time_action == time_state - 1:
            observation = {
                'time' : self.time[-1],
                'state' : self.state[:, -1].copy(),
                'dv' : (self.dv[:, -1] if time_action > 0 else np.array([])).copy(),
                'dv_t' : (self.dv_t[:, -1] if time_action_t > 0 else np.array([])).copy(),
                'goal' : self.goal[:,-1].copy()
            }
            return observation
        
        else:
            raise RuntimeError('Trying to get observation with state time index', time_state, 'and action time index', time_action)


    ########## LOAD METHODS ##########

    def __load_state(self, state):
        '''
        Function to load into the state history vectors (both as rtn and roe) the provided state, using the provided orbital elements as reference.
        '''
        # Check the correct length of the timeseries
        time_steps = self.time.shape[0]
        time_state = self.state.shape[1]
        if time_steps == time_state:
            # Update the state history
            self.state = np.hstack((self.state, state.reshape(self.N_STATE,1)))
        
        else:
            raise RuntimeError('Trying to update state at time index', time_state, 'with time at time index', time_steps)

    def __load_action(self, action):
        '''
        Function the load the action into the action history vectors.
        '''
        # Check the correct length of the timeseries
        time_steps = self.time.shape[0]
        time_state = self.state.shape[1]
        time_action = self.dv.shape[1] + 1
        time_action_t = self.dv_t.shape[1] + 1
        
        if (time_steps == time_state) and (time_steps == time_action) and (time_action == time_action_t):
            # Update the actiojn history
            action_t = self.Lambda_inv @ ((self.ff_model.R_BG(self.state[2,-1])) @ action)
            self.dv = np.hstack((self.dv, action.reshape(self.N_ACTION,1)))
            self.dv_t = np.hstack((self.dv_t, action_t.reshape(self.N_CLUSTERS,1)))
        
        else:
            raise RuntimeError('Trying to update action at time index', time_action, 'with time at time index', time_steps)

    def __load_goal(self, goal):
        '''
        Function to load into the goal history vectors the provided goal.
        '''
        # Check the correct length of the timeseries
        time_steps = self.time.shape[0]
        time_goal = self.goal.shape[1]
        if time_steps == time_goal:
            # Update the goal history
            self.goal = np.hstack((self.goal, goal.reshape(self.N_STATE,1)))
        
        else:
            raise RuntimeError('Trying to update goal at time index', time_goal, 'with time at time index', time_steps)

    def __load_reward(self, reward):
        '''
        Function to load into the reward history vector the provided reward.
        '''
        # Update the rewards vector
        self.reward = np.hstack((self.reward, reward))

    def load_prediction(self, ART_trajectory, ARTMPC_trajectory):
        instant_pred = {}
        instant_pred['state_ART'] = ART_trajectory['state']
        instant_pred['dv_ART'] = ART_trajectory['dv']
        instant_pred['state_ARTMPC'] = ARTMPC_trajectory['state']
        instant_pred['dv_ARTMPC'] = ARTMPC_trajectory['dv']
        instant_pred['time'] = ART_trajectory['time']
        self.pred_history.append(instant_pred)

    ########## PLOT METHODS ##########
    def plot(self, plan=None, history=None, maneuver=None, ax=None, mpc_label='ART'):
        '''current_state_rtn = np.full((6,), np.nan) if current is None else current['rtn']
        current_time = np.full((1,), np.nan) if current is None else current['time']'''
        plan_rtn_ART = np.full((6, 1), np.nan) if plan is None else plan['state_ART']
        plan_rtn_ARTMPC = np.full((6, 1), np.nan) if plan is None else plan['state_ARTMPC']
        plan_time = np.full((1,), np.nan) if plan is None else plan['time']
        history_rtn = np.full((6, 1), np.nan) if history is None else history['state']
        history_time = np.full((1,), np.nan) if history is None else history['time']
        maneuver_rtn_ART = np.full((6, 1), np.nan) if maneuver is None else maneuver['state_ART']
        maneuver_rtn_scpART = np.full((6, 1), np.nan) if maneuver is None else maneuver['state_scpART']
        maneuver_time = np.full((1,), np.nan) if maneuver is None else maneuver['time']

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
                    ylabels = ['X [m]', 'Y [m]', '$\psi$ [deg]']
                elif i == 1:
                    ylabels = ['$v_X$ [m/s]', '$v_Y$ [m/s]', '$\omega$ [deg/s]']
                elif i == 2:
                    ylabels = ['$\Delta v_X$ [mm/s]', '$Delta v_Y$ [mm/s]', '$\Delta \omega$ [mm/s]']
                for j in range(3):
                    ax[i,j].grid(True)
                    ax[i,j].set_xlabel('time', fontsize=10)
                    ax[i,j].set_ylabel(ylabels[j], fontsize=10)
                    ax[i,j].legend(loc='best', fontsize=10)
                    ax[i,j].set_xlim((0, self.T*1.1))
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
        plan_rtn_ART = np.full((6, 1), np.nan) if plan is None else plan['state_ART']
        plan_rtn_ARTMPC = np.full((6, 1), np.nan) if plan is None else plan['state_ARTMPC']
        plan_time = np.full((1,), np.nan) if plan is None else plan['time']
        history_rtn = np.full((6, 1), np.nan) if history is None else history['state']
        history_time = np.full((1,), np.nan) if history is None else history['time']
        maneuver_rtn_ART = np.full((6, 1), np.nan) if maneuver is None else maneuver['state_ART']
        maneuver_rtn_scpART = np.full((6, 1), np.nan) if maneuver is None else maneuver['state_scpART']
        maneuver_time = np.full((1,), np.nan) if maneuver is None else maneuver['time']
        
        if ax is None:
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot()
            ART_lines, ARTMPC_lines, history_lines, maneuver_ART_lines, maneuver_ARTscp_lines = [], [], [], [], []
            
            maneuver_ART_lines = ax.plot(maneuver_rtn_ART[0,:], maneuver_rtn_ART[1,:], color=[0.2,0.2,0.2], linewidth=0.5, label=mpc_label+'$_{total} $', zorder=3)[0]
            maneuver_ARTscp_lines = ax.plot(maneuver_rtn_scpART[0,:], maneuver_rtn_scpART[1,:], color='r', linewidth=0.5, label='scp'+mpc_label+'$_{total} $', zorder=3)[0]
            ax.add_patch(Rectangle((0,0), self.table['xy_up'][0], self.table['xy_up'][1], fc=(0.5,0.5,0.5,0.2), ec='k', label='table', zorder=2.5))
            for n_obs in range(self.obs['radius'].shape[0]):
                label_obs = 'obs' if n_obs == 0 else None
                label_robot = 'robot radius' if n_obs == 0 else None
                ax.add_patch(Circle(self.obs['position'][n_obs,:], self.obs['radius'][n_obs], fc='r', label=label_obs, zorder=2.5))
                ax.add_patch(Circle(self.obs['position'][n_obs,:], self.obs['radius'][n_obs]+self.robot_radius, fc='r', alpha=0.2, label=label_robot, zorder=2.5))
            ART_lines = ax.plot(plan_rtn_ART[0,:], plan_rtn_ART[1,:], color="b", linewidth=1.5, label=mpc_label, zorder=3)[0]
            ARTMPC_lines = ax.plot(plan_rtn_ARTMPC[0,:], plan_rtn_ARTMPC[1,:], color="g", linewidth=1.5, label=mpc_label+'-MPC', zorder=3)[0]
            history_lines = ax.plot(history_rtn[0,:], history_rtn[1,:], color="k", linewidth=1.5, label='env', zorder=3)[0]
            ax.set_xlabel('X [m]', fontsize=10)
            ax.set_ylabel('Y [m]', fontsize=10)
            ax.grid(True)
            ax.legend(loc='best', fontsize=10)

        else:
            fig = ax.figure
            ART_lines = ax.lines[2+0]
            ARTMPC_lines = ax.lines[2+1]
            history_lines = ax.lines[2+2]

        ART_lines.set_data(plan_rtn_ART[0,:], plan_rtn_ART[1,:])
        ARTMPC_lines.set_data(plan_rtn_ARTMPC[0,:], plan_rtn_ARTMPC[1,:])
        history_lines.set_data(history_rtn[0,:], history_rtn[1,:])

        return fig, ax
