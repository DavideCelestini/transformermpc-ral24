import numpy as np
import cvxpy as cp

from scipy import sparse
from scipy.sparse import csr_matrix, vstack, hstack, eye

from functools import partial

from optimization.quad_scenario import *

# -----------------------------------------
class QuadModel:
    def __init__(self, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print("Initializing drone class.")

    # ---------------------------------
    # Optimization variable z
    #   z = (xs_vec, us_vec)
    # where
    # - xs_vec is of shape (S+1)*n_x
    # ----- nominal state trajectory
    # - us_vec is of shape S*n_u
    # ----- nominal control trajectory
    #@partial(jit, static_argnums=(0,))
    def convert_z_to_variables(self, z):
        xs_vec = z[:((S+1)*n_x)]
        us_vec = z[((S+1)*n_x):]
        return xs_vec, us_vec

    #@partial(jit, static_argnums=(0,))
    def convert_xs_vec_to_xs_mat(self, xs_vec):
        xs_mat = np.reshape(xs_vec, (n_x, S+1), 'F')
        xs_mat = xs_mat.T # (S+1, n_x)
        return xs_mat

    #@partial(jit, static_argnums=(0,))
    def convert_us_vec_to_us_mat(self, us_vec):
        us_mat = np.reshape(us_vec, (n_u, S), 'F')
        us_mat = us_mat.T # (S, n_u)
        return us_mat

    #@partial(jit, static_argnums=(0,))
    def convert_z_to_xs_us_mats(self, z):
        xs_vec, us_vec = self.convert_z_to_variables(z)
        xs_mat = self.convert_xs_vec_to_xs_mat(xs_vec)
        us_mat = self.convert_us_vec_to_us_mat(us_vec)
        return xs_mat, us_mat
    
    #@partial(jit, static_argnums=(0,))
    def convert_xs_mat_to_xs_vec(self, xs_mat):
        xs_vec = np.reshape(xs_mat.T, n_x*(S+1), 'F')
        return xs_vec
    
    #@partial(jit, static_argnums=(0,))
    def convert_us_mat_to_us_vec(self, us_mat):
        us_vec = np.reshape(us_mat.T, n_u*S, 'F')
        return us_vec
    
    #@partial(jit, static_argnums=(0,))
    def convert_xs_us_mats_to_z(self, xs_mat, us_mat):
        xs_vec = self.convert_xs_mat_to_xs_vec(xs_mat)
        us_vec = self.convert_us_mat_to_us_vec(us_mat)
        z = np.concatenate((np.copy(xs_vec), np.copy(us_vec)), axis=-1)
        return z

    def initial_guess(self, type='keep', x_init=None, x_final=None):
        if type == 'keep':
            z = np.concatenate((
                np.tile(x_init, S+1) + 1e-6, 
                np.zeros(S*n_u) + 1e-6), axis=-1)
        elif type == 'line':
            z = np.concatenate((
                (x_init + (x_final - x_init)/T*np.arange(0,T+dt/10,dt)[:,None]).reshape(-1),
                np.zeros(S*n_u) + 1e-6), axis=-1)
        return z
    
    def warmstart(self, xs, us):
        z = self.convert_xs_us_mats_to_z(xs, us)
        return z

    #@partial(jit, static_argnums=(0,))
    def f(self, x, u):
        v = x[3:6]
        state_dot = np.zeros(n_x)
        state_dot[:3] = v.copy()
        state_dot[3:6] = u / mass - drag_coefficient * np.linalg.norm(v,2) * v / mass
        return state_dot

    def A_d(self,xs):
        v = xs[3:].reshape(3,1)
        Avv_d = np.eye(3)*np.sqrt(v.T @ v) + (v @ v.T)/(np.sqrt(v.T @ v) + 1e-6)
        Av_d = np.hstack((np.zeros((3,3)), Avv_d))
        A_d = - drag_coefficient/mass * np.vstack((np.zeros((3,6)), Av_d))
        return A_d

    def b_d(self,xs):
        v = xs[3:]
        bv_d = - (v * np.sqrt(v.dot(v)))
        b_d = - drag_coefficient/mass * np.hstack((np.zeros(3,), bv_d))
        return b_d

    def define_problem_and_solve(self, xs, us, x_init, x_final, trust_region, scp_iter=0, obs_av=True):
        # Setup SQP problem
        xs, us = xs.T, us.T
        n_time = xs.shape[1]-1
        s = cp.Variable((6, n_time))
        a = cp.Variable((3, n_time))
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
        B = 1/mass*np.vstack((np.zeros((3,3)),np.eye(3)))
        constraints += [s[:,i+1] == (s[:,i] + ((A + self.A_d(xs[:,i])) @ s[:,i] + self.b_d(xs[:,i]) + B @ a[:,i])*dt) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + ((A + self.A_d(xs[:,-1])) @ s[:,-1] + self.b_d(xs[:,-1]) + B @ a[:,-1])*dt == x_final]
        # Control constraints
        constraints += [a >= -u_max]
        constraints += [a <= u_max]
        # Trust region
        for j in range(n_time):
            b_soc_j = -xs[:,j]
            constraints += [cp.SOC(trust_region, s[:,j] + b_soc_j)]
        # Keep-out-zone
        if obs_av:
            for n_obs in range(len(obs_radii)):
                for k in range(n_time):
                    c_koz_k = np.transpose(xs[:3,k] - obs_positions[n_obs,:]).dot(np.eye(3)/(obs_radii[n_obs]**2))
                    b_koz_k = np.sqrt(c_koz_k.dot(xs[:3,k] - obs_positions[n_obs,:]))
                    constraints += [c_koz_k @ (s[:3,k] - obs_positions[n_obs,:]) >= b_koz_k]
        # Compute Cost
        cost = 0.5*cp.sum(cp.norm(a, 2, axis=0)**2)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # SolveOSQP problem
        prob.solve(solver=cp.ECOS, verbose=False)
        if self.verbose:
            print(prob.status)
        if prob.status == 'infeasible':
            print("[solve]: Problem infeasible.")
            s_opt = None
            a_opt = None
            J = None
        else:
            s_opt = s.value.T#np.hstack((s.value, x_final.reshape(-1,1))).T
            a_opt = a.value.T
            s_opt = np.vstack((s_opt, s_opt[-1,:] + ((A + self.A_d(s_opt[-1,:])) @ s_opt[-1,:] + self.b_d(s_opt[-1,:]) +  B @ a_opt[-1,:])*dt))
            J = prob.value

        return s_opt, a_opt, J, prob.status


def L2_error_us(us_mat, us_mat_prev):
    # us_mat - (S, n_u)
    # us_mat_prev - (S, n_u)
    error = np.mean(np.linalg.norm(us_mat-us_mat_prev, axis=-1))
    error = error / np.mean(np.linalg.norm(us_mat, axis=-1))
    return error

def trust_region_diff(us_mat, us_mat_prev):
    error = np.max(np.linalg.norm(us_mat-us_mat_prev, axis=-1))
    return error

def sample_init_target():
    if 'minimum' in dataset_scenario:
        x_init = np.random.uniform(low=[-2, -1, 0, 0, 0, 0], high=[-1, 1, 2, 0, 0, 0])
        if 'random_target' in dataset_scenario:
            x_target = np.random.uniform(low=[1, -1, 0, 0, 0, 0], high=[2, 1, 2, 0, 0, 0])
        else:
            x_target = np.array([1.5, 0, 1, 0, 0, 0])

    elif 'forest' in dataset_scenario:
        valid = False
        while not valid:
            x_init = np.random.uniform(low=[-2, -0.5, 0.2, 0, 0, 0], high=[-1.75, 0.5, 0.6, 0, 0, 0])
            valid = ((np.linalg.norm(x_init[:3] - obs_positions,axis=1) - obs_radii) > 0).all()
        if 'random_target' in dataset_scenario:
            valid = False
            while not valid:
                x_target = np.random.uniform(low=[-0.2, -0.5, 0.2, 0, 0, 0], high=[0.5, 0.5, 0.6, 0, 0, 0])
                valid = ((np.linalg.norm(x_target[:3] - obs_positions,axis=1) - obs_radii) > 0).all()
        else:
            x_target = np.array([0., 0.25, 0.7, 0, 0, 0])
    return x_init, x_target

# ----------------------------------

def ocp_no_obstacle_avoidance(model, x_init, x_final, initial_guess='keep'):
    z_prev = model.initial_guess(initial_guess, x_init, x_final)
    xs_prev, us_prev = model.convert_z_to_variables(z_prev)
    xs_prev = xs_prev.reshape(-1,6)
    us_prev = us_prev.reshape(-1,3)
    
    # Initial condition for the scp
    DELTA_J = 10
    trust_region = trust_region0
    beta_SCP = (trust_regionf/trust_region0)**(1/iter_max_SCP)

    for scp_iter in range(iter_max_SCP):
        '''print("scp_iter =", scp_iter)'''
        # define and solve
        xs, us, J, status = model.define_problem_and_solve(
            xs_prev, us_prev, x_init, x_final, trust_region, scp_iter, obs_av=False)
        if status == 'infeasible':
            break

        # compute error
        L2_error = trust_region_diff(xs, xs_prev) #L2_error_us(us, us_prev)
        if scp_iter > 0:
            DELTA_J = J_prev - J
        
        # Update iterations
        us_prev = us
        xs_prev = xs
        J_prev = J
        trust_region = beta_SCP*trust_region
        if scp_iter >= 1 and (L2_error <= trust_regionf and abs(DELTA_J) < J_tol):
            break
    
    if status == 'infeasible':
        s_opt = None
        a_opt = None
        J = None
    else:
        s_opt = xs
        a_opt = us

    return s_opt, a_opt, J, status

def ocp_obstacle_avoidance_line_ws(model, x_init, x_final, initial_guess='keep'):
    z_prev = model.initial_guess(initial_guess, x_init, x_final)
    xs_prev, us_prev = model.convert_z_to_variables(z_prev)
    xs_prev = xs_prev.reshape(-1,6)
    us_prev = us_prev.reshape(-1,3)
    J_vect = np.ones(shape=(iter_max_SCP,), dtype=float)*1e12

    # Initial condition for the scp
    DELTA_J = 10
    trust_region = trust_region0
    beta_SCP = (trust_regionf/trust_region0)**(1/iter_max_SCP)

    for scp_iter in range(iter_max_SCP):
        '''print("scp_iter =", scp_iter)'''
        # define and solve
        xs, us, J, status = model.define_problem_and_solve(
            xs_prev, us_prev, x_init, x_final, trust_region, scp_iter, obs_av=True)
        if status == 'infeasible':
            break
        J_vect[scp_iter] = J

        # compute error
        L2_error = trust_region_diff(xs, xs_prev) #L2_error_us(us, us_prev)
        if scp_iter > 0:
            DELTA_J = J_prev - J

        # Update iterations
        us_prev = us
        xs_prev = xs
        J_prev = J
        trust_region = beta_SCP*trust_region
        if scp_iter >= 1 and (L2_error <= trust_regionf and abs(DELTA_J) < J_tol):
            break
    
    if status == 'infeasible':
        s_opt = None
        a_opt = None
        J = None
    else:
        s_opt = xs
        a_opt = us

    return s_opt, a_opt, J_vect, status, scp_iter

def ocp_obstacle_avoidance(model, xs_prev, us_prev, x_init, x_final):
    #z_prev = model.warmstart(xs_prev, us_prev)
    xs_prev = xs_prev.reshape(-1,6)
    us_prev = us_prev.reshape(-1,3)
    J_vect = np.ones(shape=(iter_max_SCP,), dtype=float)*1e12
    
    # Initial condition for the scp
    DELTA_J = 10
    trust_region = trust_region0
    beta_SCP = (trust_regionf/trust_region0)**(1/iter_max_SCP)

    for scp_iter in range(iter_max_SCP):
        '''print("scp_iter =", scp_iter)'''
        # define and solve
        xs, us, J, status = model.define_problem_and_solve(
            xs_prev, us_prev, x_init, x_final, trust_region, scp_iter, obs_av=True)
        if status == 'infeasible':
            break
        J_vect[scp_iter] = J

        # compute error
        L2_error = trust_region_diff(xs, xs_prev) #L2_error_us(us, us_prev)
        if scp_iter > 0:
            DELTA_J = J_prev - J
            
        # Update iterations
        us_prev = us
        xs_prev = xs
        J_prev = J
        trust_region = beta_SCP*trust_region
        if scp_iter >= 1 and (L2_error <= trust_regionf and abs(DELTA_J) < J_tol):
            break
    
    if status == 'infeasible':
        s_opt = None
        a_opt = None
        J = None
    else:
        s_opt = xs
        a_opt = us

    return s_opt, a_opt, J_vect, status, scp_iter

# Reward to go and constraints to go
def compute_reward_to_go(actions):
    if len(actions.shape) == 2:
        actions = actions[None,:,:]
    n_data, n_time = actions.shape[0], actions.shape[1]
    rewards_to_go = np.empty(shape=(n_data, n_time), dtype=float)
    for n in range(n_data):
        for t in range(n_time):
            rewards_to_go[n, t] = - np.sum((np.linalg.norm(actions[n, t:, :], axis=1))**2)/2
        
    return rewards_to_go

def compute_constraint_to_go(states, obs_positions, obs_radii):
    if len(states.shape) == 2:
        states = states[None,:,:]
    n_data, n_time = states.shape[0], states.shape[1]
    constraint_to_go = np.empty(shape=(n_data, n_time), dtype=float)
    for n in range(n_data):
        constr_koz_n, constr_koz_violation_n = check_koz_constraint(states[n, :, :], obs_positions, obs_radii)
        constraint_to_go[n,:] = np.array([np.sum(constr_koz_violation_n[:,t:]) for t in range(n_time)])

    return constraint_to_go

def check_koz_constraint(states, obs_positions, obs_radii):

    constr_koz = np.linalg.norm(states[None,:,:3] - obs_positions[:,None,:], axis=2) - obs_radii[:,None]
    constr_koz_violation = 1*(constr_koz <= 0)

    return constr_koz, constr_koz_violation

    