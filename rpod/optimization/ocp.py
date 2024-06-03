import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import cvxpy as cp
import mosek as mk
import time

'''
    TODO:
        ocp_cvx_complex():
          - implement the constraints extraction also in the normalized optimization part
        
        ocp_scp_complex():
          - add plume impingement constraint to the normalized version of scp --> How to normalize u_ref?
          - add RFbody bounding box constraint to the normalized version of scp --> How to normalize u_ref?
          - implement the constraints extraction also in the normalized optimization part
        
        solve_scp_complex():
          - should the trust region be checked only on states<=>states_ref or should we include actions<=>actions_ref too?
            Probably the case should be the latter, even though they are on a different scale, so how to compare them to states? Actually similar problem for the velcities
        
        check_body_bb_constraint():
          - Vectorize the computation of DCM_bi --> it may be done also in the ocp_scp_complex function?
        
        compute_constraints_to_go():
          - add other NL constraints in the computation?
'''

from optimization.rpod_scenario import n_ref, a_ref, D_pos, DEED_koz, EE_koz, dock_wyp_sample, trust_region0, trust_regionf, iter_max_SCP, J_tol

'''def ocp_cvx_tpbvp(stm, cim, psi, s_0, n_time):

    s = cp.Variable((6, n_time))
    a = cp.Variable((3, n_time))

    solve_normalized = False

    if not solve_normalized:

        # Compute parameters
        s_f = state_roe_target

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim[:,:,-1] @ a[:,-1] == s_f]

        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value
        
    else:

        # Compute normalized parameters
        cim_n = cim*n_ref
        s_0_n = s_0/a_ref
        s_f_n = state_roe_target/a_ref

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0_n]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim_n[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim_n[:,:,-1] @ a[:,-1] == s_f_n]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.MOSEK, verbose=False)

        s_opt = s.value*a_ref
        a_opt = a.value*a_ref*n_ref

    return s_opt, a_opt, prob.status'''

def ocp_cvx(stm, cim, psi, s_0, dock_param, n_time):

    s = cp.Variable((6, n_time))
    a = cp.Variable((3, n_time))

    solve_normalized = False

    # Docking port parameter extraction
    state_roe_target, dock_axis, dock_port, dock_cone_angle, dock_wyp = dock_param['state_roe_target'], dock_param['dock_axis'], dock_param['dock_port'], dock_param['dock_cone_angle'], dock_param['dock_wyp']

    if not solve_normalized:

        # Compute parameters
        s_f = state_roe_target
        d_soc = -np.transpose(dock_axis).dot(dock_port)/np.cos(dock_cone_angle)

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim[:,:,-1] @ a[:,-1] == s_f]
        # Docking waypoint
        constraints += [psi[:,:,dock_wyp_sample] @ s[:,dock_wyp_sample] == dock_wyp]
        # Approach cone
        for j in range(dock_wyp_sample, n_time):
            c_soc_j = np.transpose(dock_axis).dot(np.matmul(D_pos, psi[:,:,j]))/np.cos(dock_cone_angle)
            A_soc_j = np.matmul(D_pos, psi[:,:,j])
            b_soc_j = -dock_port
            constraints += [cp.SOC(c_soc_j @ s[:,j] + d_soc, A_soc_j @ s[:,j] + b_soc_j)]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value
        
    else:

        # Compute normalized parameters
        cim_n = cim*n_ref
        psi_norm_vect = np.array([1, 1, 1, 1/n_ref, 1/n_ref, 1/n_ref]).reshape(6,)
        s_0_n = s_0/a_ref
        s_f_n = state_roe_target/a_ref
        dock_wyp_n = np.multiply(dock_wyp, np.array([1/a_ref, 1/a_ref, 1/a_ref, 1/(a_ref*n_ref), 1/(a_ref*n_ref), 1/(a_ref*n_ref)]).reshape(6,))
        dock_port_n = dock_port/a_ref
        d_soc = -np.transpose(dock_axis).dot(dock_port_n)/np.cos(dock_cone_angle)

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0_n]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim_n[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim_n[:,:,-1] @ a[:,-1] == s_f_n]
        # Docking waypoint
        psi_wyp_n = psi[:,:,dock_wyp_sample]*psi_norm_vect[:, np.newaxis]
        constraints += [psi_wyp_n @ s[:,dock_wyp_sample] == dock_wyp_n]
        # Approach cone
        for j in range(dock_wyp_sample, n_time):
            psi_j_n = psi[:,:,j]*psi_norm_vect[:, np.newaxis]
            c_soc_j = np.transpose(dock_axis).dot(np.matmul(D_pos, psi_j_n))/np.cos(dock_cone_angle)
            A_soc_j = np.matmul(D_pos, psi_j_n)
            b_soc_j = -dock_port_n
            constraints += [cp.SOC(c_soc_j @ s[:,j] + d_soc, A_soc_j @ s[:,j] + b_soc_j)]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.MOSEK, verbose=False)

        s_opt = s.value*a_ref
        a_opt = a.value*a_ref*n_ref

    return s_opt, a_opt, prob.status

def ocp_scp(stm, cim, psi, s_0, dock_param, s_ref, trust_region, n_time):

    s = cp.Variable((6, n_time))
    a = cp.Variable((3, n_time))

    solve_normalized = False

    # Docking port parameter extraction
    state_roe_target, dock_axis, dock_port, dock_cone_angle, dock_wyp = dock_param['state_roe_target'], dock_param['dock_axis'], dock_param['dock_port'], dock_param['dock_cone_angle'], dock_param['dock_wyp']

    if not solve_normalized:

        # Compute parameters
        s_f = state_roe_target
        d_soc = -np.transpose(dock_axis).dot(dock_port)/np.cos(dock_cone_angle)

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim[:,:,-1] @ a[:,-1] == s_f]
        # Docking waypoint
        constraints += [psi[:,:,dock_wyp_sample] @ s[:,dock_wyp_sample] == dock_wyp]
        # Approach cone
        for j in range(dock_wyp_sample, n_time):
            c_soc_j = np.transpose(dock_axis).dot(np.matmul(D_pos, psi[:,:,j]))/np.cos(dock_cone_angle)
            A_soc_j = np.matmul(D_pos, psi[:,:,j])
            b_soc_j = -dock_port
            constraints += [cp.SOC(c_soc_j @ s[:,j] + d_soc, A_soc_j @ s[:,j] + b_soc_j)]
        # Keep-out-zone plus trust region
        for k in range(dock_wyp_sample):
            c_koz_k = np.transpose(s_ref[:,k]).dot(np.matmul(np.transpose(psi[:,:,k]), np.matmul(DEED_koz, psi[:,:,k])))
            b_koz_k = np.sqrt(c_koz_k.dot(s_ref[:,k]))
            constraints += [c_koz_k @ s[:,k] >= b_koz_k]
            b_soc_k = -s_ref[:,k]
            constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value

    else:

        # Compute normalized parameters
        s_ref_n = s_ref/a_ref
        cim_n = cim*n_ref
        psi_norm_vect = np.array([1, 1, 1, 1/n_ref, 1/n_ref, 1/n_ref]).reshape(6,)
        s_0_n = s_0/a_ref
        s_f_n = state_roe_target/a_ref
        dock_wyp_n = np.multiply(dock_wyp, np.array([1/a_ref, 1/a_ref, 1/a_ref, 1/(a_ref*n_ref), 1/(a_ref*n_ref), 1/(a_ref*n_ref)]).reshape(6,))
        dock_port_n = dock_port/a_ref
        trust_region_n = trust_region/a_ref
        d_soc = -np.transpose(dock_axis).dot(dock_port_n)/np.cos(dock_cone_angle)
        DEED_koz_n = DEED_koz*a_ref**2

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0_n]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim_n[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim_n[:,:,-1] @ a[:,-1] == s_f_n]
        # Docking waypoint
        psi_wyp_n = psi[:,:,dock_wyp_sample]*psi_norm_vect[:, np.newaxis]
        constraints += [psi_wyp_n @ s[:,dock_wyp_sample] == dock_wyp_n]
        # Approach cone
        for j in range(dock_wyp_sample, n_time):
            psi_j_n = psi[:,:,j]*psi_norm_vect[:, np.newaxis]
            c_soc_j = np.transpose(dock_axis).dot(np.matmul(D_pos, psi_j_n))/np.cos(dock_cone_angle)
            A_soc_j = np.matmul(D_pos, psi_j_n)
            b_soc_j = -dock_port_n
            constraints += [cp.SOC(c_soc_j @ s[:,j] + d_soc, A_soc_j @ s[:,j] + b_soc_j)]
        # Keep-out-zone plus trust region
        for k in range(dock_wyp_sample):
            psi_k_n = psi[:,:,k]*psi_norm_vect[:, np.newaxis]
            c_koz_k = np.transpose(s_ref_n[:,k]).dot(np.matmul(np.transpose(psi_k_n), np.matmul(DEED_koz_n, psi_k_n)))
            b_koz_k = np.sqrt(c_koz_k.dot(s_ref_n[:,k]))
            constraints += [c_koz_k @ s[:,k] >= b_koz_k]
            b_soc_k = -s_ref_n[:,k]
            constraints += [cp.SOC(trust_region_n, s[:,k] + b_soc_k)]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.MOSEK, verbose=False)

        s_opt = s.value*a_ref
        a_opt = a.value*a_ref*n_ref

    return s_opt, a_opt, prob.status, prob.value

def solve_scp(stm, cim, psi, state_roe_0, dock_param, states_roe_ref, n_time):

    beta_SCP = (trust_regionf/trust_region0)**(1/iter_max_SCP)

    iter_SCP = 0
    DELTA_J = 10
    J_vect = np.ones(shape=(iter_max_SCP,), dtype=float)*1e12

    diff = trust_region0
    trust_region = trust_region0
    
    runtime0_scp = time.time()
    while (iter_SCP < iter_max_SCP) and ((diff > trust_regionf) or (DELTA_J > J_tol)):
        
        # Solve OCP (safe)
        try:
            [states_roe, actions, feas, cost] = ocp_scp(stm, cim, psi, state_roe_0, dock_param, states_roe_ref, trust_region, n_time)
        except:
            states_roe = None
            actions = None
            feas = 'infeasible'

        if not np.char.equal(feas,'infeasible'):#np.char.equal(feas,'optimal'):
            if iter_SCP == 0:
                states_roe_vect = states_roe[None,:,:].copy()
                actions_vect = actions[None,:,:].copy()
            else:
                states_roe_vect = np.vstack((states_roe_vect, states_roe[None,:,:]))
                actions_vect = np.vstack((actions_vect, actions[None,:,:]))

            # Compute performances
            diff = np.max(la.norm(states_roe - states_roe_ref, axis=0))
            # print('scp gap:', diff)
            J = sum(la.norm(actions,axis=0));#2,1
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

            #  Update trust region
            trust_region = beta_SCP * trust_region
            
            #  Update reference
            states_roe_ref = states_roe
        else:
            print('unfeasible scp')
            break;
    
    runtime1_scp = time.time()
    runtime_scp = runtime1_scp - runtime0_scp
    
    ind_J_min = iter_SCP-1#np.argmin(J_vect)
    if not np.char.equal(feas,'infeasible'):#np.char.equal(feas,'optimal'):
        states_roe = states_roe_vect[ind_J_min,:,:]
        actions = actions_vect[ind_J_min,:,:]
    else:
        states_roe = None
        actions = None

    return states_roe, actions, feas, iter_SCP, J_vect, runtime_scp

def solve_scp_line_ws(stm, cim, psi, state_roe_0, dock_param, n_time):

    beta_SCP = (trust_regionf/trust_region0)**(1/iter_max_SCP)

    iter_SCP = 0
    DELTA_J = 10
    J_vect = np.ones(shape=(iter_max_SCP,), dtype=float)*1e12

    diff = trust_region0
    trust_region = trust_region0
    
    state_roe_init = state_roe_0
    state_roe_final = dock_param['state_roe_target']
    states_roe_ref = (state_roe_init + ((state_roe_final - state_roe_init)/n_time)*np.arange(0,n_time,1)[:,None]).T

    '''state_rtn_init = psi[:,:,0] @ state_roe_0
    state_rtn_final = dock_param['state_rtn_target']
    states_rtn_ref = (state_rtn_init + ((state_rtn_final - state_rtn_init)/n_time)*np.arange(0,n_time,1)[:,None]).reshape(-1,6,1)
    psi_inv = np.linalg.solve(psi.transpose(2,0,1), np.eye(6)[None,:,:]).transpose(1,2,0)
    states_roe_ref = (psi_inv.transpose(2,0,1) @ states_rtn_ref).reshape(-1,6).T'''
    
    runtime0_scp = time.time()
    while (iter_SCP < iter_max_SCP) and ((diff > trust_regionf) or (DELTA_J > J_tol)):
        
        # Solve OCP (safe)
        try:
            [states_roe, actions, feas, cost] = ocp_scp(stm, cim, psi, state_roe_0, dock_param, states_roe_ref, trust_region, n_time)
        except:
            states_roe = None
            actions = None
            feas = 'infeasible'

        if not np.char.equal(feas,'infeasible'):#np.char.equal(feas,'optimal'):
            if iter_SCP == 0:
                states_roe_vect = states_roe[None,:,:].copy()
                actions_vect = actions[None,:,:].copy()
            else:
                states_roe_vect = np.vstack((states_roe_vect, states_roe[None,:,:]))
                actions_vect = np.vstack((actions_vect, actions[None,:,:]))

            # Compute performances
            diff = np.max(la.norm(states_roe - states_roe_ref, axis=0))
            # print('scp gap:', diff)
            J = sum(la.norm(actions,axis=0));#2,1
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

            #  Update trust region
            trust_region = beta_SCP * trust_region
            
            #  Update reference
            states_roe_ref = states_roe
        else:
            print('unfeasible scp')
            break;
    
    runtime1_scp = time.time()
    runtime_scp = runtime1_scp - runtime0_scp
    
    ind_J_min = iter_SCP-1#np.argmin(J_vect)
    if not np.char.equal(feas,'infeasible'):#np.char.equal(feas,'optimal'):
        states_roe = states_roe_vect[ind_J_min,:,:]
        actions = actions_vect[ind_J_min,:,:]
    else:
        states_roe = None
        actions = None

    return states_roe, actions, feas, iter_SCP, J_vect, runtime_scp

'''
def ocp_cvx_complex(stm, cim, psi, s_0, n_time, constr_range):

    s = cp.Variable((6, n_time))
    a = cp.Variable((3, n_time))

    solve_normalized = False

    if not solve_normalized:

        # Compute parameters
        s_f = state_roe_target
        d_soc = -np.transpose(dock_axis).dot(dock_port)/np.cos(dock_cone_angle)

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
            constraints += [psi[:,:,i] @ s[:,i] == dock_wyp for i in constr_range['docking_waypoint']]
        # Approach cone
        if len(constr_range['approach_cone']) > 0:
            for j in constr_range['approach_cone']:
                c_soc_j = np.transpose(dock_axis).dot(np.matmul(D_pos, psi[:,:,j]))/np.cos(dock_cone_angle)
                A_soc_j = np.matmul(D_pos, psi[:,:,j])
                b_soc_j = -dock_port
                constraints += [cp.SOC(c_soc_j @ s[:,j] + d_soc, A_soc_j @ s[:,j] + b_soc_j)]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value
        
    else:

        # Compute normalized parameters
        cim_n = cim*n_ref
        psi_norm_vect = np.array([1, 1, 1, 1/n_ref, 1/n_ref, 1/n_ref]).reshape(6,)
        s_0_n = s_0/a_ref
        s_f_n = state_roe_target/a_ref
        dock_wyp_n = np.multiply(dock_wyp, np.array([1/a_ref, 1/a_ref, 1/a_ref, 1/(a_ref*n_ref), 1/(a_ref*n_ref), 1/(a_ref*n_ref)]).reshape(6,))
        dock_port_n = dock_port/a_ref
        d_soc = -np.transpose(dock_axis).dot(dock_port_n)/np.cos(dock_cone_angle)

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0_n]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim_n[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim_n[:,:,-1] @ a[:,-1] == s_f_n]
        # Docking waypoint
        psi_wyp_n = psi[:,:,dock_wyp_sample]*psi_norm_vect[:, np.newaxis]
        constraints += [psi_wyp_n @ s[:,dock_wyp_sample] == dock_wyp_n]
        # Approach cone
        for j in range(dock_wyp_sample, n_time):
            psi_j_n = psi[:,:,j]*psi_norm_vect[:, np.newaxis]
            c_soc_j = np.transpose(dock_axis).dot(np.matmul(D_pos, psi_j_n))/np.cos(dock_cone_angle)
            A_soc_j = np.matmul(D_pos, psi_j_n)
            b_soc_j = -dock_port_n
            constraints += [cp.SOC(c_soc_j @ s[:,j] + d_soc, A_soc_j @ s[:,j] + b_soc_j)]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.MOSEK, verbose=False)

        s_opt = s.value*a_ref
        a_opt = a.value*a_ref*n_ref

    return s_opt, a_opt, prob.status

def solve_scp_complex(stm, cim, psi, state_roe_0, states_roe_ref, action_rtn_ref, n_time, constr_range):

    beta_SCP = (trust_regionf/trust_region0)**(1/iter_max_SCP)

    iter_SCP = 0
    DELTA_J = 10
    J_vect = np.ones(shape=(iter_max_SCP,), dtype=float)*1e12

    diff = trust_region0
    trust_region = trust_region0
    
    runtime0_scp = time.time()
    while (iter_SCP < iter_max_SCP) and ((diff > trust_regionf) or (DELTA_J > J_tol)):
        
        # Solve OCP (safe)
        try:
            [states_roe, actions, feas, cost] = ocp_scp_complex(stm, cim, psi, state_roe_0, states_roe_ref, action_rtn_ref, trust_region, n_time, constr_range)
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
            J = sum(la.norm(actions,axis=0));#2,1
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
            print(feas)

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
        actions = actions_vect[ind_J_min,:,:]
    else:
        states_roe = None
        actions = None

    return states_roe, actions, feas, iter_SCP, J_vect, runtime_scp

def ocp_scp_complex(stm, cim, psi, s_0, s_ref, u_ref, trust_region, n_time, constr_range):

    s = cp.Variable((6, n_time))
    a = cp.Variable((3, n_time))

    solve_normalized = False

    if not solve_normalized:

        # Compute parameters
        s_f = state_roe_target
        d_soc = -np.transpose(dock_axis).dot(dock_port)/np.cos(dock_cone_angle)

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim[:,:,-1] @ a[:,-1] == s_f]
        # Trust region
        for j in range(dock_wyp_sample):
            b_soc_j = -s_ref[:,j]
            constraints += [cp.SOC(trust_region, s[:,j] + b_soc_j)]

        # Additional constraints based on constr_range dictionary
        # Docking waypoint
        if len(constr_range['docking_waypoint']) > 0:
            constraints += [psi[:,:,k] @ s[:,k] == dock_wyp for k in constr_range['docking_waypoint']]
        # Approach cone
        if len(constr_range['approach_cone']) > 0:
            for l in constr_range['approach_cone']:
                c_soc_l = np.transpose(dock_axis).dot(np.matmul(D_pos, psi[:,:,l]))/np.cos(dock_cone_angle)
                A_soc_l = np.matmul(D_pos, psi[:,:,l])
                b_soc_l = -dock_port
                constraints += [cp.SOC(c_soc_l @ s[:,l] + d_soc, A_soc_l @ s[:,l] + b_soc_l)]
        # Plume impingement
        if len(constr_range['plume_impingement']) > 0:
            for m in constr_range['plume_impingement']:
                A_plum_m = (dock_axis.T - (np.cos(plume_imp_angle)*u_ref[:,m].T)/np.sqrt(u_ref[:,m].T @ u_ref[:,m]))
                constraints += [A_plum_m @ a[:,m] <= 0]
        # Keep-out-zone plus trust region
        if len(constr_range['keep_out_zone']) > 0:
            for n in constr_range['keep_out_zone']:
                c_koz_n = np.transpose(s_ref[:,n]).dot(np.matmul(np.transpose(psi[:,:,n]), np.matmul(DEED_koz, psi[:,:,n])))
                b_koz_n = np.sqrt(c_koz_n.dot(s_ref[:,n]))
                constraints += [c_koz_n @ s[:,n] >= b_koz_n]
        # Thrusters bounding box
        if len(constr_range['actions_bounding_box']) > 0:
            # If target poiting is active, then compute boudning box constraint in body RF
            if len(constr_range['target_pointing']) > 0:
                b_bb_o = np.array([dv_max, dv_max, dv_max])**2
                for o in constr_range['target_pointing']:
                    As_bb_o, Au_bb_o, c_bb_o = bounding_box_body_linearization(psi[:,:,o] @ s_ref[:,o], u_ref[:,o])
                    constraints += [As_bb_o @ (psi[:,:,o] @ s[:,o]) + Au_bb_o @ a[:,o] + c_bb_o <= b_bb_o]
            # Otherwise assume body RF == rtn RF
            else:
                upper_bb_o = np.array([dv_max, dv_max, dv_max])
                lower_bb_o = -np.array([dv_max, dv_max, dv_max])
                for o in constr_range['actions_bounding_box']:
                    constraints += [a[:,o] <= upper_bb_o]
                    constraints += [a[:,o] >= lower_bb_o]

        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value

    else:

        # Compute normalized parameters
        s_ref_n = s_ref/a_ref
        cim_n = cim*n_ref
        psi_norm_vect = np.array([1, 1, 1, 1/n_ref, 1/n_ref, 1/n_ref]).reshape(6,)
        s_0_n = s_0/a_ref
        s_f_n = state_roe_target/a_ref
        dock_wyp_n = np.multiply(dock_wyp, np.array([1/a_ref, 1/a_ref, 1/a_ref, 1/(a_ref*n_ref), 1/(a_ref*n_ref), 1/(a_ref*n_ref)]).reshape(6,))
        dock_port_n = dock_port/a_ref
        trust_region_n = trust_region/a_ref
        d_soc = -np.transpose(dock_axis).dot(dock_port_n)/np.cos(dock_cone_angle)
        DEED_koz_n = DEED_koz*a_ref**2

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0_n]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim_n[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim_n[:,:,-1] @ a[:,-1] == s_f_n]
        # Docking waypoint
        psi_wyp_n = psi[:,:,dock_wyp_sample]*psi_norm_vect[:, np.newaxis]
        constraints += [psi_wyp_n @ s[:,dock_wyp_sample] == dock_wyp_n]
        # Approach cone
        for j in range(dock_wyp_sample, n_time):
            psi_j_n = psi[:,:,j]*psi_norm_vect[:, np.newaxis]
            c_soc_j = np.transpose(dock_axis).dot(np.matmul(D_pos, psi_j_n))/np.cos(dock_cone_angle)
            A_soc_j = np.matmul(D_pos, psi_j_n)
            b_soc_j = -dock_port_n
            constraints += [cp.SOC(c_soc_j @ s[:,j] + d_soc, A_soc_j @ s[:,j] + b_soc_j)]
        # Keep-out-zone plus trust region
        for k in range(dock_wyp_sample):
            psi_k_n = psi[:,:,k]*psi_norm_vect[:, np.newaxis]
            c_koz_k = np.transpose(s_ref_n[:,k]).dot(np.matmul(np.transpose(psi_k_n), np.matmul(DEED_koz_n, psi_k_n)))
            b_koz_k = np.sqrt(c_koz_k.dot(s_ref_n[:,k]))
            constraints += [c_koz_k @ s[:,k] >= b_koz_k]
            b_soc_k = -s_ref_n[:,k]
            constraints += [cp.SOC(trust_region_n, s[:,k] + b_soc_k)]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.MOSEK, verbose=False)

        s_opt = s.value*a_ref
        a_opt = a.value*a_ref*n_ref

    return s_opt, a_opt, prob.status, prob.value
'''
def check_koz_constraint(states_rtn, n_time):

    constr_koz = np.empty(shape=(n_time,), dtype=float)
    constr_koz_violation = np.zeros(shape=(n_time,), dtype=float)

    for i in range(n_time):
        constr_koz[i] = np.transpose(states_rtn[:3, i]).dot(EE_koz.dot(states_rtn[:3, i]))
        if (constr_koz[i] < 1) and (i < dock_wyp_sample):
            constr_koz_violation[i] = 1

    return constr_koz, constr_koz_violation
'''
def check_plume_constraint(states_rtn, actions_rtn, n_time):

    constr_plume = np.empty(shape=(n_time,), dtype=float)
    constr_plume_violation = np.zeros(shape=(n_time,), dtype=float)

    for i in range(n_time):
        if np.linalg.norm(actions_rtn[:,i]) > 1e-9:
            constr_plume[i] = (dock_axis.T @ actions_rtn[:,i])/np.linalg.norm(actions_rtn[:,i])#np.transpose(states_rtn[:3, i]).dot(EE_plume.dot(states_rtn[:3, i]))
            if (constr_plume[i] > np.cos(plume_imp_angle)) and (i >= dock_wyp_sample):
                constr_plume_violation[i] = 1
        else:
            constr_plume[i] = np.nan

    return constr_plume, constr_plume_violation

def check_body_bb_constraint(states_rtn, actions_rtn, n_time):

    dv_bb = np.empty(shape=(3,n_time), dtype=float)
    constr_bb_violation = np.zeros(shape=(n_time,), dtype=float)

    # Body axes for the entire trajectory
    #dr = dock_port.reshape((1,3,1)) - states_rtn[None,:3,:].transpose(2,1,0) # dim 100x3x1

    for i in range(n_time):
        # Body axes and matrix rotation
        dr = dock_port - states_rtn[:3,i]
        xB = dr/np.linalg.norm(dr)
        yB = np.array([-xB[1], xB[0], 0])/np.sqrt(xB[0]**2 + xB[1]**2)
        zB = np.cross(xB, yB)
        DCM_bi = np.vstack((xB, yB, zB))

        # DeltaV in body RF
        dv_bb[:,i] = DCM_bi @ actions_rtn[:,i]
        
        if (dv_bb[:,i]**2 > dv_max**2).any():
            constr_bb_violation[i] = np.sum(dv_bb[:,i]**2 > dv_max**2)

    return dv_bb, constr_bb_violation

def bounding_box_body_linearization(s_rtn_ref, u_ref):
    #
    #Linearization of the bounding box constraint around a reference state and action trajectory.
    #ATTENTION: Both states and actions are expected to be expressed in rtn RF.
    #
    # Relative position wrt the dock port
    dr = dock_port - s_rtn_ref[:3]

    # Current body axes given target poiting capabilities
    xB = dr/np.linalg.norm(dr)
    yB = np.array([-xB[1], xB[0], 0])/np.sqrt(xB[0]**2 + xB[1]**2)
    zB = np.cross(xB,yB)
    DCM_bi = np.vstack((xB, yB, zB))

    # Linearization of h1 = (xB'*u)^2 <= dv_max^2, h2 = (yB'*u)^2 <= dv_max^2, h3 = ||dv||^2 - f1 - f2 <= dv_max^2
    # h computed in the reference
    h_ref = (DCM_bi @ u_ref)**2

    # Dependency from the states
    # dh1/dr --> As00, As01, As02, dh2/dr --> As10, As11, As12, dh2/dr --> As20, As21, As22    
    dh1_dr = 2*(xB.T @ u_ref)*(u_ref.T @ (-np.eye(3)*(dr.T @ dr) + np.outer(dr,dr)))/((dr.T @ dr)**(3/2))
    dh2_dr = 2*(yB.T @ u_ref)*np.array([-u_ref[0]*dr[0]*dr[1] - u_ref[1]*dr[1]**2,
                                        u_ref[0]*dr[0]**2 + u_ref[1]*dr[0]*dr[1],
                                        0.]).T/((dr[0]**2 + dr[1]**2)**(3/2))
    dh3_dr = -dh1_dr - dh2_dr
    As_bb_i = np.hstack( (np.vstack( (dh1_dr, dh2_dr, dh3_dr) ), np.zeros((3,3))) )

    # Dependency from the action
    # dh1/du --> Au00, Au01, Au02, dh2/du --> Au10, Au11, Au12, dh2/du --> Au20, Au21, Au22
    dh1_du = 2*(xB.T @ u_ref)*xB.T
    dh2_du = 2*(yB.T @ u_ref)*yB.T
    dh3_du = 2*u_ref.T - dh1_du - dh2_du
    Au_bb_i = np.vstack( (dh1_du, dh2_du, dh3_du) )

    # Constant term c = h_ref - As @ s_ref - Au @ u_ref
    c_bb_i = h_ref - As_bb_i @ s_rtn_ref - Au_bb_i @ u_ref

    return As_bb_i, Au_bb_i, c_bb_i
'''
def compute_constraint_to_go(states_rtn, n_data, n_time):

    constraint_to_go = np.empty(shape=(n_data, n_time), dtype=float)
    for n in range(n_data):
        constr_koz_n, constr_koz_violation_n = check_koz_constraint(np.transpose(np.squeeze(states_rtn[n, :, :])), n_time)
        for t in range(n_time):
            constraint_to_go[n, t] = np.sum(constr_koz_violation_n[t:])

    return constraint_to_go

def compute_reward_to_go(actions, n_data, n_time):

    rewards_to_go = np.empty(shape=(n_data, n_time), dtype=float)
    for n in range(n_data):
        for t in range(n_time):
            rewards_to_go[n, t] = - np.sum(la.norm(actions[n, t:, :], axis=1))
        
    return rewards_to_go