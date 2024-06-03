# Orbit dynamics
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import copy

# Constants
J2 = 0.001082635819197
R_E = 6.3781363e+06
mu_E = 3.986004415e+14

# Functions
def map_mtx_roe_to_rtn(oe):

    a = oe.item(0)
    u = oe.item(4) + oe.item(5)
    n = np.sqrt(mu_E/a**3)
    
    map_1 = np.array([1, 0, -np.cos(u), -np.sin(u), 0, 0]).reshape((1,6))
    map_2 = np.array([0, 1, 2*np.sin(u), -2*np.cos(u), 0, 0]).reshape((1,6))
    map_3 = np.array([0, 0, 0, 0, np.sin(u), -np.cos(u)]).reshape((1,6))
    map_4 = np.array([0, 0, np.sin(u)*n, -np.cos(u)*n, 0, 0]).reshape((1,6))
    map_5 = np.array([-(3/2)*n, 0, 2*np.cos(u)*n, 2*np.sin(u)*n, 0, 0]).reshape((1,6))
    map_6 = np.array([0, 0, 0, 0, np.cos(u)*n, np.sin(u)*n]).reshape((1,6))

    map = np.concatenate((map_1, map_2, map_3, map_4, map_5, map_6), axis=0)

    return map

def map_roe_to_rtn(roe, oe):

    map = map_mtx_roe_to_rtn(oe)
    
    return map.dot(roe)

def map_rtn_to_roe(rtn, oe):

    map = map_mtx_roe_to_rtn(oe)
        
    return la.solve(map, rtn)

def map_body_to_rtn(rtn, target):
    '''
    Function to compute the rotation matrix from body RF to rtn RF assuming perfect target pointing capabilities.
    '''

    # Axis xB aligned with the LOS, axis yB in O_{rt}, axis zB placed accordingly
    dr = target[:3] - rtn[:3]
    xB = dr/np.linalg.norm(dr)
    yB = np.array([-xB[1], xB[0], 0])/np.sqrt(xB[0]**2 + xB[1]**2)
    zB = np.cross(xB, yB)
    DCM_ib = np.vstack((xB, yB, zB)).T

    return DCM_ib

def state_transition(oe, t):

    # From : Koenig A.W., Guffanti T., D'Amico S.; 
    # New State Transition Matrices for Spacecraft Relative Motion in Perturbed Orbits; 
    # Journal of Guidance, Control, and Dynamics, Vol. 40, No. 7, pp. 1749-1768 (September 2017).

    a = oe.item(0)
    e = oe.item(1)
    i = oe.item(2)
    w = oe.item(4)

    n = np.sqrt(mu_E/a**3)
    eta=np.sqrt(1-e**2)
    k=3/4*J2*R_E**2*np.sqrt(mu_E)/(a**(7/2)*eta**4)
    E=1+eta
    F=4+3*eta
    G=1/eta**2
    P=3*np.cos(i)**2-1
    Q=5*np.cos(i)**2-1
    S=np.sin(2*i)
    T=np.sin(i)**2

    w_dot=k*Q
    w_f=w+w_dot*t
    e_xi=e*np.cos(w)
    e_yi=e*np.sin(w)
    e_xf=e*np.cos(w_f)
    e_yf=e*np.sin(w_f)

    Phi_11=1
    Phi_12=0
    Phi_13=0
    Phi_14=0 
    Phi_15=0 
    Phi_16=0
    Phi_1 = np.array([Phi_11,Phi_12,Phi_13,Phi_14,Phi_15,Phi_16]).reshape((1,6))
    Phi_21= -(7/2*k*E*P+3/2*n)*t
    Phi_22=1
    Phi_23=k*e_xi*F*G*P*t
    Phi_24=k*e_yi*F*G*P*t
    Phi_25=-k*F*S*t
    Phi_26=0
    Phi_2 = np.array([Phi_21,Phi_22,Phi_23,Phi_24,Phi_25,Phi_26]).reshape((1,6))
    Phi_31=7/2*k*e_yf*Q*t
    Phi_32=0
    Phi_33=np.cos(w_dot*t)-4*k*e_xi*e_yf*G*Q*t
    Phi_34=-np.sin(w_dot*t)-4*k*e_yi*e_yf*G*Q*t
    Phi_35=5*k*e_yf*S*t
    Phi_36=0
    Phi_3 = np.array([Phi_31,Phi_32,Phi_33,Phi_34,Phi_35,Phi_36]).reshape((1,6))
    Phi_41=-7/2*k*e_xf*Q*t
    Phi_42=0
    Phi_43=np.sin(w_dot*t)+4*k*e_xi*e_xf*G*Q*t
    Phi_44=np.cos(w_dot*t)+4*k*e_yi*e_xf*G*Q*t
    Phi_45=-5*k*e_xf*S*t
    Phi_46=0
    Phi_4 = np.array([Phi_41,Phi_42,Phi_43,Phi_44,Phi_45,Phi_46]).reshape((1,6))
    Phi_51=0
    Phi_52=0
    Phi_53=0
    Phi_54=0
    Phi_55=1
    Phi_56=0
    Phi_5 = np.array([Phi_51,Phi_52,Phi_53,Phi_54,Phi_55,Phi_56]).reshape((1,6))
    Phi_61=7/2*k*S*t
    Phi_62=0
    Phi_63=-4*k*e_xi*G*S*t
    Phi_64=-4*k*e_yi*G*S*t
    Phi_65=2*k*T*t
    Phi_66=1
    Phi_6 = np.array([Phi_61,Phi_62,Phi_63,Phi_64,Phi_65,Phi_66]).reshape((1,6))

    return np.concatenate((Phi_1, Phi_2, Phi_3, Phi_4, Phi_5, Phi_6), axis=0)

def control_input_matrix(oe):

    a = oe.item(0)
    u = oe.item(4) + oe.item(5)
    n = np.sqrt(mu_E/a**3)

    b_1 = np.array([0, 2/n, 0]).reshape((1,3))
    b_2 = np.array([-2/n, 0, 0]).reshape((1,3))
    b_3 = np.array([np.sin(u)/n, 2*np.cos(u)/n, 0]).reshape((1,3))
    b_4 = np.array([-np.cos(u)/n, 2*np.sin(u)/n, 0]).reshape((1,3))
    b_5 = np.array([0, 0, np.cos(u)/n]).reshape((1,3))
    b_6 = np.array([0, 0, np.sin(u)/n]).reshape((1,3))

    return np.concatenate((b_1, b_2, b_3, b_4, b_5, b_6), axis=0)

def dynamics(state, action, oe, t):

    stm = state_transition(oe, t)
    cim = control_input_matrix(oe)

    new_state = stm.dot(state + cim.dot(action))

    return new_state

def kepler_dyanmics(oe, dt):

    a = oe.item(0)
    n = np.sqrt(mu_E/a**3)
    new_oe = np.array([oe.item(0), oe.item(1), oe.item(2), oe.item(3), oe.item(4), oe.item(5) + n*dt]).reshape((6,))

    return new_oe

def dynamics_roe_train(oe, dt, n_t, n_d):

    stm  = np.empty((n_d, n_t, 6, 6))
    cim  = np.empty((n_d, n_t, 6, 3))
    for i in range(n_d):
        for t in range(n_t):
            stm[i,t] = state_transition(oe[i, t], dt[i])
            cim[i,t] = control_input_matrix(oe[i, t])

    return stm, cim

def dynamics_rtn_train(oe, dt, n_t, n_d):
    
    # Note : there is an approximation here, there should be a mean to osculating roe conversion in the middle (for small spacecraft separation the error should be small)

    stm  = np.empty((n_d, n_t, 6, 6))
    cim  = np.empty((n_d, n_t, 6, 3))
    for i in range(n_d):
        for t in range(n_t):
            stm_roe = state_transition(oe[i, t], dt[i])
            cim_roe = control_input_matrix(oe[i, t])
            map_t = map_mtx_roe_to_rtn(oe[i, t])
            if t < n_t - 1:
                map_t_new = map_mtx_roe_to_rtn(oe[i, t+1])
            else: 
                a = oe[i, t][0]
                n = np.sqrt(mu_E/a**3)
                oe_new = oe[i, t] + np.array([0, 0, 0, 0, 0, n*dt.item(i)]).reshape((6,))
                map_t_new = map_mtx_roe_to_rtn(oe_new)
            stm[i,t] = np.matmul(map_t_new, np.matmul(stm_roe, np.linalg.inv(map_t)))
            cim[i,t] = np.matmul(map_t, cim_roe)

    return stm, cim

def dynamics_roe_optimization(oe_0, t_0, horizon, n_time):

    a = oe_0.item(0)
    n = np.sqrt(mu_E/a**3)
    period = 2*np.pi/n

    # Time discretization (given the number of samples defined in rpod_scenario)
    if n_time-1 != 0:
        dt = horizon*period/(n_time-1)
    else:
        dt = 0.

    stm = np.empty(shape=(6, 6, n_time-1), dtype=float)
    cim = np.empty(shape=(6, 3, n_time), dtype=float)
    psi = np.empty(shape=(6, 6, n_time), dtype=float)

    time = np.empty(shape=(n_time,), dtype=float)
    oe = np.empty(shape=(6, n_time), dtype=float)

    # Time 0
    time[0] = t_0
    oe[:,0] = oe_0

    cim[:,:,0] = control_input_matrix(oe[:,0])
    psi[:,:,0] = map_mtx_roe_to_rtn(oe[:,0])

    # Time Loop
    for iter in range(n_time-1):
        
        # Precompute the STM
        stm[:,:,iter] = state_transition(oe[:,iter],dt) # N.B.: is a-dimentional

        # Propagate reference orbit (this assumes keplerian dynamics on the OE, it is an approximation)
        time[iter+1] = time[iter] + dt
        oe[:,iter+1] = np.array([oe_0.item(0), oe_0.item(1), oe_0.item(2), oe_0.item(3), oe_0.item(4), oe_0.item(5) + n*(time.item(iter+1)-t_0)]).reshape((6,))

        # Control input matrix
        cim[:,:,iter+1] = control_input_matrix(oe[:,iter+1]) # N.B.: has dimension [s]
        
        # ROE to RTN map
        psi[:,:,iter+1] = map_mtx_roe_to_rtn(oe[:,iter+1]) # N.B.: has dimension [-,-,-,1/s,1/s,1/s]

    return stm, cim, psi, oe, time, dt

def roe_to_rtn_horizon(roe, oe, n_time):

    rtn = np.empty(shape=(6, n_time), dtype=float)
    for i in range(n_time):
        rtn[:,i] = map_roe_to_rtn(roe[:,i], oe[:,i])

    return rtn

def roe_to_relativeorbit(roe, oe):

    # This function is used just for plotting, assumes keplerian dynamics
    nn = 100
    rtn = np.empty(shape=(6, nn), dtype=float)
    a = oe.item(0)
    n = np.sqrt(mu_E/a**3)
    period = 2*np.pi/n
    dt = period/(nn-1)
    for i in range(nn):
        rtn[:,i] = map_roe_to_rtn(roe, oe)
        oe[5] += n*dt

    return rtn
