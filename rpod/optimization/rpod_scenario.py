import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
from dynamics.orbit_dynamics import map_rtn_to_roe, J2, R_E, mu_E

# ISS Reference orbit parameters
t_0 = 0
a_ref = R_E+416e3 # [m]
M_0 = 68.2333*np.pi/180
oe_0_ref = np.array([a_ref, 0.0005581, 51.6418*np.pi/180, 301.0371*np.pi/180, 26.1813*np.pi/180, M_0]).reshape((6,))
n_ref = np.sqrt(mu_E/a_ref**3)
period_ref = 2*np.pi/n_ref

#  Transfer Horizons Set
n_time_rpod = 100 # Number of time sample in transfer horizon

#  Space Station nominal Docking parameters
width_ss = 108 #[m]
length_ss = 74 #[m]
height_ss = 45 #[m]

dock_wyp_sample = n_time_rpod - 10

E_koz = np.diag([1/(height_ss+15), 1/(length_ss+20), 1/(width_ss+15)]).reshape((3,3))
EE_koz = np.matmul(np.transpose(E_koz), E_koz)
D_pos = np.eye(3, 6, dtype=float)
ED_koz = D_pos * np.diag(E_koz)[:, np.newaxis]
DEED_koz = np.matmul(np.transpose(ED_koz), ED_koz)

# Plume impingement and thrusters maximum dv
plume_imp_angle = np.pi/4
dv_max = 20*0.001 #[m/s]

# Keep-out-zone Ellipse (for plotting)
coefs_ell = np.diag(EE_koz)
rx_ell, ry_ell, rz_ell = 1/np.sqrt(coefs_ell)
u_ell = np.linspace(0, 2 * np.pi, 100)
v_ell = np.linspace(0, np.pi, 100)
x_ell = rx_ell * np.outer(np.cos(u_ell), np.sin(v_ell))
y_ell = ry_ell * np.outer(np.sin(u_ell), np.sin(v_ell))
z_ell = rz_ell * np.outer(np.ones_like(u_ell), np.cos(v_ell))

# SCP data
iter_max_SCP = 20 # [-]
trust_region0 = 200 # [m]
trust_regionf = 1 # [m]
J_tol = 1e-6 # [m/s]

# Function to sample initial conditions
def sample_init_final():
    horizon = np.linspace(1,3, 100) # [# orbits] - transfer horizon
    da = np.linspace(-5, 5, 100) # [m]
    dlambda = np.linspace(-100, 100, 100) # [m]
    de = np.linspace((1/E_koz.item((0,0)))+5, (1/E_koz.item((0,0)))+30, 100)
    di = np.linspace((1/E_koz.item((2,2)))+5, (1/E_koz.item((2,2)))+30, 100)
    ph_de = np.pi/2 + np.linspace(-5, 5, 100)*np.pi/180 # [m]
    ph_di = np.pi/2 + np.linspace(-5, 5, 100)*np.pi/180 # [m]
        
    # Pick horizon
    hrz_i = horizon[np.random.randint(0, 100-1)]

    #  Pick initial passively safe relative orbit around the space station
    da_i = da[np.random.randint(0, 100-1)]
    dlambda_i = dlambda[np.random.randint(0, 100-1)]
    de_i = de[np.random.randint(0, 100-1)]
    di_i = di[np.random.randint(0, 100-1)]
    ph_de_i = ph_de[np.random.randint(0, 100-1)]
    ph_di_i = ph_di[np.random.randint(0, 100-1)]

    state_roe_0 = np.array([da_i, dlambda_i, de_i*np.cos(ph_de_i), de_i*np.sin(ph_de_i), di_i*np.cos(ph_di_i), di_i*np.sin(ph_di_i)]).reshape((6,))

    # Pick docking port
    port_i = np.random.choice([-1,1])
    cone_vertex_displ = 5
    dock_port_t = port_i*(length_ss-cone_vertex_displ)
    state_rtn_target_t = port_i*(length_ss)
    dock_port = np.array([0, dock_port_t, 0]).reshape((3,))
    state_rtn_target = np.array([dock_port.item(0), state_rtn_target_t, dock_port.item(2), 0, 0, 0]).reshape((6,))
    state_roe_target = map_rtn_to_roe(state_rtn_target, oe_0_ref) # NOTE: this is correct just because the dock_port is purely in T-bar
    dock_axis = np.array([0, port_i*1, 0]).reshape((3,))
    dock_wyp_dist = 30
    dock_wyp = state_rtn_target + np.array([dock_wyp_dist*dock_axis.item(0), dock_wyp_dist*dock_axis.item(1), dock_wyp_dist*dock_axis.item(2), 0, 0, 0]).reshape((6,))
    dock_cone_angle = 30*np.pi/180

    # Approach Cone (for plotting)
    cone_length = abs(cone_vertex_displ) + abs(dock_wyp_dist)
    rad_cone = np.linspace(0,cone_length*np.tan(dock_cone_angle),100)
    ang_cone = np.linspace(0,2*np.pi,100)
    Rad_cone, Ang_cone = np.meshgrid(rad_cone, ang_cone) 
    r_cone = Rad_cone * np.cos(Ang_cone)
    n_cone = Rad_cone * np.sin(Ang_cone)
    t_cone = (port_i*Rad_cone / np.tan(dock_cone_angle) + dock_port_t)

    # Docking parameters
    dock_param = {
        'state_roe_target' : state_roe_target,
        'state_rtn_target' : state_rtn_target,
        'dock_axis' : dock_axis,
        'dock_port' : dock_port,
        'dock_cone_angle' : dock_cone_angle,
        'dock_wyp' : dock_wyp
    }
    cone_plotting_param = {
        'r_cone' : r_cone,
        't_cone' : t_cone,
        'n_cone' : n_cone
    }

    return hrz_i, state_roe_0, dock_param, cone_plotting_param

def dock_param_maker(state_rtn_target):
    # Pick docking port
    state_rtn_target_t = state_rtn_target[1]
    port_i = np.sign(state_rtn_target_t)
    cone_vertex_displ = 5
    dock_port_t = port_i*(length_ss-cone_vertex_displ)
    dock_port = np.array([0, dock_port_t, 0]).reshape((3,))
    state_roe_target = map_rtn_to_roe(state_rtn_target, oe_0_ref) # NOTE: this is correct just because the dock_port is purely in T-bar
    dock_axis = np.array([0, port_i*1, 0]).reshape((3,))
    dock_wyp_dist = 30
    dock_wyp = state_rtn_target + np.array([dock_wyp_dist*dock_axis.item(0), dock_wyp_dist*dock_axis.item(1), dock_wyp_dist*dock_axis.item(2), 0, 0, 0]).reshape((6,))
    dock_cone_angle = 30*np.pi/180

    # Approach Cone (for plotting)
    cone_length = abs(cone_vertex_displ) + abs(dock_wyp_dist)
    rad_cone = np.linspace(0,cone_length*np.tan(dock_cone_angle),100)
    ang_cone = np.linspace(0,2*np.pi,100)
    Rad_cone, Ang_cone = np.meshgrid(rad_cone, ang_cone) 
    r_cone = Rad_cone * np.cos(Ang_cone)
    n_cone = Rad_cone * np.sin(Ang_cone)
    t_cone = (port_i*Rad_cone / np.tan(dock_cone_angle) + dock_port_t)

    # Docking parameters
    dock_param = {
        'state_roe_target' : state_roe_target,
        'state_rtn_target' : state_rtn_target,
        'dock_axis' : dock_axis,
        'dock_port' : dock_port,
        'dock_cone_angle' : dock_cone_angle,
        'dock_wyp' : dock_wyp
    }
    cone_plotting_param = {
        'r_cone' : r_cone,
        't_cone' : t_cone,
        'n_cone' : n_cone
    }
    
    return dock_param, cone_plotting_param