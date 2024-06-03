import os
import sys
import argparse

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt
import copy
import time

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments, get_scheduler
from accelerate import Accelerator

from dynamics.orbit_dynamics import map_roe_to_rtn, map_rtn_to_roe, dynamics
from optimization.rpod_scenario import mu_E, EE_koz, dock_wyp_sample
from optimization.ocp import check_koz_constraint
from decision_transformer.art import AutonomousRendezvousTransformer

# select device based on availability of GPU
verbose = False # set to True to get additional print statements
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

'''
    TODO: 
        - 
'''

class RpodDataset(Dataset):
    # Create a Dataset object
    def __init__(self, data, mdp_constr, state_representation='rtn', target=False):
        self.data_stats = data['data_stats']
        self.data = data
        self.n_data, self.max_len, self.n_state = self.data['states'].shape
        self.n_action = self.data['actions'].shape[2]
        self.mdp_constr = mdp_constr
        self.state_representation = state_representation
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ix = torch.randint(self.n_data, (1,))
        states = torch.stack([self.data['states'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_state).float()
        actions = torch.stack([self.data['actions'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_action).float()
        rtgs = torch.stack([self.data['rtgs'][i, :]
                        for i in ix]).view(self.max_len, 1).float()
        goal = torch.stack([self.data['goal'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_state).float()
        timesteps = torch.tensor([[i for i in range(self.max_len)] for _ in ix]).view(self.max_len).long()
        attention_mask = torch.ones(1, self.max_len).view(self.max_len).long()

        horizons = self.data['data_param']['horizons'][ix].item()
        oe = np.transpose(self.data['data_param']['oe'][ix])
        time_discr = self.data['data_param']['time_discr'][ix].item()
        time_sec = self.data['data_param']['time_sec'][ix].reshape((1, self.max_len))

        if self.target == False:
            if not self.mdp_constr:
                return states, actions, rtgs, goal, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix
            else:
                ctgs = torch.stack([self.data['ctgs'][i, :]
                            for i in ix]).view(self.max_len, 1).float()
                return states, actions, rtgs, ctgs, goal, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix
        else:
            target_states = torch.stack([self.data['target_states'][i, :, :]
                            for i in ix]).view(self.max_len-1, self.n_state).float()
            target_actions = torch.stack([self.data['target_actions'][i, :, :]
                            for i in ix]).view(self.max_len, self.n_action).float()
            
            if not self.mdp_constr:
                return states, actions, rtgs, goal, target_states, target_actions, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix
            else:
                ctgs = torch.stack([self.data['ctgs'][i, :]
                            for i in ix]).view(self.max_len, 1).float()
                return states, actions, rtgs, ctgs, goal, target_states, target_actions, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix
    
    def getix(self, ix):
        ix = [ix]
        states = torch.stack([self.data['states'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_state).float().unsqueeze(0)
        actions = torch.stack([self.data['actions'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_action).float().unsqueeze(0)
        rtgs = torch.stack([self.data['rtgs'][i, :]
                        for i in ix]).view(self.max_len, 1).float().unsqueeze(0)
        goal = torch.stack([self.data['goal'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_state).float().unsqueeze(0)
        timesteps = torch.tensor([[i for i in range(self.max_len)] for _ in ix]).view(self.max_len).long().unsqueeze(0)
        attention_mask = torch.ones(1, self.max_len).view(self.max_len).long().unsqueeze(0)

        horizons = torch.tensor(self.data['data_param']['horizons'][ix].item())
        oe = torch.tensor(np.transpose(self.data['data_param']['oe'][ix])).unsqueeze(0)
        time_discr = torch.tensor(self.data['data_param']['time_discr'][ix].item())
        time_sec = torch.tensor(self.data['data_param']['time_sec'][ix].reshape((1, self.max_len))).unsqueeze(0)

        if self.target == False:
            if not self.mdp_constr:
                return states, actions, rtgs, goal, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix
            else:
                ctgs = torch.stack([self.data['ctgs'][i, :]
                            for i in ix]).view(self.max_len, 1).float()
                return states, actions, rtgs, ctgs, goal, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix
        else:
            target_states = torch.stack([self.data['target_states'][i, :, :]
                            for i in ix]).view(self.max_len-1, self.n_state).float().unsqueeze(0)
            target_actions = torch.stack([self.data['target_actions'][i, :, :]
                            for i in ix]).view(self.max_len, self.n_action).float().unsqueeze(0)

            if not self.mdp_constr:
                return states, actions, rtgs, goal, target_states, target_actions, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix
            else:
                ctgs = torch.stack([self.data['ctgs'][i, :]
                            for i in ix]).view(self.max_len, 1).float()
                return states, actions, rtgs, ctgs, goal, target_states, target_actions, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix

    def get_data_size(self):
        return self.n_data

def transformer_import_config(model_name):
    config = {}
    config['model_name'] = model_name
    config['state_representation'] = 'rtn'
    config['dataset_to_use'] = 'both'
    config['mdp_constr'] = True
    config['timestep_norm'] = False
    '''else:
        raise NameError('No transformer model with name', model_name, 'found!')'''

    return config

def get_train_val_test_data(state_representation, dataset_to_use, mdp_constr, model_name, timestep_norm):

    # Import and normalize torch dataset, then save data statistics
    torch_data, data_param = import_dataset_for_DT_eval_vXX(mdp_constr)
    states_norm, states_mean, states_std = normalize(torch_data['torch_states'], timestep_norm)
    actions_norm, actions_mean, actions_std = normalize(torch_data['torch_actions'], timestep_norm)
    goal_norm, goal_mean, goal_std = normalize(torch_data['torch_goal'], timestep_norm)
    target_states_norm = states_norm[:,1:,:].clone().detach()
    target_actions_norm = actions_norm.clone().detach()
    if mdp_constr:
        rtgs_norm, rtgs_mean, rtgs_std = torch_data['torch_rtgs'], None, None
        ctgs_norm, ctgs_mean, ctgs_std = torch_data['torch_ctgs'], None, None
    else:
        rtgs_norm, rtgs_mean, rtgs_std = normalize(torch_data['torch_rtgs'], timestep_norm)
    
    data_stats = {
        'states_mean' : states_mean,
        'states_std' : states_std,
        'actions_mean' : actions_mean,
        'actions_std' : actions_std,
        'rtgs_mean' : rtgs_mean,
        'rtgs_std' : rtgs_std,
        'ctgs_mean' : ctgs_mean if mdp_constr else None,
        'ctgs_std' : ctgs_std if mdp_constr else None,
        'goal_mean' : goal_mean,
        'goal_std' : goal_std
    }

    # Split dataset into training and validation
    n = int(0.9*states_norm.shape[0])
    train_data = {
        'states' : states_norm[:n, :],
        'actions' : actions_norm[:n, :],
        'rtgs' : rtgs_norm[:n, :],
        'ctgs' : ctgs_norm[:n, :] if mdp_constr else None,
        'target_states' : target_states_norm[:n, :],
        'target_actions' : target_actions_norm[:n, :],
        'goal' : goal_norm[:n, :],
        'data_param' : {
            'horizons' : data_param['horizons'][:n],
            'time_discr' : data_param['time_discr'][:n],
            'time_sec' : data_param['time_sec'][:n, :],
            'oe' : data_param['oe'][:n, :]
            },
        'data_stats' : data_stats
        }
    val_data = {
        'states' : states_norm[n:, :],
        'actions' : actions_norm[n:, :],
        'rtgs' : rtgs_norm[n:, :],
        'ctgs' : ctgs_norm[n:, :] if mdp_constr else None,
        'target_states' : target_states_norm[n:, :],
        'target_actions' : target_actions_norm[n:, :],
        'goal' : goal_norm[n:, :],
        'data_param' : {
            'horizons' : data_param['horizons'][n:],
            'time_discr' : data_param['time_discr'][n:],
            'time_sec' : data_param['time_sec'][n:, :],
            'oe' : data_param['oe'][n:, :]
            },
        'data_stats' : data_stats
        }
    
    # Create datasets
    train_dataset = RpodDataset(train_data, mdp_constr)
    val_dataset = RpodDataset(val_data, mdp_constr)
    test_dataset = RpodDataset(val_data, mdp_constr)
    datasets = (train_dataset, val_dataset, test_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=4,
        num_workers=0,
    )
    eval_loader = DataLoader(
        val_dataset,
        sampler=torch.utils.data.RandomSampler(
            val_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=4,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=torch.utils.data.RandomSampler(
            test_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=1,
        num_workers=0,
    )
    dataloaders = (train_loader, eval_loader, test_loader)
    
    return datasets, dataloaders

def import_dataset_for_DT_eval_vXX(mdp_constr):
    # Load the data
    print('Loading data from root/dataset/torch/...', end='')

    data_dir = root_folder + '/dataset'
    data_dir_torch = root_folder + '/dataset/torch/v05'
    states_cvx = torch.load(data_dir_torch + '/torch_states_rtn_cvx.pth')
    states_scp = torch.load(data_dir_torch + '/torch_states_rtn_scp.pth')
    actions_cvx = torch.load(data_dir_torch + '/torch_actions_cvx.pth')
    actions_scp = torch.load(data_dir_torch + '/torch_actions_scp.pth')
    rtgs_cvx = torch.load(data_dir_torch + '/torch_rtgs_cvx.pth')
    rtgs_scp = torch.load(data_dir_torch + '/torch_rtgs_scp.pth')
    ctgs_cvx = torch.load(data_dir_torch + '/torch_ctgs_cvx.pth')
    ctgs_scp = torch.load(data_dir_torch + '/torch_ctgs_scp.pth')
    data_param = np.load(data_dir + '/dataset-rpod-v05-param.npz')

    print('Completed, DATA IS NOT SHUFFLED YET.\n')

    # Output dictionary
    if mdp_constr:
        perm = np.load(data_dir_torch + '/permutation.npy')
        torch_states = torch.concatenate((states_scp, states_cvx), axis=0)[perm]
        torch_actions = torch.concatenate((actions_scp, actions_cvx), axis=0)[perm]
        torch_rtgs = torch.concatenate((rtgs_scp, rtgs_cvx), axis=0)[perm]
        torch_ctgs = torch.concatenate((ctgs_scp, ctgs_cvx), axis=0)[perm]
        goal_timeseq = torch.tensor(np.repeat(data_param['target_state'][:,None,:], torch_states.shape[1], axis=1))
        torch_goal = torch.concatenate((goal_timeseq, goal_timeseq), axis=0)[perm]
        data_param = {
            'horizons' : np.concatenate((data_param['horizons'], data_param['horizons']), axis=0)[perm],
            'time_discr' : np.concatenate((data_param['dtime'], data_param['dtime']), axis=0)[perm],
            'time_sec' : np.concatenate((data_param['time'], data_param['time']), axis=0)[perm],
            'oe' : np.concatenate((data_param['oe'], data_param['oe']), axis=0)[perm]
        }
    else:
        torch_states = states_scp
        torch_actions = actions_scp
        torch_rtgs = rtgs_scp
        torch_ctgs = ctgs_scp
        torch_goal = torch.tensor(np.repeat(data_param['target_state'][:,None,:], torch_states.shape[1], axis=1))
        data_param = {
            'horizons' : data_param['horizons'],
            'time_discr' : data_param['dtime'],
            'time_sec' : data_param['time'],
            'oe' : data_param['oe']
        }

    torch_data = {
        'torch_states' : torch_states,
        'torch_actions' : torch_actions,
        'torch_rtgs' : torch_rtgs,
        'torch_ctgs' : torch_ctgs,
        'torch_goal' : torch_goal
    }

    return torch_data, data_param

def normalize(data, timestep_norm):
    # Normalize and return normalized data, mean and std
    if timestep_norm:
        data_mean = data.mean(dim=0)
        data_std = data.std(dim=0)
        data_norm = (data - data_mean)/(data_std + 1e-6)
    else:
        time_length, size_data = data.shape[1:]
        data_mean = torch.ones((time_length, size_data)) * data.view(-1,size_data).mean(dim=0)
        data_std = torch.ones((time_length, size_data)) * data.view(-1,size_data).std(dim=0)
        data_norm = (data - data_mean)/(data_std + 1e-6)

    return data_norm, data_mean, data_std

def get_DT_model(model_name, train_loader, eval_loader):
    # DT model creation
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
    if 'checkpoint_rtn_ctgrtg' in model_name:
        model = AutonomousRendezvousTransformer(config)
    else:
        model = DecisionTransformerModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT size: {model_size/1000**2:.1f}M parameters")
    model.to(device);

    # DT optimizer and accelerator
    optimizer = AdamW(model.parameters(), lr=3e-5)
    accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )
    accelerator.load_state(root_folder + '/decision_transformer/saved_files/checkpoints/' + model_name)

    return model.eval()

def use_model_for_imitation_learning(model, test_loader, data_sample, state_representation, rtg_perc=1., ctg_perc=1., rtg=None, use_dynamics = True, output_attentions = False):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len
    n_action = test_loader.dataset.n_action
    data_stats = test_loader.dataset.data_stats

    # Unnormalize the data sample and compute orbital period
    if test_loader.dataset.mdp_constr:
        states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
        ctgs_i = ctgs_i.view(1, n_time, 1)
    else:
        states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    states_i_unnorm = (states_i * data_stats['states_std']) + data_stats['states_mean']
    actions_i_unnorm = (actions_i * data_stats['actions_std']) + data_stats['actions_mean']
    if not test_loader.dataset.mdp_constr:
        rtgs_i_unnorm = (rtgs_i * data_stats['rtgs_std']) + data_stats['rtgs_mean']
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.empty(shape=(1, n_time+1), dtype=float)

    # Retrieve decoded states and actions for different inference cases
    roe_true = np.empty(shape=(n_state, n_time), dtype=float)
    roe_ol = np.empty(shape=(n_state, n_time), dtype=float)
    roe_dyn = np.empty(shape=(n_state, n_time), dtype=float)
    rtn_true = np.empty(shape=(n_state, n_time), dtype=float)
    rtn_ol = np.empty(shape=(n_state, n_time), dtype=float)
    rtn_dyn = np.empty(shape=(n_state, n_time), dtype=float)
    dv_true = np.empty(shape=(n_action, n_time), dtype=float)
    dv_ol = np.empty(shape=(n_action, n_time), dtype=float)
    dv_dyn = np.empty(shape=(n_action, n_time), dtype=float)

    # Open-loop initialization
    states_ol = states_i[:, 0, :][None, :, :].float().to(device)
    actions_ol = torch.zeros((1, 1, n_action), device=device).float()
    if rtg is None:
        rtgs_ol = rtgs_i[:, 0, :].view(1, 1, 1).float().to(device)*rtg_perc
    else:
        rtgs_ol = torch.tensor(rtg).view(1, 1, 1).float().to(device)
    #print(rtgs_ol)
    if test_loader.dataset.mdp_constr:
        ctgs_ol = ctgs_i[:, 0, :].view(1, 1, 1).float().to(device)*ctg_perc
    goal_ol = goal_i[:, 0, :][None, :, :].float().to(device)
    timesteps_ol = timesteps_i[:, 0][None, :].long().to(device)
    attention_mask_ol = attention_mask_i[:, 0][None, :].long().to(device)

    state_ol_t_unnorm = (states_ol[:,0,:].to(device) * data_stats['states_std'][0].to(device)) + data_stats['states_mean'][0].to(device)
    if state_representation == 'roe':
        roe_ol[:, 0] = [state_ol_t_unnorm[0,i].item() for i in range(n_state)]
        rtn_ol[:, 0] = map_roe_to_rtn(roe_ol[:, 0], oe[:, 0])
    elif state_representation == 'rtn':
        rtn_ol[:, 0] = [state_ol_t_unnorm[0,i].item() for i in range(n_state)]
        roe_ol[:, 0] = map_rtn_to_roe(rtn_ol[:, 0], oe[:, 0])

    # Dynamics-in-the-loop initialization
    if use_dynamics:
        states_dyn = states_i[:, 0, :][None, :, :].float().to(device)
        actions_dyn = torch.zeros((1, 1, n_action), device=device).float()
        if rtg is None:
            rtgs_dyn = rtgs_i[:, 0, :].view(1, 1, 1).float().to(device)*rtg_perc
        else:
            rtgs_dyn = torch.tensor(rtg).view(1, 1, 1).float().to(device)
        #print(rtgs_dyn)
        #rtgs_dyn = rtgs_i[:, 0, :].view(1, 1, 1).float().to(device)*rtg_perc
        if test_loader.dataset.mdp_constr:
            ctgs_dyn = ctgs_i[:, 0, :].view(1, 1, 1).float().to(device)*ctg_perc
        goal_dyn = goal_i[:, 0, :][None, :, :].float().to(device)
        timesteps_dyn = timesteps_i[:, 0][None, :].long().to(device)
        attention_mask_dyn = attention_mask_i[:, 0][None, :].long().to(device)

        state_dyn_t_unnorm = (states_dyn[:,0,:].to(device) * data_stats['states_std'][0].to(device)) + data_stats['states_mean'][0].to(device)
        if state_representation == 'roe':
            roe_dyn[:, 0] = [state_dyn_t_unnorm[0,i].item() for i in range(n_state)]
            rtn_dyn[:, 0] = map_roe_to_rtn(roe_dyn[:, 0], oe[:, 0])
        elif state_representation == 'rtn':
            rtn_dyn[:, 0] = [state_dyn_t_unnorm[0,i].item() for i in range(n_state)]
            roe_dyn[:, 0] = map_rtn_to_roe(rtn_dyn[:, 0], oe[:, 0])

    attentions_dyn = []
    attentions_ol_a = []
    attentions_ol_s = []
    # For loop trajectory generation
    for t in range(n_time):
        
        ##### Decode true data sample
        state_true_t = states_i_unnorm[0,t].cpu()
        action_true_t = actions_i_unnorm[0,t].cpu()
        if state_representation == 'roe':
            roe_true[:, t] = [state_true_t[i].item() for i in range(n_state)]
            rtn_true[:, t] = map_roe_to_rtn(roe_true[:, t], oe[:, t])
        elif state_representation == 'rtn':
            rtn_true[:, t] = [state_true_t[i].item() for i in range(n_state)]
            roe_true[:, t] = map_rtn_to_roe(rtn_true[:, t], oe[:, t])
        dv_true[:, t] = [action_true_t[i].item() for i in range(n_action)]

        ##### Open-loop inference
        # Compute action pred for open-loop model
        with torch.no_grad():
            if test_loader.dataset.mdp_constr:
                output_ol = model(
                states=states_ol.to(device),
                actions=actions_ol.to(device),
                goal=goal_ol.to(device),
                returns_to_go=rtgs_ol.to(device),
                constraints_to_go=ctgs_ol.to(device),
                timesteps=timesteps_ol.to(device),
                attention_mask=attention_mask_ol.to(device),
                return_dict=True,
                output_attentions=output_attentions
                )
            else:
                output_ol = model(
                states=states_ol.to(device),
                actions=actions_ol.to(device),
                goal=goal_ol.to(device),
                returns_to_go=rtgs_ol.to(device),
                timesteps=timesteps_ol.to(device),
                attention_mask=attention_mask_ol.to(device),
                return_dict=True,
                output_attentions=output_attentions
                )
        '''if test_loader.dataset.mdp_constr:
            (_state_preds_ol, action_preds_ol) = output_ol
        else:
            (_state_preds_ol, action_preds_ol, _rtgs_preds_ol) = output_ol'''
        action_preds_ol = output_ol.action_preds
        attentions_ol_a.append(torch.cat(output_ol.attentions).to('cpu'))

        action_ol_t = action_preds_ol[0,t].cpu()
        actions_ol[:,-1,:] = action_preds_ol[0,t][None,None,:].float()
        action_ol_t_unnorm = (action_ol_t.to(device) * (data_stats['actions_std'][t].to(device)+1e-6)) + data_stats['actions_mean'][t].to(device)
        dv_ol[:, t] = [action_ol_t_unnorm[i].item() for i in range(n_action)]

        with torch.no_grad():
            if test_loader.dataset.mdp_constr:
                output_ol = model(
                states=states_ol.to(device),
                actions=actions_ol.to(device),
                goal=goal_ol.to(device),
                returns_to_go=rtgs_ol.to(device),
                constraints_to_go=ctgs_ol.to(device),
                timesteps=timesteps_ol.to(device),
                attention_mask=attention_mask_ol.to(device),
                return_dict=True,
                output_attentions=output_attentions
                )
            else:
                output_ol = model(
                states=states_ol.to(device),
                actions=actions_ol.to(device),
                goal=goal_ol.to(device),
                returns_to_go=rtgs_ol.to(device),
                timesteps=timesteps_ol.to(device),
                attention_mask=attention_mask_ol.to(device),
                return_dict=True,
                output_attentions=output_attentions
                )

        '''if test_loader.dataset.mdp_constr:
            (state_preds_ol, _action_preds_ol) = output_ol
        else:
            (state_preds_ol, _action_preds_ol, _return_preds_ol) = output_ol'''
        state_preds_ol = output_ol.state_preds
        attentions_ol_s.append(torch.cat(output_ol.attentions).to('cpu'))
        state_ol_t = state_preds_ol[0,t].cpu()

        # Open-loop propagation of state variable
        if t != n_time-1:
            states_ol = torch.cat((states_ol, torch.tensor(state_preds_ol[:, t][None,:,:]).to(device)), dim=1).float()
            state_ol_t_unnorm = (state_ol_t.to(device) * (data_stats['states_std'][t+1].to(device)+1e-6)) + data_stats['states_mean'][t+1].to(device)
            if state_representation == 'roe':
                roe_ol[:, t+1] = [state_ol_t_unnorm[i].item() for i in range(n_state)]
                rtn_ol[:, t+1] = map_roe_to_rtn(roe_ol[:, t+1], oe[:, t+1])
            elif state_representation == 'rtn':
                rtn_ol[:, t+1] = [state_ol_t_unnorm[i].item() for i in range(n_state)]
                roe_ol[:, t+1] = map_rtn_to_roe(rtn_ol[:, t+1], oe[:, t+1])
            if test_loader.dataset.mdp_constr:
                reward_ol_t = - la.norm(dv_ol[:, t])
                rtgs_ol = torch.cat((rtgs_ol, rtgs_ol[0, -1].view(1, 1, 1) - reward_ol_t), dim=1)
            else:
                rtgs_ol = torch.cat((rtgs_ol, rtgs_i[0,t+1][None,None,:].to(device)), dim=1).float()
            if test_loader.dataset.mdp_constr:
                _, viol_ol = check_koz_constraint(rtn_ol, t+1)
                ctgs_ol = torch.cat((ctgs_ol, ctgs_ol[0, -1].view(1, 1, 1) - viol_ol[-1]), dim=1)
            goal_ol = torch.cat((goal_ol, goal_i[:, t+1][None,:].to(device)), dim=1).float()
            timesteps_ol = torch.cat((timesteps_ol, timesteps_i[:, t+1][None,:].to(device)), dim=1).long()
            attention_mask_ol = torch.cat((attention_mask_ol, attention_mask_i[:, t+1][None,:].to(device)), dim=1).long()
            actions_ol = torch.cat((actions_ol, torch.zeros((1, 1, n_action), device=device).float()), dim=1)
        
        ##### Dynamics inference        
        if use_dynamics:

            # Compute action pred for dynamics model
            with torch.no_grad():
                if test_loader.dataset.mdp_constr:
                    output_dyn = model(
                    states=states_dyn.to(device),
                    actions=actions_dyn.to(device),
                    goal=goal_dyn.to(device),
                    returns_to_go=rtgs_dyn.to(device),
                    constraints_to_go=ctgs_dyn.to(device),
                    timesteps=timesteps_dyn.to(device),
                    attention_mask=attention_mask_dyn.to(device),
                    return_dict=True,
                    output_attentions=output_attentions
                    )
                else:
                    output_dyn = model(
                    states=states_dyn.to(device),
                    actions=actions_dyn.to(device),
                    goal=goal_dyn.to(device),
                    returns_to_go=rtgs_dyn.to(device),
                    timesteps=timesteps_dyn.to(device),
                    attention_mask=attention_mask_dyn.to(device),
                    return_dict=True,
                    output_attentions=output_attentions
                    )
            '''if test_loader.dataset.mdp_constr:
                (_state_preds_dyn, action_preds_dyn) = output_dyn
            else:
                (_state_preds_dyn, action_preds_dyn, _return_preds_dyn) = output_dyn'''
            action_preds_dyn = output_dyn.action_preds
            attentions_dyn.append(torch.cat(output_dyn.attentions).to('cpu'))
            action_dyn_t = action_preds_dyn[0,t].cpu()
            actions_dyn[:,-1,:] = action_preds_dyn[0,t][None,None,:].float()
            action_dyn_t_unnorm = (action_dyn_t.to(device) * (data_stats['actions_std'][t].to(device)+1e-6)) + data_stats['actions_mean'][t].to(device)
            dv_dyn[:, t] = [action_dyn_t_unnorm[i].item() for i in range(n_action)]

            # Dynamics propagation of state variable 
            if t != n_time-1:
                #roe_dyn[:, t] = map_rtn_to_roe(rtn_dyn[:, t], oe[:, t])
                roe_dyn[:, t+1] = dynamics(roe_dyn[:, t], dv_dyn[:, t], oe[:, t], dt)
                rtn_dyn[:, t+1] = map_roe_to_rtn(roe_dyn[:, t+1], oe[:, t+1])
                if state_representation == 'roe':
                    states_dyn_norm = (torch.tensor(roe_dyn[:, t+1]).to(device) - data_stats['states_mean'][t+1].to(device)) / (data_stats['states_std'][t+1].to(device)+1e-6)
                elif state_representation == 'rtn':
                    states_dyn_norm = (torch.tensor(rtn_dyn[:, t+1]).to(device) - data_stats['states_mean'][t+1].to(device)) / (data_stats['states_std'][t+1].to(device)+1e-6)
                states_dyn = torch.cat((states_dyn, states_dyn_norm[None,None,:]), dim=1).to(device).float()
                
                if test_loader.dataset.mdp_constr:
                    reward_dyn_t = - la.norm(dv_dyn[:, t])
                    rtgs_dyn = torch.cat((rtgs_dyn, rtgs_dyn[0, -1].view(1, 1, 1) - reward_dyn_t), dim=1)
                else:
                    rtgs_dyn = torch.cat((rtgs_dyn, rtgs_i[0,t+1][None,None,:].to(device)), dim=1).float()
                if test_loader.dataset.mdp_constr:
                    _, viol_dyn = check_koz_constraint(rtn_dyn, t+1)
                    ctgs_dyn = torch.cat((ctgs_dyn, ctgs_dyn[0, -1].view(1, 1, 1) - viol_dyn[-1]), dim=1)
                goal_dyn = torch.cat((goal_dyn, goal_i[:, t+1][None,:].to(device)), dim=1).float()
                timesteps_dyn = torch.cat((timesteps_dyn, timesteps_i[:, t+1][None,:].to(device)), dim=1).long()
                attention_mask_dyn = torch.cat((attention_mask_dyn, attention_mask_i[:, t+1][None,:].to(device)), dim=1).long()
                actions_dyn = torch.cat((actions_dyn, torch.zeros((1, 1, n_action), device=device).float()), dim=1)
        
        time_orb[:, t] = time_sec[:, t]/period_ref
    time_orb[:, n_time] = time_orb[:, n_time-1] + dt/period_ref

    # Pack trajectory's data in a dictionary
    DT_trajectory = {
        'rtn_true' : rtn_true,
        'rtn_dyn' : rtn_dyn,
        'rtn_ol' : rtn_ol,
        'roe_true' : roe_true,
        'roe_dyn' : roe_dyn,
        'roe_ol' : roe_ol,
        'dv_true' : dv_true,
        'dv_dyn' : dv_dyn,
        'dv_ol' : dv_ol,
        'time_orb' : time_orb
    }
    DT_attentions = {
        'dyn' : attentions_dyn,
        'ol_a' : attentions_ol_a,
        'ol_s' : attentions_ol_s
    }

    return DT_trajectory, DT_attentions

'''def model_inference_dyn(model, test_loader, data_sample, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len
    n_action = test_loader.dataset.n_action
    data_stats = test_loader.dataset.data_stats

    # Unnormalize the data sample and compute orbital period
    if test_loader.dataset.mdp_constr:
        states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
        ctgs_i = ctgs_i.view(1, n_time, 1) # probably not needed??
    else:
        states_i, actions_i, rtgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.empty(shape=(1, n_time+1), dtype=float)

    # Retrieve decoded states and actions for different inference cases
    roe_dyn = np.empty(shape=(n_state, n_time), dtype=float)
    rtn_dyn = np.empty(shape=(n_state, n_time), dtype=float)
    dv_dyn = np.empty(shape=(n_action, n_time), dtype=float)

    runtime0_DT = time.time()
    # Dynamics-in-the-loop initialization
    states_dyn = states_i[:, 0, :][None, :, :].float().to(device)
    actions_dyn = torch.zeros((1, 1, n_action), device=device).float()
    if rtg is None:
        rtgs_dyn = rtgs_i[:, 0, :].view(1, 1, 1).float().to(device)*rtg_perc
    else:
        rtgs_dyn = torch.tensor(rtg).view(1, 1, 1).float().to(device)
    if test_loader.dataset.mdp_constr:
        ctgs_dyn = ctgs_i[:, 0, :].view(1, 1, 1).float().to(device)*ctg_perc
    timesteps_dyn = timesteps_i[:, 0][None, :].long().to(device)
    attention_mask_dyn = attention_mask_i[:, 0][None, :].long().to(device)

    state_dyn_t_unnorm = (states_dyn[:,0,:].to(device) * data_stats['states_std'][0].to(device)) + data_stats['states_mean'][0].to(device)
    if state_representation == 'roe':
        roe_dyn[:, 0] = [state_dyn_t_unnorm[0,i].item() for i in range(n_state)]
        rtn_dyn[:, 0] = map_roe_to_rtn(roe_dyn[:, 0], oe[:, 0])
    elif state_representation == 'rtn':
        rtn_dyn[:, 0] = [state_dyn_t_unnorm[0,i].item() for i in range(n_state)]
        roe_dyn[:, 0] = map_rtn_to_roe(rtn_dyn[:, 0], oe[:, 0])

    # For loop trajectory generation
    for t in range(n_time):

        ##### Dynamics inference        
        # Compute action pred for dynamics model
        with torch.no_grad():
            if test_loader.dataset.mdp_constr:
                output_dyn = model(
                    states=states_dyn.to(device),
                    actions=actions_dyn.to(device),
                    rewards=None,
                    returns_to_go=rtgs_dyn.to(device),
                    constraints_to_go=ctgs_dyn.to(device),
                    timesteps=timesteps_dyn.to(device),
                    attention_mask=attention_mask_dyn.to(device),
                    return_dict=False,
                )
                (_state_preds_dyn, action_preds_dyn) = output_dyn
            else:
                output_dyn = model(
                    states=states_dyn.to(device),
                    actions=actions_dyn.to(device),
                    rewards=None,
                    returns_to_go=rtgs_dyn.to(device),
                    timesteps=timesteps_dyn.to(device),
                    attention_mask=attention_mask_dyn.to(device),
                    return_dict=False,
                )
                (_state_preds_dyn, action_preds_dyn, _return_preds_dyn) = output_dyn

        action_dyn_t = action_preds_dyn[0,t].cpu()
        actions_dyn[:,-1,:] = action_preds_dyn[0,t][None,None,:].float()
        action_dyn_t_unnorm = (action_dyn_t.to(device) * (data_stats['actions_std'][t].to(device)+1e-6)) + data_stats['actions_mean'][t].to(device)
        dv_dyn[:, t] = [action_dyn_t_unnorm[i].item() for i in range(n_action)]

        # Dynamics propagation of state variable 
        if t != n_time-1:
            #roe_dyn[:, t] = map_rtn_to_roe(rtn_dyn[:, t], oe[:, t])
            roe_dyn[:, t+1] = dynamics(roe_dyn[:, t], dv_dyn[:, t], oe[:, t], dt)
            rtn_dyn[:, t+1] = map_roe_to_rtn(roe_dyn[:, t+1], oe[:, t+1])
            if state_representation == 'roe':
                states_dyn_norm = (torch.tensor(roe_dyn[:, t+1]).to(device) - data_stats['states_mean'][t+1].to(device)) / (data_stats['states_std'][t+1].to(device)+1e-6)
            elif state_representation == 'rtn':
                states_dyn_norm = (torch.tensor(rtn_dyn[:, t+1]).to(device) - data_stats['states_mean'][t+1].to(device)) / (data_stats['states_std'][t+1].to(device)+1e-6)
            states_dyn = torch.cat((states_dyn, states_dyn_norm[None,None,:]), dim=1).to(device).float()
            
            if test_loader.dataset.mdp_constr:
                reward_dyn_t = - la.norm(dv_dyn[:, t])
                rtgs_dyn = torch.cat((rtgs_dyn, rtgs_dyn[0, -1].view(1, 1, 1) - reward_dyn_t), dim=1)
                _, viol_dyn = check_koz_constraint(rtn_dyn, t+1)
                ctgs_dyn = torch.cat((ctgs_dyn, ctgs_dyn[0, -1].view(1, 1, 1) - viol_dyn[-1]), dim=1)
            else:
                rtgs_dyn = torch.cat((rtgs_dyn, rtgs_i[0,t+1][None,None,:].to(device)), dim=1).float()
            timesteps_dyn = torch.cat((timesteps_dyn, timesteps_i[:, t+1][None,:].to(device)), dim=1).long()
            attention_mask_dyn = torch.cat((attention_mask_dyn, attention_mask_i[:, t+1][None,:].to(device)), dim=1).long()
            actions_dyn = torch.cat((actions_dyn, torch.zeros((1, 1, n_action), device=device).float()), dim=1)
        
        time_orb[:, t] = time_sec[:, t]/period_ref
    time_orb[:, n_time] = time_orb[:, n_time-1] + dt/period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    DT_trajectory = {
        'rtn_dyn' : rtn_dyn,
        'roe_dyn' : roe_dyn,
        'dv_dyn' : dv_dyn,
        'time_orb' : time_orb
    }
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT

    return DT_trajectory, runtime_DT'''

def torch_model_inference_dyn(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None, ctg_clipped=True):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)
    data_stats['goal_mean'] = data_stats['goal_mean'].float().to(device)
    data_stats['goal_std'] = data_stats['goal_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    if test_loader.dataset.mdp_constr:
        states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
        ctgs_i = ctgs_i.view(1, n_time, 1).to(device) # probably not needed??
    else:
        states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    states_i = states_i.to(device)
    rtgs_i = rtgs_i.to(device)
    goal_i = goal_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
    stm = torch.from_numpy(stm).float().to(device)
    cim = torch.from_numpy(cim).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)
    psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

    # Retrieve decoded states and actions for different inference cases
    roe_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    rtn_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    dv_dyn = torch.empty(size=(n_action, n_time), device=device).float()
    states_dyn = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    if test_loader.dataset.mdp_constr:
        ctgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()

    runtime0_DT = time.time()
    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_dyn[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_dyn[:,0,:] = rtg
    if test_loader.dataset.mdp_constr:
        ctgs_dyn[:,0,:] = ctgs_i[:,0,:]*ctg_perc

    if state_representation == 'roe':
        roe_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        rtn_dyn[:, 0] = psi[:,:,0] @ roe_dyn[:,0]
    elif state_representation == 'rtn':
        rtn_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        roe_dyn[:, 0] = psi_inv[:,:,0] @ rtn_dyn[:,0]
    
    # For loop trajectory generation
    for t in np.arange(n_time):
        
        ##### Dynamics inference        
        # Compute action pred for dynamics model
        with torch.no_grad():
            if test_loader.dataset.mdp_constr:
                output_dyn = model(
                    states=states_dyn[:,:t+1,:],
                    actions=actions_dyn[:,:t+1,:],
                    goal=goal_i[:,:t+1,:],
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
                    goal=goal_i[:,:t+1,:],
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
        if t != n_time-1:
            roe_dyn[:,t+1] = stm[:,:,t] @ (roe_dyn[:,t] + cim[:,:,t] @ dv_dyn[:,t])
            rtn_dyn[:,t+1] = psi[:,:,t+1] @ roe_dyn[:,t+1]
            if state_representation == 'roe':
                states_dyn_norm = (roe_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
            elif state_representation == 'rtn':
                states_dyn_norm = (rtn_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
            states_dyn[:,t+1,:] = states_dyn_norm
            
            if test_loader.dataset.mdp_constr:
                reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t])
                rtgs_dyn[:,t+1,:] = rtgs_dyn[0,t] - reward_dyn_t
                viol_dyn = torch_check_koz_constraint(rtn_dyn[:,t+1], t+1)
                ctgs_dyn[:,t+1,:] = ctgs_dyn[0,t] - (viol_dyn if (not ctg_clipped) else 0)
            else:
                '''reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t])
                rtgs_dyn[:,t+1,:] = rtgs_dyn[0,t] - (reward_dyn_t/(data_stats['rtgs_std'][t]+1e-6))'''
                rtgs_dyn[:,t+1,:] = rtgs_i[0,t+1]
            actions_dyn[:,t+1,:] = 0
        
        time_orb[:, t] = time_sec[:, t]/period_ref
    time_orb[:, n_time] = time_orb[:, n_time-1] + dt/period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'rtn_dyn' : rtn_dyn.cpu().numpy(),
        'roe_dyn' : roe_dyn.cpu().numpy(),
        'dv_dyn' : dv_dyn.cpu().numpy(),
        'time_orb' : time_orb
    }

    return DT_trajectory, runtime_DT

def torch_check_koz_constraint(states_rtn, n_time):

    # Ellipse equation check for a single instant in the trajectory
    constr_koz = (states_rtn[:3]) @ (torch.from_numpy(EE_koz).float().to(device) @ states_rtn[:3])
    if (constr_koz < 1) and (n_time < dock_wyp_sample):
        constr_koz_violation = 1.
    else:
        constr_koz_violation = 0.

    return constr_koz_violation

'''def model_inference_ol(model, test_loader, data_sample, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len
    n_action = test_loader.dataset.n_action
    data_stats = test_loader.dataset.data_stats

    # Unnormalize the data sample and compute orbital period
    if test_loader.dataset.mdp_constr:
        states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
        ctgs_i = ctgs_i.view(1, n_time, 1)
    else:
        states_i, actions_i, rtgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    states_i_unnorm = (states_i * data_stats['states_std']) + data_stats['states_mean']
    actions_i_unnorm = (actions_i * data_stats['actions_std']) + data_stats['actions_mean']
    if not test_loader.dataset.mdp_constr:
        rtgs_i_unnorm = (rtgs_i * data_stats['rtgs_std']) + data_stats['rtgs_mean']
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.empty(shape=(1, n_time+1), dtype=float)

    # Retrieve decoded states and actions for different inference cases
    roe_ol = np.empty(shape=(n_state, n_time), dtype=float)
    rtn_ol = np.empty(shape=(n_state, n_time), dtype=float)
    dv_ol = np.empty(shape=(n_action, n_time), dtype=float)
    
    runtime0_DT = time.time()
    # Open-loop initialization
    states_ol = states_i[:, 0, :][None, :, :].float().to(device)
    actions_ol = torch.zeros((1, 1, n_action), device=device).float()
    if rtg is None:
        rtgs_ol = rtgs_i[:, 0, :].view(1, 1, 1).float().to(device)*rtg_perc
    else:
        rtgs_ol = torch.tensor(rtg).view(1, 1, 1).float().to(device)
    if test_loader.dataset.mdp_constr:
        ctgs_ol = ctgs_i[:, 0, :].view(1, 1, 1).float().to(device)*ctg_perc
    timesteps_ol = timesteps_i[:, 0][None, :].long().to(device)
    attention_mask_ol = attention_mask_i[:, 0][None, :].long().to(device)

    state_ol_t_unnorm = (states_ol[:,0,:].to(device) * data_stats['states_std'][0].to(device)) + data_stats['states_mean'][0].to(device)
    if state_representation == 'roe':
        roe_ol[:, 0] = [state_ol_t_unnorm[0,i].item() for i in range(n_state)]
        rtn_ol[:, 0] = map_roe_to_rtn(roe_ol[:, 0], oe[:, 0])
    elif state_representation == 'rtn':
        rtn_ol[:, 0] = [state_ol_t_unnorm[0,i].item() for i in range(n_state)]
        roe_ol[:, 0] = map_rtn_to_roe(rtn_ol[:, 0], oe[:, 0])

    # For loop trajectory generation
    for t in range(n_time):

        ##### Open-loop inference
        # Compute action pred for open-loop model
        with torch.no_grad():
            if test_loader.dataset.mdp_constr:
                output_ol = model(
                    states=states_ol.to(device),
                    actions=actions_ol.to(device),
                    rewards=None,
                    returns_to_go=rtgs_ol.to(device),
                    constraints_to_go=ctgs_ol.to(device),
                    timesteps=timesteps_ol.to(device),
                    attention_mask=attention_mask_ol.to(device),
                    return_dict=False,
                )
                (_state_preds_ol, action_preds_ol) = output_ol
            else:
                output_ol = model(
                    states=states_ol.to(device),
                    actions=actions_ol.to(device),
                    rewards=None,
                    returns_to_go=rtgs_ol.to(device),
                    timesteps=timesteps_ol.to(device),
                    attention_mask=attention_mask_ol.to(device),
                    return_dict=False,
                )
                (_state_preds_ol, action_preds_ol, _rtgs_preds_ol) = output_ol

        action_ol_t = action_preds_ol[0,t].cpu()
        actions_ol[:,-1,:] = action_preds_ol[0,t][None,None,:].float()
        action_ol_t_unnorm = (action_ol_t.to(device) * (data_stats['actions_std'][t].to(device)+1e-6)) + data_stats['actions_mean'][t].to(device)
        dv_ol[:, t] = [action_ol_t_unnorm[i].item() for i in range(n_action)]

        # Compute states pred for open-loop model
        with torch.no_grad():
            if test_loader.dataset.mdp_constr:
                output_ol = model(
                    states=states_ol.to(device),
                    actions=actions_ol.to(device),
                    rewards=None,
                    returns_to_go=rtgs_ol.to(device),
                    constraints_to_go=ctgs_ol.to(device),
                    timesteps=timesteps_ol.to(device),
                    attention_mask=attention_mask_ol.to(device),
                    return_dict=False,
                )
                (state_preds_ol, _action_preds_ol) = output_ol
            else:
                output_ol = model(
                    states=states_ol.to(device),
                    actions=actions_ol.to(device),
                    rewards=None,
                    returns_to_go=rtgs_ol.to(device),
                    timesteps=timesteps_ol.to(device),
                    attention_mask=attention_mask_ol.to(device),
                    return_dict=False,
                )
                (state_preds_ol, _action_preds_ol, _return_preds_ol) = output_ol

        state_ol_t = state_preds_ol[0,t].cpu()

        # Open-loop propagation of state variable
        if t != n_time-1:
            states_ol = torch.cat((states_ol, torch.tensor(state_preds_ol[:, t][None,:,:]).to(device)), dim=1).float()
            state_ol_t_unnorm = (state_ol_t.to(device) * (data_stats['states_std'][t+1].to(device)+1e-6)) + data_stats['states_mean'][t+1].to(device)
            if state_representation == 'roe':
                roe_ol[:, t+1] = [state_ol_t_unnorm[i].item() for i in range(n_state)]
                rtn_ol[:, t+1] = map_roe_to_rtn(roe_ol[:, t+1], oe[:, t+1])
            elif state_representation == 'rtn':
                rtn_ol[:, t+1] = [state_ol_t_unnorm[i].item() for i in range(n_state)]
                roe_ol[:, t+1] = map_rtn_to_roe(rtn_ol[:, t+1], oe[:, t+1])

            if test_loader.dataset.mdp_constr:
                reward_ol_t = - la.norm(dv_ol[:, t])
                rtgs_ol = torch.cat((rtgs_ol, rtgs_ol[0, -1].view(1, 1, 1) - reward_ol_t), dim=1)
                _, viol_ol = check_koz_constraint(rtn_ol, t+1)
                ctgs_ol = torch.cat((ctgs_ol, ctgs_ol[0, -1].view(1, 1, 1) - viol_ol[-1]), dim=1)
            else:
                rtgs_ol = torch.cat((rtgs_ol, rtgs_i[0,t+1][None,None,:].to(device)), dim=1).float()
            timesteps_ol = torch.cat((timesteps_ol, timesteps_i[:, t+1][None,:].to(device)), dim=1).long()
            attention_mask_ol = torch.cat((attention_mask_ol, attention_mask_i[:, t+1][None,:].to(device)), dim=1).long()
            actions_ol = torch.cat((actions_ol, torch.zeros((1, 1, n_action), device=device).float()), dim=1)
        
        time_orb[:, t] = time_sec[:, t]/period_ref
    time_orb[:, n_time] = time_orb[:, n_time-1] + dt/period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    DT_trajectory = {
        'rtn_ol' : rtn_ol,
        'roe_ol' : roe_ol,
        'dv_ol' : dv_ol,
        'time_orb' : time_orb
    }
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT

    return DT_trajectory, runtime_DT'''

def torch_model_inference_ol(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None, ctg_clipped=True):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)
    data_stats['goal_mean'] = data_stats['goal_mean'].float().to(device)
    data_stats['goal_std'] = data_stats['goal_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    if test_loader.dataset.mdp_constr:
        states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
        ctgs_i = ctgs_i.view(1, n_time, 1)
    else:
        states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    states_i = states_i.to(device)
    rtgs_i = rtgs_i.to(device)
    goal_i = goal_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
    stm = torch.from_numpy(stm).float().to(device)
    cim = torch.from_numpy(cim).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)
    psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

    # Retrieve decoded states and actions for different inference cases
    roe_ol = torch.empty(size=(n_state, n_time), device=device).float()
    rtn_ol = torch.empty(size=(n_state, n_time), device=device).float()
    dv_ol = torch.empty(size=(n_action, n_time), device=device).float()
    states_ol = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_ol = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_ol = torch.empty(size=(1, n_time, 1), device=device).float()
    if test_loader.dataset.mdp_constr:
        ctgs_ol = torch.empty(size=(1, n_time, 1), device=device).float()
    
    runtime0_DT = time.time()
    # Open-loop initialization
    states_ol[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_ol[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_ol[:,0,:] = rtg
    if test_loader.dataset.mdp_constr:
        ctgs_ol[:,0,:] = ctgs_i[:,0,:]*ctg_perc

    if state_representation == 'roe':
        roe_ol[:, 0] = (states_ol[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        rtn_ol[:, 0] = psi[:,:,0] @ roe_ol[:,0]
    elif state_representation == 'rtn':
        rtn_ol[:, 0] = (states_ol[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        roe_ol[:, 0] = psi_inv[:,:,0] @ rtn_ol[:,0]

    # For loop trajectory generation
    for t in np.arange(n_time):

        ##### Open-loop inference
        # Compute action pred for open-loop model
        with torch.no_grad():
            if test_loader.dataset.mdp_constr:
                output_ol = model(
                    states=states_ol[:,:t+1,:],
                    actions=actions_ol[:,:t+1,:],
                    goal=goal_i[:,:t+1,:],
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
                    goal=goal_i[:,:t+1,:],
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
                    goal=goal_i[:,:t+1,:],
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
                    goal=goal_i[:,:t+1,:],
                    returns_to_go=rtgs_ol[:,:t+1,:],
                    timesteps=timesteps_i[:,:t+1],
                    attention_mask=attention_mask_i[:,:t+1],
                    return_dict=False,
                )
                (state_preds_ol, _, _) = output_ol

        state_ol_t = state_preds_ol[0,t]

        # Open-loop propagation of state variable
        if t != n_time-1:
            states_ol[:,t+1,:] = state_ol_t
            if state_representation == 'roe':
                roe_ol[:,t+1] = (state_ol_t * data_stats['states_std'][t+1]) + data_stats['states_mean'][t+1]
                rtn_ol[:,t+1] = psi[:,:,t+1] @ roe_ol[:,t+1]
            elif state_representation == 'rtn':
                rtn_ol[:,t+1] = (state_ol_t * data_stats['states_std'][t+1]) + data_stats['states_mean'][t+1]
                roe_ol[:,t+1] = psi_inv[:,:,t+1] @ rtn_ol[:,t+1]

            if test_loader.dataset.mdp_constr:
                reward_ol_t = - torch.linalg.norm(dv_ol[:, t])
                rtgs_ol[:,t+1,:] = rtgs_ol[0,t] - reward_ol_t
                viol_ol = torch_check_koz_constraint(rtn_ol[:,t+1], t+1)
                ctgs_ol[:,t+1,:] = ctgs_ol[0,t] - (viol_ol if (not ctg_clipped) else 0)
            else:
                rtgs_ol[:,t+1,:] = rtgs_i[0,t+1]
            actions_ol[:,t+1,:] = 0
        
        time_orb[:, t] = time_sec[:, t]/period_ref
    time_orb[:, n_time] = time_orb[:, n_time-1] + dt/period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'rtn_ol' : rtn_ol.cpu().numpy(),
        'roe_ol' : roe_ol.cpu().numpy(),
        'dv_ol' : dv_ol.cpu().numpy(),
        'time_orb' : time_orb
    }

    return DT_trajectory, runtime_DT

def plot_DT_trajectory(DT_trajectory, plot_orb_time = True, savefig = False, plot_dir = ''):
    # Trajectory data extraction
    rtn_true = DT_trajectory['rtn_true']
    rtn_dyn = DT_trajectory['rtn_dyn']
    rtn_ol = DT_trajectory['rtn_ol']
    roe_true = DT_trajectory['roe_true']
    roe_dyn = DT_trajectory['roe_dyn']
    roe_ol = DT_trajectory['roe_ol']
    dv_true = DT_trajectory['dv_true']
    dv_dyn = DT_trajectory['dv_dyn']
    dv_ol = DT_trajectory['dv_ol']
    if plot_orb_time:
        time_orb = DT_trajectory['time_orb']
    else:
        time_orb = 100*DT_trajectory['time_orb']/DT_trajectory['time_orb'][0,-1]
    i = 0
    idx_pred = 0
    idx_plt = 0
    # 3D position trajectory
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    p01 = ax1.plot3D(rtn_true[0,:], rtn_true[1,:], rtn_true[2,:], 'k-', linewidth=1.5, label='true')
    #p02 = ax1.plot3D(rtn_ol[0,:], rtn_ol[1,:], rtn_ol[2,:], 'b-', linewidth=1.5, label='pred o.l.')
    p03 = ax1.plot3D(rtn_dyn[0,:], rtn_dyn[1,:], rtn_dyn[2,:], 'g-', linewidth=1.5, label='pred dyn.')
    p1 = ax1.scatter(rtn_true[0,0], rtn_true[1,0], rtn_true[2,0], marker = 'o', linewidth=1.5, label='$t_0$')
    p2 = ax1.scatter(rtn_true[0,-1], rtn_true[1,-1], rtn_true[2,-1], marker = '*', linewidth=1.5, label='$t_f$')
    #p3 = ax1.scatter(rtn_true[0,context2.shape[1]//9], rtn_true[1,context2.shape[1]//9], rtn_true[2,context2.shape[1]//9], marker = '*', linewidth=1.5, label='$t_{init}$')
    ax1.set_xlabel('$\delta r_R$ [m]', fontsize=10)
    ax1.set_ylabel('$\delta r_T$ [m]', fontsize=10)
    ax1.set_zlabel('$\delta r_N$ [m]', fontsize=10)
    ax1.grid(True)
    ax1.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'pos_3d_{idx_plt}.png')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    p01 = ax.plot3D(rtn_true[0,:], rtn_true[1,:], rtn_true[2,:], 'k-', linewidth=1.5, label='true')
    p02 = ax.plot3D(rtn_ol[0,:], rtn_ol[1,:], rtn_ol[2,:], 'b-', linewidth=1.5, label='pred o.l.')
    #p03 = ax.plot3D(rtn_dyn[0,:], rtn_dyn[1,:], rtn_dyn[2,:], 'g-', linewidth=1.5, label='pred dyn.')
    p1 = ax.scatter(rtn_true[0,0], rtn_true[1,0], rtn_true[2,0], marker = 'o', linewidth=1.5, label='$t_0$')
    p2 = ax.scatter(rtn_true[0,-1], rtn_true[1,-1], rtn_true[2,-1], marker = '*', linewidth=1.5, label='$t_f$')
    #p3 = ax.scatter(rtn_true[0,context2.shape[1]//9], rtn_true[1,context2.shape[1]//9], rtn_true[2,context2.shape[1]//9], marker = '*', linewidth=1.5, label='$t_{init}$')
    ax.set_xlabel('$\delta r_R$ [m]', fontsize=10)
    ax.set_ylabel('$\delta r_T$ [m]', fontsize=10)
    ax.set_zlabel('$\delta r_N$ [m]', fontsize=10)
    ax.grid(True)
    ax.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'pos_3d_{idx_plt}.png')
    plt.show()

    plt.figure(figsize=(20,5))
    for j in range(3):
        plt.subplot(1,3,j+1)
        plt.plot(time_orb[0][:-1], rtn_true[j,:], 'k-', linewidth=1.5, label='true')
        plt.plot(time_orb[0][:-1], rtn_ol[j,:], 'b-', linewidth=1.5, label='pred o.l.')
        #plt.vlines(time_orb[0][(context2.shape[1]//9)+1], np.min(rtn_ol[j,:]), np.max(rtn_ol[j,:]), label='t_{init}', linewidth=2, color='red')
        plt.plot(time_orb[0][:-1], rtn_dyn[j,:], 'g-', linewidth=1.5, label='pred dyn')
        if j == 0:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$ \delta r_r$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 1:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$\delta r_t$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 2:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$\delta r_n$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'rtn_pos_{idx_plt}.png')
    plt.show()

    # velocity vs time
    plt.figure(figsize=(20,5))
    for j in range(3):
        plt.subplot(1,3,j+1)
        plt.plot(time_orb[0][:-1], rtn_true[j+3,:], 'k-', linewidth=1.5, label='true')
        plt.plot(time_orb[0][:-1], rtn_ol[j+3,:], 'b-', linewidth=1.5, label='pred o.l.')
        #plt.vlines(time_orb[0][(context2.shape[1]//9)+1], np.min(rtn_ol[j+3,:]), np.max(rtn_ol[j+3,:]), label='t_{init}', linewidth=2, color='red')
        plt.plot(time_orb[0][:-1], rtn_dyn[j+3,:], 'g-', linewidth=1.5, label='pred dyn')
        if j == 0:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$ \delta v_r$ [m/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 1:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$\delta v_t$ [m/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 2:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$\delta v_n$ [m/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'rtn_vel_{idx_plt}.png')
    plt.show()
    ###### DELTA-V

    # components
    plt.figure(figsize=(20,5))
    for j in range(3):
        plt.subplot(1,3,j+1)
        plt.stem(time_orb[0][:-1], dv_true[j,:]*1000., 'k-', label='true')
        plt.stem(time_orb[0][:-1], dv_ol[j,:]*1000., 'b-', label='pred o.l.')
        plt.stem(time_orb[0][:-1], dv_dyn[j,:]*1000., 'g-', label='pred dyn.')
        #plt.vlines(time_orb[0][(context2.shape[1]//9)+1], np.min(dv_ol[j,:]*1000.), np.max(dv_ol[j,:]*1000.), label='t_{init}', linewidth=2, color='red')
        if j == 0:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$ \Delta \delta v_r$ [mm/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 1:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$ \Delta \delta v_t$ [mm/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 2:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$ \Delta \delta v_n$ [mm/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'delta_v_{idx_plt}.png')
    plt.show()

    # norm
    plt.figure()
    plt.stem(time_orb[0][:-1], la.norm(dv_true*1000., axis=0), 'k-', label='true')
    plt.stem(time_orb[0][:-1], la.norm(dv_ol*1000., axis=0), 'b-', label='pred o.l.')
    plt.stem(time_orb[0][:-1], la.norm(dv_dyn*1000., axis=0), 'g-', label='pred dyn')
    plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
    plt.ylabel('$ || \Delta \delta v || $ [mm/s]', fontsize=10)
    plt.grid(True)
    plt.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'delta_v_norm_{idx_plt}.png')
    plt.show()

    ###### ROE STATE

    # ROE space
    plt.figure()
    p01 = plt.plot(roe_true[1, :], roe_true[0, :], 'k-', linewidth=1.5)
    p02 = plt.plot(roe_ol[1, :], roe_ol[0, :], 'b-', linewidth=1.5)
    p03 = plt.plot(roe_dyn[1, :], roe_dyn[0, :], 'g-', linewidth=1.5)
    p1 = plt.plot(roe_true[1, 0], roe_true[0, 0], 'ko', linewidth=1.5)
    p2 = plt.plot(roe_true[1, -1], roe_true[0, -1], 'k*', linewidth=1.5)
    plt.xlabel('$a \delta \lambda$ [m]', fontsize=10)
    plt.ylabel('$a \delta a$ [m]', fontsize=10)
    plt.grid(True)
    plt.legend([p01[0], p1[0], p2[0]], [
            'true', 'pred o.l.', '$t_0$', '$t_f$'], loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'roe_12_{idx_plt}.png')
    plt.show()

    plt.figure()
    p01 = plt.plot(roe_true[2, :], roe_true[3, :], 'k-', linewidth=1.5)
    p02 = plt.plot(roe_ol[2, :], roe_ol[3, :], 'b-', linewidth=1.5)
    p03 = plt.plot(roe_dyn[2, :], roe_dyn[3, :], 'g-', linewidth=1.5)
    p1 = plt.plot(roe_true[2, 0], roe_true[3, 0], 'ko', linewidth=1.5)
    p2 = plt.plot(roe_true[2, -1], roe_true[3, -1], 'k*', linewidth=1.5)
    plt.xlabel('$a \delta e_x$ [m]', fontsize=10)
    plt.ylabel('$a \delta e_y$ [m]', fontsize=10)
    plt.grid(True)
    plt.legend([p01[0], p1[0], p2[0]], [
            'true', 'pred o.l.', '$t_0$', '$t_f$'], loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'roe_34_{idx_plt}.png')
    plt.show()

    plt.figure()
    p01 = plt.plot(roe_true[4, :], roe_true[5, :], 'k-', linewidth=1.5)
    p02 = plt.plot(roe_ol[4, :], roe_ol[5, :], 'b-', linewidth=1.5)
    p03 = plt.plot(roe_dyn[4, :], roe_dyn[5, :], 'g-', linewidth=1.5)
    p1 = plt.plot(roe_true[4, 0], roe_true[5, 0], 'ko', linewidth=1.5)
    p2 = plt.plot(roe_true[4, -1], roe_true[5, -1], 'k*', linewidth=1.5)
    plt.xlabel('$a \delta i_x$ [m]', fontsize=10)
    plt.ylabel('$a \delta i_y$ [m]', fontsize=10)
    plt.grid(True)
    plt.legend([p01[0], p1[0], p2[0]], [
            'true', 'pred o.l.', '$t_0$', '$t_f$'], loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'roe_56_{idx_plt}.png')
    plt.show()

    # ROE vs time
    plt.figure(figsize=(20,5))
    for j in range(6):
        plt.subplot(2,3,j+1)
        plt.plot(time_orb[0][:-1], roe_true[j,:], 'k-', linewidth=1.5, label='true')
        plt.plot(time_orb[0][:-1], roe_ol[j,:], 'b-', linewidth=1.5, label='pred o.l.')
        plt.plot(time_orb[0][:-1], roe_dyn[j,:], 'g-', linewidth=1.5, label='pred dyn')
        if j == 0:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$a \delta a$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 1:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$a \delta \lambda$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 2:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$a \delta e_x$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 3:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$a \delta e_y$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 4:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$a \delta i_x$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 5:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$a \delta i_y$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'roe_{idx_plt}.png')
    plt.show()
