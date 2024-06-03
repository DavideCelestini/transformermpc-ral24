import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.decision_transformer.configuration_decision_transformer import DecisionTransformerConfig
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerPreTrainedModel, DecisionTransformerGPT2Model, DecisionTransformerOutput

DECISION_TRANSFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~DecisionTransformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DECISION_TRANSFORMER_INPUTS_DOCSTRING = r"""
    Args:
        states (`torch.FloatTensor` of shape `(batch_size, episode_length, state_dim)`):
            The states for each step in the trajectory
        actions (`torch.FloatTensor` of shape `(batch_size, episode_length, act_dim)`):
            The actions taken by the "expert" policy for the current state, these are masked for auto regressive
            prediction
        rewards (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):
            The rewards for each state, action
        returns_to_go (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):
            The returns for each state in the trajectory
        timesteps (`torch.LongTensor` of shape `(batch_size, episode_length)`):
            The timestep for each step in the trajectory
        attention_mask (`torch.FloatTensor` of shape `(batch_size, episode_length)`):
            Masking, used to mask the actions when performing autoregressive prediction
"""

_CONFIG_FOR_DOC = "DecisionTransformerConfig"

class AutonomousFreeflyerTransformer(DecisionTransformerPreTrainedModel):
    """

    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_goal = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_constraint = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DECISION_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=DecisionTransformerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        goal: Optional[torch.FloatTensor] = None,
        returns_to_go: Optional[torch.FloatTensor] = None,
        constraints_to_go: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DecisionTransformerOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import DecisionTransformerModel
        >>> import torch

        >>> model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
        >>> # evaluation
        >>> model = model.to(device)
        >>> model.eval()

        >>> env = gym.make("Hopper-v3")
        >>> state_dim = env.observation_space.shape[0]
        >>> act_dim = env.action_space.shape[0]

        >>> state = env.reset()
        >>> states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
        >>> actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
        >>> rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
        >>> target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
        >>> timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        >>> attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

        >>> # forward pass
        >>> with torch.no_grad():
        ...     state_preds, action_preds, return_preds = model(
        ...         states=states,
        ...         actions=actions,
        ...         rewards=rewards,
        ...         returns_to_go=target_return,
        ...         timesteps=timesteps,
        ...         attention_mask=attention_mask,
        ...         return_dict=False,
        ...     )
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        goal_embeddings = self.embed_goal(goal)
        returns_embeddings = self.embed_return(returns_to_go)
        constraints_embeddings = self.embed_constraint(constraints_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        constraints_embeddings = constraints_embeddings + time_embeddings

        # this makes the sequence look like (T_1, R_1, C_1 s_1, a_1, T_2, R_2, C_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((goal_embeddings, returns_embeddings, constraints_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 5 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 5 * seq_length)
        )
        device = stacked_inputs.device
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # goal (0), return_to_go (1), constraint_to_go (2), states (3), or actions (4); i.e. x[:,3,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        state_preds = self.predict_state(x[:, 4])  # predict next state given (T+R+C)+state and action
        action_preds = self.predict_action(x[:, 3])  # predict next action given (T+R+C)+state
        if not return_dict:
            return (state_preds, action_preds)

        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class AutonomousFreeflyerTransformer_pred_time(DecisionTransformerPreTrainedModel):
    """

    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_goal = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_constraint = torch.nn.Linear(1, config.hidden_size)
        self.embed_time = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
        self.predict_time = torch.nn.Linear(config.hidden_size, 1)
        self.predict_action = nn.Sequential(
            *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DECISION_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=DecisionTransformerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        goal: Optional[torch.FloatTensor] = None,
        returns_to_go: Optional[torch.FloatTensor] = None,
        constraints_to_go: Optional[torch.FloatTensor] = None,
        times_to_go: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DecisionTransformerOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import DecisionTransformerModel
        >>> import torch

        >>> model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
        >>> # evaluation
        >>> model = model.to(device)
        >>> model.eval()

        >>> env = gym.make("Hopper-v3")
        >>> state_dim = env.observation_space.shape[0]
        >>> act_dim = env.action_space.shape[0]

        >>> state = env.reset()
        >>> states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
        >>> actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
        >>> rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
        >>> target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
        >>> timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        >>> attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

        >>> # forward pass
        >>> with torch.no_grad():
        ...     state_preds, action_preds, return_preds = model(
        ...         states=states,
        ...         actions=actions,
        ...         rewards=rewards,
        ...         returns_to_go=target_return,
        ...         timesteps=timesteps,
        ...         attention_mask=attention_mask,
        ...         return_dict=False,
        ...     )
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        goal_embeddings = self.embed_goal(goal)
        returns_embeddings = self.embed_return(returns_to_go)
        constraints_embeddings = self.embed_constraint(constraints_to_go)
        timetogo_embeddings = self.embed_time(times_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        constraints_embeddings = constraints_embeddings + time_embeddings
        timetogo_embeddings = timetogo_embeddings + time_embeddings

        # this makes the sequence look like (T_1, R_1, C_1 s_1, t_1, a_1, T_2, R_2, C_2, s_2, t_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((goal_embeddings, returns_embeddings, constraints_embeddings, state_embeddings, timetogo_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 6 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 6 * seq_length)
        )
        device = stacked_inputs.device
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # goal (0), return_to_go (1), constraint_to_go (2), states (3), time_to_go(4) or actions (5); i.e. x[:,3,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 6, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        state_preds = self.predict_state(x[:, 5])  # predict next state given (T+R+C)+state+(t) and action
        action_preds = self.predict_action(x[:, 4])  # predict next action given (T+R+C)+state+(t)
        timetogo_preds = self.predict_time(x[:, 3])  # predict next timetogo given (T+R+C)+state
        if not return_dict:
            return (state_preds, action_preds, timetogo_preds)

        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            timetogo_preds=timetogo_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )