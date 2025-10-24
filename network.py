# Copyright (c) 2024, Go1 Sim2Real Policy Inference
# All rights reserved.

"""Neural network definitions for Go1 Sim2Real Policy Inference."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .config import Go1Config


def get_activation(activation: str) -> nn.Module:
    """Get activation function from string."""
    if activation.lower() == "elu":
        return nn.ELU()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "tanh":
        return nn.Tanh()
    elif activation.lower() == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation: {activation}")


class DepthOnlyFCBackbone(nn.Module):
    """Depth image processing CNN backbone."""
    
    def __init__(
        self, 
        output_dim: int, 
        hidden_dim: int, 
        activation: nn.Module,
        num_frames: int = 1,
        input_shape: Tuple[int, int] = (24, 32)
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.output_dim = output_dim
        self.input_shape = input_shape
        
        # Calculate output size after convolutions
        # Input: (batch, 1, 24, 32)
        # Conv1: kernel=5 -> (batch, 16, 20, 28)
        # MaxPool1: stride=2 -> (batch, 16, 10, 14)
        # Conv2: kernel=3 -> (batch, 32, 8, 12)
        # MaxPool2: stride=2 -> (batch, 32, 4, 6)
        conv_output_size = 32 * 4 * 6  # 768
        
        self.image_compression = nn.Sequential(
            # [1, 24, 32]
            nn.Conv2d(in_channels=self.num_frames, out_channels=16, kernel_size=5),
            # [16, 20, 28]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [16, 10, 14]
            activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            # [32, 8, 12]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 4, 6]
            activation,
            nn.Flatten(),
            
            nn.Linear(conv_output_size, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            activation
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through depth CNN."""
        # Add channel dimension if needed: (batch, H, W) -> (batch, 1, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(1)
        
        latent = self.image_compression(images)
        return latent


class ActorDepthCNN(nn.Module):
    """Actor network with depth CNN."""
    
    def __init__(
        self, 
        num_obs_proprio: int,
        obs_depth_shape: Tuple[int, int],
        num_actions: int,
        activation: nn.Module,
        hidden_dims: List[int] = [256, 256, 128]
    ):
        super().__init__()
        
        self.num_obs_proprio = num_obs_proprio
        self.obs_depth_shape = obs_depth_shape
        
        # Proprioceptive MLP
        self.prop_mlp = nn.Sequential(
            nn.Linear(num_obs_proprio, hidden_dims[0]),
            activation,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            activation,
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            activation,
        )
        
        # Depth CNN backbone
        self.depth_backbone = DepthOnlyFCBackbone(
            output_dim=hidden_dims[2],
            hidden_dim=hidden_dims[1],
            activation=activation,
            num_frames=1,
            input_shape=obs_depth_shape
        )
        
        # Action head
        self.action_head = nn.Linear(2 * hidden_dims[2], num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network."""
        prop_input = x[..., :self.num_obs_proprio]
        prop_latent = self.prop_mlp(prop_input)
        
        depth_input = x[..., self.num_obs_proprio:]
        ori_shape = depth_input.shape
        depth_input = depth_input.reshape(-1, *self.obs_depth_shape)
        depth_latent = self.depth_backbone(depth_input)
        
        actions = self.action_head(torch.cat((prop_latent, depth_latent), dim=-1))
        return actions
    
    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent representation."""
        original_shape = observations.shape
        
        if observations.dim() == 3:
            observations = observations.reshape(-1, original_shape[-1])
        
        prop_input = observations[..., :self.num_obs_proprio]
        prop_latent = self.prop_mlp(prop_input)
        
        depth_input = observations[..., self.num_obs_proprio:]
        depth_input = depth_input.reshape(-1, *self.obs_depth_shape)
        depth_latent = self.depth_backbone(depth_input)
        
        if len(original_shape) == 3:
            return torch.cat((prop_latent, depth_latent), dim=-1).reshape(*original_shape[:-1], -1)
        
        return torch.cat((prop_latent, depth_latent), dim=-1)


class Memory(nn.Module):
    """RNN Memory module."""
    
    def __init__(
        self, 
        input_size: int, 
        type: str = "lstm", 
        num_layers: int = 1, 
        hidden_size: int = 256
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if type.lower() == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif type.lower() == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown RNN type: {type}")
        
        self.hidden_states = None
    
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None, hidden_states: Optional[Tuple] = None) -> torch.Tensor:
        """Forward pass through RNN."""
        batch_mode = masks is not None
        
        if batch_mode:
            # Batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(x, hidden_states)
        else:
            # Inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(x.unsqueeze(0), self.hidden_states)
            out = out.squeeze(0)
        
        return out
    
    def reset(self, dones: Optional[torch.Tensor] = None):
        """Reset hidden states."""
        if self.hidden_states is not None and dones is not None:
            if isinstance(self.hidden_states, tuple):  # LSTM
                h, c = self.hidden_states
                h[..., dones, :] = 0.0
                c[..., dones, :] = 0.0
                self.hidden_states = (h, c)
            else:  # GRU
                self.hidden_states[..., dones, :] = 0.0


class ActorCriticDepthCNN(nn.Module):
    """Actor-Critic network with depth CNN."""
    
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_actor_obs_prop: int = 48,
        obs_depth_shape: Tuple[int, int] = (24, 32),
        actor_hidden_dims: List[int] = [256, 256, 128],
        critic_hidden_dims: List[int] = [256, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs
    ):
        if kwargs:
            print(f"ActorCriticDepth.__init__ got unexpected arguments, which will be ignored: {list(kwargs.keys())}")
        
        super().__init__()
        activation_fn = get_activation(activation)
        
        # Policy Function
        self.actor = ActorDepthCNN(
            num_actor_obs_prop, 
            obs_depth_shape, 
            num_actions, 
            activation_fn, 
            actor_hidden_dims
        )
        
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation_fn)
        
        self.critic = nn.Sequential(*critic_layers)
        
        print(f"Actor MLP+CNN: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # Disable args validation for speedup
        from torch.distributions import Normal
        Normal.set_default_validate_args = False
    
    def reset(self, dones: Optional[torch.Tensor] = None):
        """Reset network state."""
        pass
    
    def forward(self):
        """Not implemented - use act_inference or evaluate instead."""
        raise NotImplementedError
    
    @property
    def action_mean(self):
        """Get action mean from distribution."""
        return self.distribution.mean
    
    @property
    def action_std(self):
        """Get action std from distribution."""
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """Get entropy from distribution."""
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations: torch.Tensor):
        """Update action distribution."""
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
    
    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample action from distribution."""
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions."""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Get deterministic action for inference."""
        actions_mean = self.actor(observations)
        return actions_mean
    
    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate value function."""
        value = self.critic(critic_observations)
        return value


class ActorCriticDepthCNNRecurrent(ActorCriticDepthCNN):
    """Actor-Critic network with depth CNN and RNN."""
    
    is_recurrent = True
    
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_actor_obs_prop: int = 48,
        num_critic_obs_prop: int = 48,
        obs_depth_shape: Tuple[int, int] = (24, 32),
        actor_hidden_dims: List[int] = [256, 256, 128],
        critic_hidden_dims: List[int] = [256, 256, 128],
        activation: str = "elu",
        rnn_type: str = "lstm",
        rnn_input_size: int = 256,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 1.0,
        **kwargs
    ):
        if kwargs:
            print(f"ActorCriticDepthCNNRecurrent.__init__ got unexpected arguments, which will be ignored: {list(kwargs.keys())}")
        
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_actor_obs_prop=num_actor_obs_prop,
            num_critic_obs_prop=num_critic_obs_prop,
            obs_depth_shape=obs_depth_shape,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )
        
        self.memory_a = Memory(rnn_input_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(rnn_input_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        
        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
    
    def reset(self, dones: Optional[torch.Tensor] = None):
        """Reset RNN hidden states."""
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)
    
    def act(self, observations: torch.Tensor, masks: Optional[torch.Tensor] = None, hidden_states: Optional[Tuple] = None) -> torch.Tensor:
        """Sample action with RNN."""
        observations = self.actor.encode(observations)
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act_hidden(input_a.squeeze(0))
    
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Get deterministic action for inference with RNN."""
        observations = self.actor.encode(observations)
        input_a = self.memory_a(observations)
        return super().act_hidden_inference(input_a.squeeze(0))
    
    def evaluate(self, critic_observations: torch.Tensor, masks: Optional[torch.Tensor] = None, hidden_states: Optional[Tuple] = None) -> torch.Tensor:
        """Evaluate value function with RNN."""
        # Note: For critic, we assume same encoding as actor for simplicity
        critic_observations = self.actor.encode(critic_observations)
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate_hidden(input_c.squeeze(0))
    
    def get_hidden_states(self) -> Tuple:
        """Get hidden states from both RNNs."""
        return self.memory_a.hidden_states, self.memory_c.hidden_states
    
    def act_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Act from hidden states."""
        mean = self.actor.action_head(hidden_states)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return self.distribution.sample()
    
    def act_hidden_inference(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Act deterministically from hidden states."""
        actions_mean = self.actor.action_head(hidden_states)
        return actions_mean
    
    def evaluate_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Evaluate from hidden states."""
        return self.critic(hidden_states)


def create_network(config: Go1Config) -> nn.Module:
    """Create network based on configuration."""
    if config.use_cnn and config.use_rnn:
        return ActorCriticDepthCNNRecurrent(
            num_actor_obs=config.total_policy_input_dim,
            num_critic_obs=config.total_policy_input_dim,
            num_actions=config.num_actions,
            num_actor_obs_prop=config.num_actor_obs_prop,
            obs_depth_shape=config.depth_image_shape,
            actor_hidden_dims=config.actor_hidden_dims,
            critic_hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
            rnn_type=config.rnn_type,
            rnn_input_size=config.rnn_input_size,
            rnn_hidden_size=config.rnn_hidden_size,
            rnn_num_layers=config.rnn_num_layers,
        )
    elif config.use_cnn:
        return ActorCriticDepthCNN(
            num_actor_obs=config.total_policy_input_dim,
            num_critic_obs=config.total_policy_input_dim,
            num_actions=config.num_actions,
            num_actor_obs_prop=config.num_actor_obs_prop,
            obs_depth_shape=config.depth_image_shape,
            actor_hidden_dims=config.actor_hidden_dims,
            critic_hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
        )
    else:
        raise NotImplementedError("Non-CNN networks not implemented yet")
