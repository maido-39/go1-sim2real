# Copyright (c) 2024, Go1 Sim2Real Policy Inference
# All rights reserved.

"""Configuration parameters for Go1 Sim2Real Policy Inference."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch


@dataclass
class Go1Config:
    """Configuration for Go1 Sim2Real Policy Inference."""
    
    # Network architecture parameters
    use_cnn: bool = True
    use_rnn: bool = False
    history_length: int = 9
    
    # Observation dimensions
    depth_image_shape: Tuple[int, int] = (24, 32)  # height, width
    num_joints: int = 12
    num_proprio_obs: int = 48  # base_ang_vel(3) + base_rpy(3) + velocity_commands(3) + joint_pos(12) + joint_vel(12) + actions(12) + depth_image(768) - depth_image(768) = 48
    num_policy_obs: int = 816  # proprio(48) + depth_image(768)
    
    # Network architecture
    actor_hidden_dims: List[int] = None
    critic_hidden_dims: List[int] = None
    activation: str = "elu"
    
    # RNN parameters (if use_rnn=True)
    rnn_type: str = "lstm"
    rnn_input_size: int = 256  # 2 * actor_hidden_dims[-1]
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1
    
    # Action parameters
    action_scale: float = 0.25
    num_actions: int = 12
    
    # Default joint positions (radians) - Go1 default stance
    default_joint_positions: List[float] = None
    
    # Depth image processing parameters
    depth_near_clip: float = 0.3
    depth_far_clip: float = 2.0
    
    # Device
    device: str = "cpu"
    
    def __post_init__(self):
        """Post-initialization to set default values."""
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [512, 256, 128]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [512, 256, 128]
        if self.default_joint_positions is None:
            # Go1 default joint positions (radians)
            self.default_joint_positions = [
                0.1,   # FL_hip_joint
                0.8,   # FL_thigh_joint
                -1.5,  # FL_calf_joint
                -0.1,  # FR_hip_joint
                0.8,   # FR_thigh_joint
                -1.5,  # FR_calf_joint
                0.1,   # RL_hip_joint
                1.0,   # RL_thigh_joint
                -1.5,  # RL_calf_joint
                -0.1,  # RR_hip_joint
                1.0,   # RR_thigh_joint
                -1.5   # RR_calf_joint
            ]
        
        # Calculate derived parameters
        self.depth_image_size = self.depth_image_shape[0] * self.depth_image_shape[1]  # 768
        self.num_actor_obs_prop = self.num_proprio_obs * (self.history_length + 1)  # 48 * 10 = 480
        
        # Total policy input dimension
        if self.use_cnn:
            self.total_policy_input_dim = self.num_policy_obs + self.num_actor_obs_prop  # 816 + 480 = 1296
        else:
            self.total_policy_input_dim = self.num_proprio_obs * (self.history_length + 1)  # 48 * 10 = 480
        
        # RNN parameters
        if self.use_rnn:
            self.rnn_input_size = 2 * self.actor_hidden_dims[-1]  # 2 * 128 = 256
            self.rnn_hidden_size = 2 * self.actor_hidden_dims[-1]  # 2 * 128 = 256


@dataclass
class ObservationConfig:
    """Configuration for observation processing."""
    
    # Observation component dimensions
    base_ang_vel_dim: int = 3
    base_rpy_dim: int = 3
    velocity_commands_dim: int = 3
    joint_pos_dim: int = 12
    joint_vel_dim: int = 12
    actions_dim: int = 12
    depth_image_dim: int = 768  # 24 * 32
    
    # Noise parameters (for training, not used in inference)
    base_ang_vel_noise: Tuple[float, float] = (-0.2, 0.2)
    base_rpy_noise: Tuple[float, float] = (-0.1, 0.1)
    joint_pos_noise: Tuple[float, float] = (-0.01, 0.01)
    joint_vel_noise: Tuple[float, float] = (-1.5, 1.5)
    depth_image_noise: Tuple[float, float] = (-0.1, 0.1)
    
    # Joint names (for reference)
    joint_names: List[str] = None
    
    def __post_init__(self):
        if self.joint_names is None:
            self.joint_names = [
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", 
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
            ]


def get_default_config() -> Go1Config:
    """Get default configuration for Go1 Sim2Real."""
    return Go1Config()


def get_config_from_cli_args(
    use_cnn: bool = True,
    use_rnn: bool = False,
    history_length: int = 9,
    depth_shape: Tuple[int, int] = (24, 32),
    device: str = "cpu"
) -> Go1Config:
    """Create configuration from CLI-like arguments."""
    config = Go1Config()
    config.use_cnn = use_cnn
    config.use_rnn = use_rnn
    config.history_length = history_length
    config.depth_image_shape = depth_shape
    config.device = device
    config.__post_init__()  # Recalculate derived parameters
    return config
