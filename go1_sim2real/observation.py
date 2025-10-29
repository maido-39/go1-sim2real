# Copyright (c) 2024, Go1 Sim2Real Policy Inference
# All rights reserved.

"""Observation processing and management for Go1 Sim2Real Policy Inference."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from .config import Go1Config, ObservationConfig


@dataclass
class ObservationData:
    """Structured observation data."""
    
    # Base state
    base_ang_vel: torch.Tensor  # [3] - angular velocity
    base_rpy: torch.Tensor      # [3] - roll, pitch, yaw
    velocity_commands: torch.Tensor  # [3] - linear_x, linear_y, angular_z
    
    # Joint state
    joint_pos: torch.Tensor     # [12] - joint positions (relative to default)
    joint_vel: torch.Tensor     # [12] - joint velocities
    
    # Previous action
    actions: torch.Tensor       # [12] - previous action
    
    # Depth image
    depth_image: torch.Tensor   # [768] - flattened depth image (24x32)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to concatenated tensor."""
        return torch.cat([
            self.base_ang_vel,
            self.base_rpy,
            self.velocity_commands,
            self.joint_pos,
            self.joint_vel,
            self.actions,
            self.depth_image
        ], dim=-1)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, obs_config: ObservationConfig) -> 'ObservationData':
        """Create from concatenated tensor."""
        idx = 0
        
        base_ang_vel = tensor[..., idx:idx+obs_config.base_ang_vel_dim]
        idx += obs_config.base_ang_vel_dim
        
        base_rpy = tensor[..., idx:idx+obs_config.base_rpy_dim]
        idx += obs_config.base_rpy_dim
        
        velocity_commands = tensor[..., idx:idx+obs_config.velocity_commands_dim]
        idx += obs_config.velocity_commands_dim
        
        joint_pos = tensor[..., idx:idx+obs_config.joint_pos_dim]
        idx += obs_config.joint_pos_dim
        
        joint_vel = tensor[..., idx:idx+obs_config.joint_vel_dim]
        idx += obs_config.joint_vel_dim
        
        actions = tensor[..., idx:idx+obs_config.actions_dim]
        idx += obs_config.actions_dim
        
        depth_image = tensor[..., idx:idx+obs_config.depth_image_dim]
        
        return cls(
            base_ang_vel=base_ang_vel,
            base_rpy=base_rpy,
            velocity_commands=velocity_commands,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            actions=actions,
            depth_image=depth_image
        )


class ObservationBuffer:
    """Buffer for managing observations."""
    
    def __init__(self, config: Go1Config):
        self.config = config
        self.obs_config = ObservationConfig()
        
        # Current observation
        self.current_obs: Optional[ObservationData] = None
        
        # Observation validation
        self.expected_dims = {
            'base_ang_vel': 3,
            'base_rpy': 3,
            'velocity_commands': 3,
            'joint_pos': 12,
            'joint_vel': 12,
            'actions': 12,
            'depth_image': self.config.depth_image_size
        }
    
    def update_observation(
        self,
        base_ang_vel: Union[torch.Tensor, np.ndarray, List[float]],
        base_rpy: Union[torch.Tensor, np.ndarray, List[float]],
        velocity_commands: Union[torch.Tensor, np.ndarray, List[float]],
        joint_pos: Union[torch.Tensor, np.ndarray, List[float]],
        joint_vel: Union[torch.Tensor, np.ndarray, List[float]],
        actions: Union[torch.Tensor, np.ndarray, List[float]],
        depth_image: Union[torch.Tensor, np.ndarray, List[float]]
    ) -> None:
        """Update current observation."""
        
        # Convert to tensors and validate
        base_ang_vel = self._to_tensor(base_ang_vel, 'base_ang_vel')
        base_rpy = self._to_tensor(base_rpy, 'base_rpy')
        velocity_commands = self._to_tensor(velocity_commands, 'velocity_commands')
        joint_pos = self._to_tensor(joint_pos, 'joint_pos')
        joint_vel = self._to_tensor(joint_vel, 'joint_vel')
        actions = self._to_tensor(actions, 'actions')
        depth_image = self._to_tensor(depth_image, 'depth_image')
        
        # Create observation data
        self.current_obs = ObservationData(
            base_ang_vel=base_ang_vel,
            base_rpy=base_rpy,
            velocity_commands=velocity_commands,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            actions=actions,
            depth_image=depth_image
        )
    
    def _to_tensor(self, data: Union[torch.Tensor, np.ndarray, List[float]], name: str) -> torch.Tensor:
        """Convert data to tensor and validate dimensions."""
        if isinstance(data, (list, tuple)):
            data = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            data = data.float()
        else:
            raise TypeError(f"Unsupported data type for {name}: {type(data)}")
        
        # Validate dimensions
        expected_dim = self.expected_dims[name]
        if data.shape[-1] != expected_dim:
            raise ValueError(f"Expected {name} to have {expected_dim} dimensions, got {data.shape[-1]}")
        
        return data
    
    def get_current_observation(self) -> torch.Tensor:
        """Get current observation as tensor."""
        if self.current_obs is None:
            raise RuntimeError("No observation available. Call update_observation() first.")
        
        return self.current_obs.to_tensor()
    
    def get_proprio_observation(self) -> torch.Tensor:
        """Get proprioceptive observation (without depth image)."""
        if self.current_obs is None:
            raise RuntimeError("No observation available. Call update_observation() first.")
        
        return torch.cat([
            self.current_obs.base_ang_vel,
            self.current_obs.base_rpy,
            self.current_obs.velocity_commands,
            self.current_obs.joint_pos,
            self.current_obs.joint_vel,
            self.current_obs.actions
        ], dim=-1)
    
    def get_depth_image(self) -> torch.Tensor:
        """Get depth image."""
        if self.current_obs is None:
            raise RuntimeError("No observation available. Call update_observation() first.")
        
        return self.current_obs.depth_image
    
    def validate_observation(self, obs: torch.Tensor) -> bool:
        """Validate observation tensor."""
        expected_size = self.config.num_policy_obs
        if obs.shape[-1] != expected_size:
            print(f"Warning: Expected observation size {expected_size}, got {obs.shape[-1]}")
            return False
        return True


class ProprioHistoryBuffer:
    """Buffer for managing proprioceptive observation history."""
    
    def __init__(self, config: Go1Config):
        self.config = config
        self.history_length = config.history_length
        self.proprio_dim = config.num_proprio_obs
        
        # History buffer: [history_length, proprio_dim]
        self.history_buffer: Optional[torch.Tensor] = None
        self.is_initialized = False
    
    def update_history(self, proprio_obs: torch.Tensor) -> None:
        """Update history buffer with new proprioceptive observation."""
        if proprio_obs.shape[-1] != self.proprio_dim:
            raise ValueError(f"Expected proprio obs dim {self.proprio_dim}, got {proprio_obs.shape[-1]}")
        
        if not self.is_initialized:
            # Initialize buffer with current observation repeated
            self.history_buffer = proprio_obs.unsqueeze(0).repeat(self.history_length, 1)
            self.is_initialized = True
        else:
            # Shift buffer and add new observation
            self.history_buffer = torch.cat([
                self.history_buffer[1:],  # Remove oldest
                proprio_obs.unsqueeze(0)  # Add newest
            ], dim=0)
    
    def get_history(self) -> torch.Tensor:
        """Get flattened history buffer."""
        if not self.is_initialized:
            raise RuntimeError("History buffer not initialized. Call update_history() first.")
        
        return self.history_buffer.flatten()
    
    def reset(self) -> None:
        """Reset history buffer."""
        self.history_buffer = None
        self.is_initialized = False


class ObservationManager:
    """Main observation manager."""
    
    def __init__(self, config: Go1Config):
        self.config = config
        self.obs_buffer = ObservationBuffer(config)
        self.history_buffer = ProprioHistoryBuffer(config)
    
    def update_observation(
        self,
        base_ang_vel: Union[torch.Tensor, np.ndarray, List[float]],
        base_rpy: Union[torch.Tensor, np.ndarray, List[float]],
        velocity_commands: Union[torch.Tensor, np.ndarray, List[float]],
        joint_pos: Union[torch.Tensor, np.ndarray, List[float]],
        joint_vel: Union[torch.Tensor, np.ndarray, List[float]],
        actions: Union[torch.Tensor, np.ndarray, List[float]],
        depth_image: Union[torch.Tensor, np.ndarray, List[float]]
    ) -> None:
        """Update observation and history."""
        # Update current observation
        self.obs_buffer.update_observation(
            base_ang_vel, base_rpy, velocity_commands,
            joint_pos, joint_vel, actions, depth_image
        )
        
        # Update history buffer
        proprio_obs = self.obs_buffer.get_proprio_observation()
        self.history_buffer.update_history(proprio_obs)
    
    def get_policy_observation(self) -> torch.Tensor:
        """Get observation for policy inference."""
        # Get current observation
        current_obs = self.obs_buffer.get_current_observation()
        
        # Get history
        history_obs = self.history_buffer.get_history()
        
        # Concatenate
        policy_obs = torch.cat([current_obs, history_obs], dim=-1)
        
        # Validate
        expected_size = self.config.total_policy_input_dim
        if policy_obs.shape[-1] != expected_size:
            raise ValueError(f"Expected policy obs size {expected_size}, got {policy_obs.shape[-1]}")
        
        return policy_obs
    
    def reset(self) -> None:
        """Reset observation manager."""
        self.obs_buffer.current_obs = None
        self.history_buffer.reset()
    
    def get_observation_info(self) -> Dict[str, any]:
        """Get information about current observation."""
        if self.obs_buffer.current_obs is None:
            return {"status": "no_observation"}
        
        return {
            "status": "ready",
            "current_obs_size": self.obs_buffer.get_current_observation().shape[-1],
            "history_size": self.history_buffer.get_history().shape[-1] if self.history_buffer.is_initialized else 0,
            "policy_obs_size": self.config.total_policy_input_dim,
            "joint_names": self.obs_buffer.obs_config.joint_names
        }


def process_depth_image(
    depth_data: Union[torch.Tensor, np.ndarray, List[float]],
    near_clip: float = 0.3,
    far_clip: float = 2.0,
    target_shape: Tuple[int, int] = (24, 32)
) -> torch.Tensor:
    """Process depth image data similar to Isaac Lab."""
    
    # Convert to tensor
    if isinstance(depth_data, (list, tuple)):
        depth_data = torch.tensor(depth_data, dtype=torch.float32)
    elif isinstance(depth_data, np.ndarray):
        depth_data = torch.from_numpy(depth_data).float()
    elif isinstance(depth_data, torch.Tensor):
        depth_data = depth_data.float()
    
    # Reshape if needed
    if depth_data.shape != target_shape:
        if depth_data.numel() == target_shape[0] * target_shape[1]:
            depth_data = depth_data.reshape(target_shape)
        else:
            raise ValueError(f"Cannot reshape depth data from {depth_data.shape} to {target_shape}")
    
    # Add batch dimension
    if depth_data.dim() == 2:
        depth_data = depth_data.unsqueeze(0)
    
    # Process similar to Isaac Lab
    output = depth_data.clone()
    
    # Handle NaN and Inf values
    output[torch.isnan(output)] = far_clip
    output[torch.isinf(output)] = far_clip
    
    # Clip values
    output = torch.clamp(output, near_clip, far_clip)
    
    # Subtract near clip
    output = output - near_clip
    
    # Flatten
    output = output.reshape(output.shape[0], -1)
    
    return output


def create_dummy_observation(config: Go1Config) -> torch.Tensor:
    """Create dummy observation for testing."""
    obs_manager = ObservationManager(config)
    
    # Create dummy data
    base_ang_vel = torch.zeros(3)
    base_rpy = torch.zeros(3)
    velocity_commands = torch.zeros(3)
    joint_pos = torch.zeros(12)
    joint_vel = torch.zeros(12)
    actions = torch.zeros(12)
    depth_image = torch.ones(config.depth_image_size) * 0.5  # Mid-range depth
    
    # Update observation
    obs_manager.update_observation(
        base_ang_vel, base_rpy, velocity_commands,
        joint_pos, joint_vel, actions, depth_image
    )
    
    return obs_manager.get_policy_observation()
