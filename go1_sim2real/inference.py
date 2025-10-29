# Copyright (c) 2024, Go1 Sim2Real Policy Inference
# All rights reserved.

"""Main inference class for Go1 Sim2Real Policy Inference."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import numpy as np

# 상대 임포트 대신 절대 임포트 사용
try:
    # 패키지로 실행될 때
    from .config import Go1Config, get_default_config
    from .network import create_network
    from .observation import ObservationManager, process_depth_image
    from .utils import load_checkpoint, validate_checkpoint, get_model_info_from_checkpoint
except ImportError:
    # 직접 실행될 때
    from config import Go1Config, get_default_config
    from network import create_network
    from observation import ObservationManager, process_depth_image
    from utils import load_checkpoint, validate_checkpoint, get_model_info_from_checkpoint


class Go1PolicyInference:
    """Main class for Go1 policy inference."""
    
    def __init__(self, config: Optional[Go1Config] = None, device: str = "cpu"):
        """Initialize the policy inference system."""
        self.config = config or get_default_config()
        self.config.device = device
        
        # Initialize components
        self.network: Optional[nn.Module] = None
        self.observation_manager: Optional[ObservationManager] = None
        self.is_initialized = False
        
        # Current state
        self.current_observation: Optional[torch.Tensor] = None
        self.current_action: Optional[torch.Tensor] = None
        self.active_command: Optional[torch.Tensor] = None
        
        print(f"Go1PolicyInference initialized with device: {device}")
        print(f"Configuration: CNN={self.config.use_cnn}, RNN={self.config.use_rnn}, History={self.config.history_length}")
    
    def init(self) -> None:
        """Initialize the network and observation manager."""
        print("Initializing Go1PolicyInference...")
        
        # Create network
        self.network = create_network(self.config)
        self.network.to(self.config.device)
        self.network.eval()
        
        # Create observation manager
        self.observation_manager = ObservationManager(self.config)
        
        # Initialize active command (velocity commands)
        self.active_command = torch.zeros(3, device=self.config.device)
        
        self.is_initialized = True
        print("Go1PolicyInference initialized successfully!")
    
    def load_policy(self, policy_path: str) -> None:
        """Load policy from .pt file."""
        if not self.is_initialized:
            raise RuntimeError("Must call init() before load_policy()")
        
        print(f"Loading policy from {policy_path}...")
        
        # Load checkpoint
        checkpoint = load_checkpoint(policy_path, self.config.device)
        
        # Validate checkpoint
        if not validate_checkpoint(checkpoint):
            print("Warning: Checkpoint validation failed")
        
        # Get model info
        model_info = get_model_info_from_checkpoint(checkpoint)
        print(f"Model info: {model_info}")
        
        # Check if we need to recreate the network with correct dimensions
        needs_recreation = False
        
        # Check actor prop_mlp input size
        if "prop_mlp_input_size" in model_info:
            expected_prop_size = model_info["prop_mlp_input_size"]
            current_prop_size = self.config.num_actor_obs_prop
            if expected_prop_size != current_prop_size:
                print(f"Warning: Prop MLP input size mismatch. Expected: {expected_prop_size}, Current: {current_prop_size}")
                needs_recreation = True
        
        # Check critic input size
        if "critic_input_size" in model_info:
            expected_critic_size = model_info["critic_input_size"]
            current_critic_size = self.config.total_policy_input_dim
            if expected_critic_size != current_critic_size:
                print(f"Warning: Critic input size mismatch. Expected: {expected_critic_size}, Current: {current_critic_size}")
                needs_recreation = True
        
        # Recreate network if needed
        if needs_recreation:
            print("Recreating network with correct dimensions...")
            try:
                from .utils import create_config_from_checkpoint
            except ImportError:
                from utils import create_config_from_checkpoint
            
            # Create new config from checkpoint
            new_config = create_config_from_checkpoint(policy_path, self.config.device)
            
            # Update current config
            self.config.history_length = new_config.history_length
            self.config.num_actor_obs_prop = new_config.num_actor_obs_prop
            self.config.total_policy_input_dim = new_config.total_policy_input_dim
            
            # Recreate network
            self.network = create_network(self.config)
            self.network.to(self.config.device)
            self.network.eval()
            
            print("Network recreated successfully!")
        
        # Load model state dict
        model_state_dict = checkpoint["model_state_dict"]
        
        # Handle potential key mismatches
        network_state_dict = {}
        for key, value in model_state_dict.items():
            # Remove 'actor_critic.' prefix if present
            if key.startswith('actor_critic.'):
                new_key = key[12:]  # Remove 'actor_critic.'
                network_state_dict[new_key] = value
            else:
                network_state_dict[key] = value
        
        # Load state dict
        self.network.load_state_dict(network_state_dict)
        
        print("Policy loaded successfully!")
    
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
        """Update observation data."""
        if not self.is_initialized:
            raise RuntimeError("Must call init() before update_observation()")
        
        # Process depth image if needed
        if isinstance(depth_image, (list, tuple, np.ndarray)):
            depth_image = process_depth_image(
                depth_image,
                near_clip=self.config.depth_near_clip,
                far_clip=self.config.depth_far_clip,
                target_shape=self.config.depth_image_shape
            )
            depth_image = depth_image.squeeze(0)  # Remove batch dimension
        
        # Update observation manager
        self.observation_manager.update_observation(
            base_ang_vel, base_rpy, velocity_commands,
            joint_pos, joint_vel, actions, depth_image
        )
        
        # Get policy observation
        self.current_observation = self.observation_manager.get_policy_observation()
        
        # Update active command
        if isinstance(velocity_commands, (list, tuple)):
            velocity_commands = torch.tensor(velocity_commands, dtype=torch.float32)
        elif isinstance(velocity_commands, np.ndarray):
            velocity_commands = torch.from_numpy(velocity_commands).float()
        
        self.active_command = velocity_commands.to(self.config.device)
    
    def step(self) -> torch.Tensor:
        """Compute action using the policy."""
        if not self.is_initialized:
            raise RuntimeError("Must call init() before step()")
        
        if self.current_observation is None:
            raise RuntimeError("No observation available. Call update_observation() first.")
        
        # Move observation to device
        obs = self.current_observation.to(self.config.device)
        
        # Add batch dimension if needed
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Compute action
        with torch.no_grad():
            if self.config.use_rnn:
                action = self.network.act_inference(obs)
            else:
                action = self.network.act_inference(obs)
        
        # Remove batch dimension
        if action.dim() == 2:
            action = action.squeeze(0)
        
        # Store current action
        self.current_action = action
        
        return action
    
    def apply_action(self, action: Optional[torch.Tensor] = None) -> None:
        """Apply action to robot (currently just prints joint values)."""
        if action is None:
            if self.current_action is None:
                raise RuntimeError("No action available. Call step() first.")
            action = self.current_action
        
        # Calculate final joint positions
        final_joint_positions = []
        for i, (action_val, default_pos) in enumerate(zip(action, self.config.default_joint_positions)):
            final_pos = default_pos + action_val.item() * self.config.action_scale
            final_joint_positions.append(final_pos)
        
        # Print joint values (replace with actual robot control)
        print("=== Robot Joint Commands ===")
        print(f"Action values: {action.tolist()}")
        print(f"Final joint positions (rad): {final_joint_positions}")
        print(f"Active command: {self.active_command.tolist()}")
        
        # Print joint names for reference
        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
        ]
        
        print("\nJoint Commands:")
        for name, pos in zip(joint_names, final_joint_positions):
            print(f"  {name}: {pos:.4f} rad ({pos * 180 / 3.14159:.2f} deg)")
        
        print("=" * 30)
    
    def reset(self) -> None:
        """Reset the inference system."""
        if self.observation_manager is not None:
            self.observation_manager.reset()
        
        self.current_observation = None
        self.current_action = None
        
        if self.config.use_rnn and self.network is not None:
            self.network.reset()
        
        print("Go1PolicyInference reset")
    
    def get_observation_info(self) -> Dict[str, Any]:
        """Get information about current observation."""
        if self.observation_manager is None:
            return {"status": "not_initialized"}
        
        return self.observation_manager.get_observation_info()
    
    def get_action_info(self) -> Dict[str, Any]:
        """Get information about current action."""
        if self.current_action is None:
            return {"status": "no_action"}
        
        return {
            "status": "ready",
            "action_values": self.current_action.tolist(),
            "action_scale": self.config.action_scale,
            "default_joint_positions": self.config.default_joint_positions,
            "final_joint_positions": [
                default + action * self.config.action_scale
                for default, action in zip(self.config.default_joint_positions, self.current_action)
            ]
        }
    
    def benchmark(self, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        if not self.is_initialized:
            raise RuntimeError("Must call init() before benchmark()")
        
        if self.current_observation is None:
            raise RuntimeError("No observation available for benchmarking")
        
        try:
            from .utils import benchmark_inference
        except ImportError:
            from utils import benchmark_inference
        
        return benchmark_inference(self.network, self.current_observation, num_runs)
    
    def create_dummy_observation(self) -> None:
        """Create dummy observation for testing."""
        if not self.is_initialized:
            raise RuntimeError("Must call init() before create_dummy_observation()")
        
        # Create dummy data
        base_ang_vel = torch.zeros(3)
        base_rpy = torch.zeros(3)
        velocity_commands = torch.zeros(3)
        joint_pos = torch.zeros(12)
        joint_vel = torch.zeros(12)
        actions = torch.zeros(12)
        depth_image = torch.ones(self.config.depth_image_size) * 0.5
        
        self.update_observation(
            base_ang_vel, base_rpy, velocity_commands,
            joint_pos, joint_vel, actions, depth_image
        )
        
        print("Dummy observation created")


def create_inference_system(
    policy_path: str,
    config: Optional[Go1Config] = None,
    device: str = "cpu"
) -> Go1PolicyInference:
    """Create and initialize inference system from policy file."""
    
    # Create config from checkpoint if not provided
    if config is None:
        try:
            from .utils import create_config_from_checkpoint
        except ImportError:
            from utils import create_config_from_checkpoint
        config = create_config_from_checkpoint(policy_path, device)
    
    # Create inference system
    inference = Go1PolicyInference(config, device)
    
    # Initialize
    inference.init()
    
    # Load policy
    inference.load_policy(policy_path)
    
    return inference
