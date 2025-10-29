# Copyright (c) 2024, Go1 Sim2Real Policy Inference
# All rights reserved.

"""Utility functions for Go1 Sim2Real Policy Inference."""

import torch
import os
from typing import Dict, Any, Optional, Union
import json


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """Load checkpoint from .pt file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def validate_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    """Validate checkpoint structure."""
    required_keys = ["model_state_dict", "optimizer_state_dict"]
    
    for key in required_keys:
        if key not in checkpoint:
            print(f"Warning: Missing key '{key}' in checkpoint")
            return False
    
    return True


def get_model_info_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model information from checkpoint."""
    info = {}
    
    # Get model state dict
    model_state = checkpoint.get("model_state_dict", {})
    
    # Extract network architecture info
    if "actor.prop_mlp.0.weight" in model_state:
        info["use_cnn"] = True
        info["has_prop_mlp"] = True
        
        # Get proprio MLP input size
        prop_mlp_input_size = model_state["actor.prop_mlp.0.weight"].shape[1]
        info["prop_mlp_input_size"] = prop_mlp_input_size
        
        # Calculate history length and proprioceptive observation dimension
        # prop_mlp_input_size = num_proprio_obs * (history_length + 1)
        
        # Try to find a reasonable history_length that gives a valid proprioceptive dimension
        # Common history lengths used in legged-loco
        possible_history_lengths = [8, 9, 10, 7, 6, 5, 4, 3, 2, 1]
        
        found_valid_combination = False
        
        for history_length in possible_history_lengths:
            if prop_mlp_input_size % (history_length + 1) == 0:
                proprio_dim = prop_mlp_input_size // (history_length + 1)
                # Check if this gives a reasonable proprioceptive dimension (typically 30-60 for legged robots)
                if 20 <= proprio_dim <= 80:
                    info["num_proprio_obs"] = proprio_dim
                    info["history_length"] = history_length
                    print(f"Detected from checkpoint: proprio_dim={proprio_dim}, history_length={history_length}")
                    found_valid_combination = True
                    break
        
        # If no reasonable combination found, try all possible divisors
        if not found_valid_combination:
            for history_length in range(1, 20):  # Try history lengths from 1 to 19
                if prop_mlp_input_size % (history_length + 1) == 0:
                    proprio_dim = prop_mlp_input_size // (history_length + 1)
                    if proprio_dim > 0:
                        info["num_proprio_obs"] = proprio_dim
                        info["history_length"] = history_length
                        print(f"Fallback detection: proprio_dim={proprio_dim}, history_length={history_length}")
                        found_valid_combination = True
                        break
        
        if not found_valid_combination:
            print(f"Warning: Could not determine proprioceptive observation dimension from prop_mlp_input_size={prop_mlp_input_size}")
            # Fallback to original calculation
            num_proprio_obs = 48
            history_length = (prop_mlp_input_size // num_proprio_obs) - 1
            info["num_proprio_obs"] = num_proprio_obs
            info["history_length"] = history_length
    
    if "actor.depth_backbone.image_compression.0.weight" in model_state:
        info["has_depth_backbone"] = True
        
        # Get depth backbone input channels
        depth_input_channels = model_state["actor.depth_backbone.image_compression.0.weight"].shape[0]
        info["depth_input_channels"] = depth_input_channels
    
    if "memory_a.rnn.weight_ih_l0" in model_state:
        info["use_rnn"] = True
        info["rnn_type"] = "lstm"  # or gru, need to check
    
    # Get action dimension
    if "actor.action_head.weight" in model_state:
        num_actions = model_state["actor.action_head.weight"].shape[0]
        info["num_actions"] = num_actions
    
    # Get critic info
    if "critic.0.weight" in model_state:
        critic_input_size = model_state["critic.0.weight"].shape[1]
        info["critic_input_size"] = critic_input_size
    
    return info


def create_config_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> 'Go1Config':
    """Create configuration from checkpoint."""
    from .config import Go1Config
    
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_info = get_model_info_from_checkpoint(checkpoint)
    
    # Create config with detected parameters
    config = Go1Config()
    config.device = device
    
    if "use_cnn" in model_info:
        config.use_cnn = model_info["use_cnn"]
    
    if "use_rnn" in model_info:
        config.use_rnn = model_info["use_rnn"]
    
    if "history_length" in model_info:
        config.history_length = model_info["history_length"]
    
    if "num_actions" in model_info:
        config.num_actions = model_info["num_actions"]
    
    # Update proprioceptive observation dimension if detected
    if "num_proprio_obs" in model_info:
        config.num_proprio_obs = model_info["num_proprio_obs"]
        print(f"Using detected proprioceptive observation dimension: {config.num_proprio_obs}")
    
    # Recalculate derived parameters
    config.__post_init__()
    
    print(f"Created config from checkpoint:")
    print(f"  History length: {config.history_length}")
    print(f"  Num proprio obs: {config.num_proprio_obs}")
    print(f"  Num actor obs prop: {config.num_actor_obs_prop}")
    print(f"  Total policy input dim: {config.total_policy_input_dim}")
    print(f"  Use CNN: {config.use_cnn}")
    print(f"  Use RNN: {config.use_rnn}")
    
    return config


def save_config(config: 'Go1Config', save_path: str) -> None:
    """Save configuration to JSON file."""
    config_dict = {
        "use_cnn": config.use_cnn,
        "use_rnn": config.use_rnn,
        "history_length": config.history_length,
        "depth_image_shape": config.depth_image_shape,
        "num_joints": config.num_joints,
        "num_proprio_obs": config.num_proprio_obs,
        "num_policy_obs": config.num_policy_obs,
        "actor_hidden_dims": config.actor_hidden_dims,
        "critic_hidden_dims": config.critic_hidden_dims,
        "activation": config.activation,
        "rnn_type": config.rnn_type,
        "rnn_input_size": config.rnn_input_size,
        "rnn_hidden_size": config.rnn_hidden_size,
        "rnn_num_layers": config.rnn_num_layers,
        "action_scale": config.action_scale,
        "num_actions": config.num_actions,
        "default_joint_positions": config.default_joint_positions,
        "depth_near_clip": config.depth_near_clip,
        "depth_far_clip": config.depth_far_clip,
        "device": config.device,
        "depth_image_size": config.depth_image_size,
        "num_actor_obs_prop": config.num_actor_obs_prop,
        "total_policy_input_dim": config.total_policy_input_dim
    }
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {save_path}")


def load_config(config_path: str) -> 'Go1Config':
    """Load configuration from JSON file."""
    from .config import Go1Config
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = Go1Config()
    
    # Update config with loaded values
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Recalculate derived parameters
    config.__post_init__()
    
    print(f"Configuration loaded from {config_path}")
    return config


def validate_observation(obs: torch.Tensor, expected_size: int) -> bool:
    """Validate observation tensor."""
    if obs.shape[-1] != expected_size:
        print(f"Warning: Expected observation size {expected_size}, got {obs.shape[-1]}")
        return False
    
    if torch.isnan(obs).any():
        print("Warning: Observation contains NaN values")
        return False
    
    if torch.isinf(obs).any():
        print("Warning: Observation contains Inf values")
        return False
    
    return True


def print_observation_info(obs: torch.Tensor, config: 'Go1Config') -> None:
    """Print detailed observation information."""
    print(f"Observation shape: {obs.shape}")
    print(f"Expected size: {config.total_policy_input_dim}")
    
    if config.use_cnn:
        print(f"Policy obs size: {config.num_policy_obs}")
        print(f"History size: {config.num_actor_obs_prop}")
        print(f"Depth image size: {config.depth_image_size}")
        print(f"Proprio obs size: {config.num_proprio_obs}")
    
    print(f"History length: {config.history_length}")
    print(f"Use CNN: {config.use_cnn}")
    print(f"Use RNN: {config.use_rnn}")


def print_action_info(actions: torch.Tensor, config: 'Go1Config') -> None:
    """Print detailed action information."""
    print(f"Action shape: {actions.shape}")
    print(f"Expected size: {config.num_actions}")
    print(f"Action scale: {config.action_scale}")
    print(f"Default joint positions: {config.default_joint_positions}")
    
    # Calculate final joint positions
    final_positions = []
    for i, (action, default_pos) in enumerate(zip(actions, config.default_joint_positions)):
        final_pos = default_pos + action * config.action_scale
        final_positions.append(final_pos.item())
    
    print(f"Final joint positions: {final_positions}")


def create_dummy_data(config: 'Go1Config') -> Dict[str, torch.Tensor]:
    """Create dummy data for testing."""
    from .observation import create_dummy_observation
    
    # Create dummy observation
    obs = create_dummy_observation(config)
    
    # Create dummy action
    actions = torch.zeros(config.num_actions)
    
    return {
        "observation": obs,
        "actions": actions
    }


def benchmark_inference(model: torch.nn.Module, obs: torch.Tensor, num_runs: int = 100) -> Dict[str, float]:
    """Benchmark model inference performance."""
    import time
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model.act_inference(obs)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.act_inference(obs)
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "average_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "fps": 1.0 / avg_time
    }
