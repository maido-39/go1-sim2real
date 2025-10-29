# Copyright (c) 2024, Go1 Sim2Real Policy Inference
# All rights reserved.

"""Example usage of Go1 Sim2Real Policy Inference."""

import torch
import numpy as np
from typing import List

# 상대 임포트 대신 절대 임포트 사용
try:
    # 패키지로 실행될 때
    from .inference import Go1PolicyInference, create_inference_system
    from .config import Go1Config, get_config_from_cli_args
    from .utils import print_observation_info, print_action_info
except ImportError:
    # 직접 실행될 때
    from inference import Go1PolicyInference, create_inference_system
    from config import Go1Config, get_config_from_cli_args
    from utils import print_observation_info, print_action_info


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Create inference system
    inference = Go1PolicyInference(device="cpu")
    
    # Initialize
    inference.init()
    
    # Load policy (replace with actual path)
    policy_path = "./model_14999.pt"
    try:
        inference.load_policy(policy_path)
    except FileNotFoundError:
        print(f"Policy file not found: {policy_path}")
        print("Creating dummy observation for demonstration...")
        inference.create_dummy_observation()
        return
    
    # Create dummy observation
    inference.create_dummy_observation()
    
    # Compute action
    action = inference.step()
    
    # Apply action
    inference.apply_action()
    
    print("Basic usage example completed!")


def example_with_real_data():
    """Example with realistic data."""
    print("\n=== Realistic Data Example ===")
    
    # Create config
    config = get_config_from_cli_args(
        use_cnn=True,
        use_rnn=False,
        history_length=9,
        device="cpu"
    )
    
    # Create inference system
    inference = Go1PolicyInference(config)
    inference.init()
    
    # Simulate realistic observation data
    base_ang_vel = torch.tensor([0.1, -0.05, 0.02])  # Small angular velocities
    base_rpy = torch.tensor([0.05, 0.1, 0.0])        # Small roll/pitch, no yaw
    velocity_commands = torch.tensor([0.5, 0.0, 0.0])  # Forward motion
    
    # Joint positions (relative to default)
    joint_pos = torch.tensor([
        0.0, 0.0, 0.0,  # FL leg
        0.0, 0.0, 0.0,  # FR leg
        0.0, 0.0, 0.0,  # RL leg
        0.0, 0.0, 0.0   # RR leg
    ])
    
    # Joint velocities
    joint_vel = torch.tensor([
        0.1, 0.0, 0.0,  # FL leg
        0.1, 0.0, 0.0,  # FR leg
        0.1, 0.0, 0.0,  # RL leg
        0.1, 0.0, 0.0   # RR leg
    ])
    
    # Previous actions
    actions = torch.zeros(12)
    
    # Depth image (simulate depth data)
    depth_image = np.random.uniform(0.0, 1.0, (24, 32)).flatten()
    
    # Update observation
    inference.update_observation(
        base_ang_vel, base_rpy, velocity_commands,
        joint_pos, joint_vel, actions, depth_image
    )
    
    # Compute action
    action = inference.step()
    
    # Apply action
    inference.apply_action()
    
    print("Realistic data example completed!")


def example_benchmark():
    """Benchmark example."""
    print("\n=== Benchmark Example ===")
    
    # Create inference system
    inference = Go1PolicyInference(device="cpu")
    inference.init()
    
    # Create dummy observation
    inference.create_dummy_observation()
    
    # Benchmark
    try:
        benchmark_results = inference.benchmark(num_runs=100)
        
        print("Benchmark Results:")
        print(f"  Average time: {benchmark_results['average_time_ms']:.2f} ms")
        print(f"  Min time: {benchmark_results['min_time_ms']:.2f} ms")
        print(f"  Max time: {benchmark_results['max_time_ms']:.2f} ms")
        print(f"  FPS: {benchmark_results['fps']:.1f}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")


def example_observation_info():
    """Example showing observation information."""
    print("\n=== Observation Info Example ===")
    
    # Create inference system
    inference = Go1PolicyInference(device="cpu")
    inference.init()
    
    # Create dummy observation
    inference.create_dummy_observation()
    
    # Get observation info
    obs_info = inference.get_observation_info()
    print("Observation Info:")
    for key, value in obs_info.items():
        print(f"  {key}: {value}")
    
    # Get action info
    action = inference.step()
    action_info = inference.get_action_info()
    print("\nAction Info:")
    for key, value in action_info.items():
        print(f"  {key}: {value}")


def example_custom_config():
    """Example with custom configuration."""
    print("\n=== Custom Config Example ===")
    
    # Create custom config
    config = Go1Config()
    config.use_cnn = True
    config.use_rnn = False
    config.history_length = 5  # Different history length
    config.actor_hidden_dims = [256, 128, 64]  # Different architecture
    config.device = "cpu"
    config.__post_init__()  # Recalculate derived parameters
    
    print(f"Custom config: {config}")
    
    # Create inference system
    inference = Go1PolicyInference(config)
    inference.init()
    
    # Create dummy observation
    inference.create_dummy_observation()
    
    # Compute action
    action = inference.step()
    
    print("Custom config example completed!")


def example_step_by_step():
    """Step-by-step example showing the full pipeline."""
    print("\n=== Step-by-Step Example ===")
    
    # Step 1: Create inference system
    print("Step 1: Creating inference system...")
    inference = Go1PolicyInference(device="cpu")
    
    # Step 2: Initialize
    print("Step 2: Initializing...")
    inference.init()
    
    # Step 3: Create dummy observation
    print("Step 3: Creating dummy observation...")
    inference.create_dummy_observation()
    
    # Step 4: Compute action
    print("Step 4: Computing action...")
    action = inference.step()
    
    # Step 5: Apply action
    print("Step 5: Applying action...")
    inference.apply_action()
    
    # Step 6: Show info
    print("Step 6: Showing information...")
    obs_info = inference.get_observation_info()
    action_info = inference.get_action_info()
    
    print(f"Observation status: {obs_info['status']}")
    print(f"Action status: {action_info['status']}")
    
    print("Step-by-step example completed!")


def main():
    """Run all examples."""
    print("Go1 Sim2Real Policy Inference - Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_with_real_data()
        example_benchmark()
        example_observation_info()
        example_custom_config()
        example_step_by_step()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
