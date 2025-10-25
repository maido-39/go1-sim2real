# Go1 Sim2Real Policy Inference

A PyTorch-only implementation for running Go1 robot policies trained with Isaac Lab in real-world scenarios.

## Overview

This repository provides a complete Sim2Real inference system for Go1 quadruped robot policies trained with the `--use_cnn` flag. It removes all Isaac Lab dependencies and provides a clean PyTorch-only interface for real robot control.

## Features

- ✅ **PyTorch-only**: No Isaac Lab dependencies
- ✅ **CNN Support**: Depth image processing with CNN backbone
- ✅ **RNN Support**: Optional recurrent neural networks
- ✅ **History Buffer**: Configurable observation history
- ✅ **Real-time Inference**: Optimized for real robot control
- ✅ **Configurable**: Flexible configuration system
- ✅ **Easy to Use**: Simple API for integration

## Installation

```bash
pip install torch torchvision
git clone https://github.com/maido-39/go1-sim2real.git
cd go1-sim2real
```

## Quick Start

```python
from go1_sim2real import Go1PolicyInference

# Create inference system
inference = Go1PolicyInference(device="cpu")
inference.init()

# Load your trained policy
inference.load_policy("path/to/policy.pt")

# Update observation with real robot data
inference.update_observation(
    base_ang_vel=[0.1, -0.05, 0.02],
    base_rpy=[0.05, 0.1, 0.0],
    velocity_commands=[0.5, 0.0, 0.0],
    joint_pos=[0.0] * 12,  # Relative to default positions
    joint_vel=[0.1] * 12,
    actions=[0.0] * 12,   # Previous actions
    depth_image=depth_data  # 24x32 depth image
)

# Compute action
action = inference.step()

# Apply to robot (currently prints joint values)
inference.apply_action()
```

## Configuration

The system supports various configurations:

```python
from go1_sim2real import Go1Config

config = Go1Config()
config.use_cnn = True
config.use_rnn = False
config.history_length = 9
config.depth_image_shape = (24, 32)
config.device = "cpu"

inference = Go1PolicyInference(config)
```

## Observation Structure

The policy expects the following observation structure:

### Policy Observation (816 dimensions)
- `base_ang_vel`: 3D angular velocity
- `base_rpy`: 3D roll, pitch, yaw
- `velocity_commands`: 3D linear_x, linear_y, angular_z
- `joint_pos`: 12D joint positions (relative to default)
- `joint_vel`: 12D joint velocities
- `actions`: 12D previous actions
- `depth_image`: 768D depth image (24×32 flattened)

### History Buffer (432 dimensions)
- Proprioceptive observations × 9 timesteps
- Total policy input: 816 + 432 = 1248 dimensions

## Action Structure

The policy outputs 12D actions representing joint position offsets:

```python
final_joint_pos = default_joint_pos + action * 0.25
```

### Default Joint Positions (radians)
```python
default_joint_positions = [
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
```

## Depth Image Processing

The system processes depth images similar to Isaac Lab:

1. **Input**: 24×32 depth image
2. **Processing**: 
   - Near clip: 0.3m, Far clip: 2.0m
   - NaN/Inf → far_clip
   - Clipping and normalization
3. **CNN Backbone**: 2-layer CNN with max pooling
4. **Output**: 128D latent representation

## Network Architecture

### ActorDepthCNN
- **Proprioceptive MLP**: 3-layer MLP processing proprioceptive data
- **Depth CNN**: 2-layer CNN processing depth images
- **Action Head**: Linear layer combining both representations

### ActorCriticDepthCNNRecurrent (Optional)
- Same as above + LSTM/GRU for temporal modeling
- Hidden states maintained between inference calls

## Examples

See `example.py` for comprehensive usage examples:

```python
python -m go1_sim2real.example
```

## API Reference

### Go1PolicyInference

Main inference class.

#### Methods

- `init()`: Initialize network and observation manager
- `load_policy(path)`: Load policy from .pt file
- `update_observation(...)`: Update observation data
- `step()`: Compute action using policy
- `apply_action()`: Apply action to robot
- `reset()`: Reset system state
- `benchmark()`: Benchmark inference performance

#### Properties

- `current_observation`: Current observation tensor
- `current_action`: Current action tensor
- `active_command`: Current velocity command

## Integration with Real Robot

To integrate with your real robot:

1. **Replace `apply_action()`**: Implement actual robot control
2. **Update observation**: Provide real sensor data
3. **Handle timing**: Ensure proper inference timing
4. **Safety**: Add safety checks and limits

Example integration:

```python
def apply_action(self, action):
    # Convert to joint commands
    joint_commands = self._action_to_joint_commands(action)
    
    # Send to robot
    self.robot.send_joint_commands(joint_commands)
    
    # Safety checks
    self._safety_check(joint_commands)
```

## Performance

Typical performance on CPU:
- **Inference time**: ~2-5ms
- **FPS**: 200-500 FPS
- **Memory usage**: ~50MB

## Troubleshooting

### Common Issues

1. **Observation size mismatch**: Check observation dimensions
2. **Policy loading fails**: Verify .pt file format
3. **Slow inference**: Use GPU if available
4. **Memory issues**: Reduce batch size

### Debug Mode

Enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on Isaac Lab and RSL-RL frameworks
- Go1 robot model from Unitree Robotics
- Depth processing inspired by Isaac Lab implementations
