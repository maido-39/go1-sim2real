# Copyright (c) 2024, Go1 Sim2Real Policy Inference
# All rights reserved.

"""Go1 Sim2Real Policy Inference Package."""

__version__ = "1.0.0"
__author__ = "Go1 Sim2Real Team"
__email__ = "contact@example.com"

from .config import Go1Config, ObservationConfig, get_default_config, get_config_from_cli_args
from .network import (
    DepthOnlyFCBackbone, 
    ActorDepthCNN, 
    ActorCriticDepthCNN, 
    ActorCriticDepthCNNRecurrent,
    create_network
)
from .observation import (
    ObservationData, 
    ObservationBuffer, 
    ProprioHistoryBuffer, 
    ObservationManager,
    process_depth_image,
    create_dummy_observation
)
from .inference import Go1PolicyInference, create_inference_system
from .utils import (
    load_checkpoint,
    validate_checkpoint,
    get_model_info_from_checkpoint,
    create_config_from_checkpoint,
    save_config,
    load_config,
    validate_observation,
    print_observation_info,
    print_action_info,
    create_dummy_data,
    benchmark_inference
)

__all__ = [
    # Config
    "Go1Config",
    "ObservationConfig", 
    "get_default_config",
    "get_config_from_cli_args",
    
    # Network
    "DepthOnlyFCBackbone",
    "ActorDepthCNN",
    "ActorCriticDepthCNN", 
    "ActorCriticDepthCNNRecurrent",
    "create_network",
    
    # Observation
    "ObservationData",
    "ObservationBuffer",
    "ProprioHistoryBuffer",
    "ObservationManager",
    "process_depth_image",
    "create_dummy_observation",
    
    # Inference
    "Go1PolicyInference",
    "create_inference_system",
    
    # Utils
    "load_checkpoint",
    "validate_checkpoint", 
    "get_model_info_from_checkpoint",
    "create_config_from_checkpoint",
    "save_config",
    "load_config",
    "validate_observation",
    "print_observation_info",
    "print_action_info",
    "create_dummy_data",
    "benchmark_inference"
]
