#!/usr/bin/env python3
"""
초간단 Go1 Base 정책 테스트 함수
- depth가 없는 base policy만 처리
- 최소한의 코드로 정책 로드 및 추론
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union


class SimpleActorCritic(nn.Module):
    """간단한 Actor-Critic 네트워크 (MLP만)"""
    
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims=None, critic_hidden_dims=None, activation="elu"):
        super().__init__()
        
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]
        
        # Activation 함수
        if activation == "elu":
            act_fn = nn.ELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            act_fn = nn.ELU()
        
        # Actor 네트워크
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(act_fn)
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(act_fn)
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic 네트워크
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(act_fn)
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            critic_layers.append(act_fn)
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)
    
    def forward(self, observations):
        """Forward pass"""
        return self.actor(observations)
    
    def act_inference(self, observations):
        """Deterministic action for inference"""
        return self.actor(observations)


def load_go1_base_policy(checkpoint_path: str, device: str = "cpu"):
    """Go1 Base 정책을 로드하고 간단한 추론 함수를 반환"""
    
    print(f"체크포인트 로딩: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint["model_state_dict"]
    
    # Actor 네트워크의 첫 번째 레이어 가중치로 input size 파악
    # actor.0.weight shape: (hidden, input_dim)
    actor_input_size = model_state["actor.0.weight"].shape[1]
    proprio_dim = actor_input_size
    print(f"actor_input_size: {actor_input_size}, proprio_dim: {proprio_dim}")
    
    # 모델 생성 - 간단한 ActorCritic (MLP만)
    model = SimpleActorCritic(
        num_actor_obs=proprio_dim,
        num_critic_obs=proprio_dim,
        num_actions=12,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu"
    )
    
    model.to(device)
    model.eval()
    model.load_state_dict(model_state)
    print("모델 로딩 완료!")
    
    def predict_action(observation: Dict[str, Union[List[float], np.ndarray]]) -> np.ndarray:
        """observation dict를 받아서 action 반환"""
        
        # Proprioceptive observation 구성
        proprio_obs = []
        proprio_obs.extend(observation['base_ang_vel'])
        proprio_obs.extend(observation['base_rpy'])
        proprio_obs.extend(observation['velocity_commands'])
        proprio_obs.extend(observation['joint_pos'])
        proprio_obs.extend(observation['joint_vel'])
        proprio_obs.extend(observation['actions'])
        
        # 차원 맞추기
        if len(proprio_obs) < proprio_dim:
            proprio_obs.extend([0.0] * (proprio_dim - len(proprio_obs)))
        elif len(proprio_obs) > proprio_dim:
            proprio_obs = proprio_obs[:proprio_dim]
        
        # 최종 observation 텐서
        obs_tensor = torch.tensor(proprio_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 액션 예측
        with torch.no_grad():
            action = model.act_inference(obs_tensor)
        
        return action.cpu().numpy()
    
    return predict_action


# 사용 예시
if __name__ == "__main__":
    # 정책 로드
    predict_action = load_go1_base_policy("./model_14999.pt", device="cpu")
    
    # 테스트 observation
    observation = {
        'base_ang_vel': [0.1, -0.05, 0.02],
        'base_rpy': [0.05, 0.1, 0.0],
        'velocity_commands': [0.5, 0.0, 0.0],
        'joint_pos': [0.0] * 12,
        'joint_vel': [0.1] * 12,
        'actions': [0.0] * 12,
    }
    
    # 액션 예측
    action = predict_action(observation)
    print(f"예측된 액션: {action}")

