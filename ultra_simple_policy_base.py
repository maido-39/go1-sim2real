#!/usr/bin/env python3
"""
초간단 Go1 Base 정책 테스트 함수
- depth가 없는 base policy만 처리
- 최소한의 코드로 정책 로드 및 추론
"""

import torch
import numpy as np
from typing import Dict, List, Union


def load_go1_base_policy(checkpoint_path: str, device: str = "cpu"):
    """Go1 Base 정책을 로드하고 간단한 추론 함수를 반환"""
    
    print(f"체크포인트 로딩: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint["model_state_dict"]
    
    # Actor 네트워크의 첫 번째 레이어 가중치로 input size 파악
    # actor.0.weight shape: (hidden, input_dim)
    actor_input_size = model_state["actor.0.weight"].shape[1]
    print(f"actor_input_size: {actor_input_size}")
    
    # 가능한 설정들 시도
    for history_length in [8, 9, 10, 7, 6, 5, 4, 3, 2, 1]:
        if actor_input_size % (history_length + 1) == 0:
            proprio_dim = actor_input_size // (history_length + 1)
            if 20 <= proprio_dim <= 80:
                print(f"감지된 설정: proprio_dim={proprio_dim}, history_length={history_length}")
                break
    else:
        proprio_dim = 45
        history_length = 9
        print("Fallback 설정 사용")
    
    # 모델 생성 - 일반 ActorCritic (MLP만)
    import sys
    import os
    # 프로젝트 루트 경로 추가
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from rsl_rl.rsl_rl.modules.actor_critic import ActorCritic
    
    model = ActorCritic(
        num_actor_obs=proprio_dim * (history_length + 1),
        num_critic_obs=proprio_dim * (history_length + 1),
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
        
        # History buffer 구성
        history_buffer = []
        for _ in range(history_length + 1):
            history_buffer.extend(proprio_obs)
        
        # 최종 observation 텐서
        obs_tensor = torch.tensor(history_buffer, dtype=torch.float32).unsqueeze(0).to(device)
        
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

