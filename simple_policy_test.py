#!/usr/bin/env python3
"""
간단한 Go1 정책 테스트용 스크립트
- 체크포인트에서 정책 로드
- observation dict를 입력받아 action 출력
- 최소한의 의존성으로 동작
"""

import torch
import numpy as np
from typing import Dict, List, Union
import os

class SimpleGo1Policy:
    """간단한 Go1 정책 클래스 - 테스트용"""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.config = None
        
        # 체크포인트에서 설정 자동 감지
        self._load_and_detect_config()
        
    def _load_and_detect_config(self):
        """체크포인트를 로드하고 설정을 자동 감지"""
        print(f"체크포인트 로딩: {self.checkpoint_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model_state = checkpoint["model_state_dict"]
        
        # 설정 자동 감지
        self.config = self._detect_config_from_checkpoint(model_state)
        print(f"감지된 설정: {self.config}")
        
        # 모델 생성 및 로드
        self._create_and_load_model(model_state)
        
    def _detect_config_from_checkpoint(self, model_state: Dict) -> Dict:
        """체크포인트에서 설정 자동 감지"""
        config = {}
        
        # Actor prop_mlp 입력 크기에서 설정 감지
        if "actor.prop_mlp.0.weight" in model_state:
            prop_mlp_input_size = model_state["actor.prop_mlp.0.weight"].shape[1]
            config["prop_mlp_input_size"] = prop_mlp_input_size
            
            # 가능한 history_length들 시도
            possible_history_lengths = [8, 9, 10, 7, 6, 5, 4, 3, 2, 1]
            
            for history_length in possible_history_lengths:
                if prop_mlp_input_size % (history_length + 1) == 0:
                    proprio_dim = prop_mlp_input_size // (history_length + 1)
                    if 20 <= proprio_dim <= 80:  # 합리적인 범위
                        config["num_proprio_obs"] = proprio_dim
                        config["history_length"] = history_length
                        print(f"감지됨: proprio_dim={proprio_dim}, history_length={history_length}")
                        break
            
            if "num_proprio_obs" not in config:
                # Fallback
                config["num_proprio_obs"] = 45
                config["history_length"] = 9
                print("Fallback 설정 사용")
        
        # Depth 이미지 설정
        config["obs_depth_shape"] = (24, 32)
        config["num_actions"] = 12
        
        # CNN 사용 여부 확인
        config["use_cnn"] = "actor.depth_backbone.image_compression.0.weight" in model_state
        
        return config
    
    def _create_and_load_model(self, model_state: Dict):
        """간단한 모델 생성 및 로드"""
        from go1_sim2real.network import ActorCriticDepthCNN
        from go1_sim2real.config import Go1Config
        
        # 설정으로 Go1Config 생성
        go1_config = Go1Config(
            device=self.device,
            num_proprio_obs=self.config["num_proprio_obs"],
            history_length=self.config["history_length"],
            obs_depth_shape=self.config["obs_depth_shape"],
            num_actions=self.config["num_actions"],
            use_cnn=self.config["use_cnn"]
        )
        
        # 네트워크 생성
        self.model = ActorCriticDepthCNN(
            num_actor_obs=go1_config.num_actor_obs_prop,
            num_critic_obs=go1_config.total_policy_input_dim,
            num_actions=go1_config.num_actions,
            num_actor_obs_prop=go1_config.num_actor_obs_prop,
            num_critic_obs_prop=go1_config.num_actor_obs_prop,
            obs_depth_shape=go1_config.obs_depth_shape,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu"
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # 체크포인트 로드
        self.model.load_state_dict(model_state)
        print("모델 로딩 완료!")
        
    def predict(self, observation: Dict[str, Union[List[float], np.ndarray]]) -> np.ndarray:
        """
        observation dict를 받아서 action을 반환
        
        Args:
            observation: {
                'base_ang_vel': [float, float, float],
                'base_rpy': [float, float, float], 
                'velocity_commands': [float, float, float],
                'joint_pos': [float] * 12,
                'joint_vel': [float] * 12,
                'actions': [float] * 12,
                'depth_image': np.ndarray (24, 32)
            }
        
        Returns:
            action: np.ndarray (12,)
        """
        # observation을 텐서로 변환
        obs_tensor = self._dict_to_tensor(observation)
        
        # 모델 추론
        with torch.no_grad():
            action = self.model.act_inference(obs_tensor)
        
        return action.cpu().numpy()
    
    def _dict_to_tensor(self, observation: Dict) -> torch.Tensor:
        """observation dict를 모델 입력 텐서로 변환"""
        # Proprioceptive observation 구성
        proprio_obs = []
        
        # 각 관측값을 추가
        proprio_obs.extend(observation['base_ang_vel'])
        proprio_obs.extend(observation['base_rpy'])
        proprio_obs.extend(observation['velocity_commands'])
        proprio_obs.extend(observation['joint_pos'])
        proprio_obs.extend(observation['joint_vel'])
        proprio_obs.extend(observation['actions'])
        
        # 현재 proprioceptive observation 차원 확인
        current_proprio_dim = len(proprio_obs)
        expected_proprio_dim = self.config["num_proprio_obs"]
        
        if current_proprio_dim != expected_proprio_dim:
            print(f"Warning: proprioceptive observation 차원 불일치")
            print(f"  현재: {current_proprio_dim}, 예상: {expected_proprio_dim}")
            
            # 차원 맞추기
            if current_proprio_dim < expected_proprio_dim:
                # 부족한 차원을 0으로 채움
                proprio_obs.extend([0.0] * (expected_proprio_dim - current_proprio_dim))
            else:
                # 초과하는 차원을 잘라냄
                proprio_obs = proprio_obs[:expected_proprio_dim]
        
        # History buffer 구성 (현재 관측값을 반복)
        history_length = self.config["history_length"]
        history_buffer = []
        for _ in range(history_length + 1):
            history_buffer.extend(proprio_obs)
        
        # Depth 이미지 처리
        depth_image = observation['depth_image']
        if isinstance(depth_image, np.ndarray):
            depth_image = torch.from_numpy(depth_image).float()
        
        # 배치 차원 추가
        depth_image = depth_image.unsqueeze(0)  # (1, 24, 32)
        
        # 최종 observation 텐서 생성
        obs_tensor = torch.cat([
            torch.tensor(history_buffer, dtype=torch.float32).unsqueeze(0),
            depth_image.flatten().unsqueeze(0)
        ], dim=1)
        
        return obs_tensor.to(self.device)


def test_simple_policy():
    """간단한 정책 테스트"""
    print("=== 간단한 Go1 정책 테스트 ===")
    
    # 정책 로드
    policy = SimpleGo1Policy("./model_14999.pt", device="cpu")
    
    # 테스트용 observation 생성
    observation = {
        'base_ang_vel': [0.1, -0.05, 0.02],
        'base_rpy': [0.05, 0.1, 0.0],
        'velocity_commands': [0.5, 0.0, 0.0],
        'joint_pos': [0.0] * 12,
        'joint_vel': [0.1] * 12,
        'actions': [0.0] * 12,
        'depth_image': np.random.rand(24, 32).astype(np.float32)
    }
    
    print(f"입력 observation:")
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # 액션 예측
    print("\n액션 예측 중...")
    action = policy.predict(observation)
    
    print(f"예측된 액션: {action}")
    print(f"액션 shape: {action.shape}")
    
    return action


if __name__ == "__main__":
    # 테스트 실행
    test_simple_policy()
