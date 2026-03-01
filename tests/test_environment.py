import pytest
import numpy as np
from rl_llm_toolkit.core.environment import RLEnvironment


class TestRLEnvironment:
    def test_environment_creation(self):
        env = RLEnvironment("CartPole-v1")
        assert env.env_id == "CartPole-v1"
        assert env.observation_space is not None
        assert env.action_space is not None
        env.close()
    
    def test_reset(self):
        env = RLEnvironment("CartPole-v1")
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.shape == env.observation_space.shape
        env.close()
    
    def test_step(self):
        env = RLEnvironment("CartPole-v1")
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()
    
    def test_episode_statistics(self):
        env = RLEnvironment("CartPole-v1")
        obs, _ = env.reset(seed=42)
        
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        stats = env.episode_statistics
        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert "mean_length" in stats
        assert "total_episodes" in stats
        assert stats["total_episodes"] > 0
        env.close()
    
    def test_state_description(self):
        env = RLEnvironment("CartPole-v1")
        obs, _ = env.reset(seed=42)
        description = env.get_state_description(obs)
        assert isinstance(description, str)
        assert len(description) > 0
        env.close()
    
    def test_action_description(self):
        env = RLEnvironment("CartPole-v1")
        action = 0
        description = env.get_action_description(action)
        assert isinstance(description, str)
        assert len(description) > 0
        env.close()
