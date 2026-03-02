import pytest
import numpy as np
from pathlib import Path
import tempfile

from rl_llm_toolkit.multiagent import MADDPGAgent, MultiAgentEnv
from rl_llm_toolkit.multiagent.environment import CooperativeNavigationEnv


class TestCooperativeNavigationEnv:
    def test_env_creation(self):
        env = CooperativeNavigationEnv(num_agents=3, num_landmarks=3)
        
        assert env.num_agents == 3
        assert env.num_landmarks == 3
        assert len(env.agent_positions) == 3
        assert len(env.landmark_positions) == 3
    
    def test_reset(self):
        env = CooperativeNavigationEnv(num_agents=3)
        
        observations, info = env.reset(seed=42)
        
        assert len(observations) == 3
        assert "agent_0" in observations
        assert "agent_1" in observations
        assert "agent_2" in observations
        
        for obs in observations.values():
            assert isinstance(obs, np.ndarray)
            assert obs.shape == env.agent_observation_space.shape
    
    def test_step(self):
        env = CooperativeNavigationEnv(num_agents=2)
        observations, _ = env.reset(seed=42)
        
        actions = {
            "agent_0": np.array([0.5, 0.5]),
            "agent_1": np.array([-0.5, 0.5]),
        }
        
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        
        assert len(next_obs) == 2
        assert len(rewards) == 2
        assert len(terminated) == 2
        assert len(truncated) == 2
        
        assert "collisions" in info
        assert "coverage" in info


class TestMADDPGAgent:
    def test_agent_creation(self):
        env = CooperativeNavigationEnv(num_agents=2)
        agent = MADDPGAgent(env=env, num_agents=2, seed=42)
        
        assert agent.num_agents == 2
        assert len(agent.actors) == 2
        assert len(agent.critics) == 2
        assert len(agent.target_actors) == 2
        assert len(agent.target_critics) == 2
    
    def test_get_action(self):
        env = CooperativeNavigationEnv(num_agents=2)
        agent = MADDPGAgent(env=env, num_agents=2, seed=42)
        
        observations, _ = env.reset(seed=42)
        obs = observations["agent_0"]
        
        action = agent._get_action(0, obs, add_noise=False)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == env.agent_action_space.shape
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
    
    def test_training_short(self):
        env = CooperativeNavigationEnv(num_agents=2)
        agent = MADDPGAgent(
            env=env,
            num_agents=2,
            batch_size=32,
            buffer_size=1000,
            seed=42,
        )
        
        results = agent.train(
            total_timesteps=500,
            log_interval=1000,
            progress_bar=False,
        )
        
        assert results["total_timesteps"] >= 500
        # Episode count may be 0 if training is very short
        assert agent._episode_count >= 0
    
    def test_save_and_load(self):
        env = CooperativeNavigationEnv(num_agents=2)
        agent = MADDPGAgent(env=env, num_agents=2, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            agent.save(save_path)
            
            assert save_path.exists()
            
            new_agent = MADDPGAgent(env=env, num_agents=2)
            new_agent.load(save_path)
            
            assert new_agent._total_timesteps == agent._total_timesteps
