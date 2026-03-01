import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
from rl_llm_toolkit.core.environment import RLEnvironment
from rl_llm_toolkit.agents.ppo import PPOAgent
from rl_llm_toolkit.agents.dqn import DQNAgent


class TestPPOAgent:
    def test_agent_creation(self):
        env = RLEnvironment("CartPole-v1")
        agent = PPOAgent(env=env, seed=42)
        
        assert agent.env == env
        assert agent.seed == 42
        assert agent.network is not None
        env.close()
    
    def test_predict(self):
        env = RLEnvironment("CartPole-v1")
        agent = PPOAgent(env=env, seed=42)
        
        obs, _ = env.reset(seed=42)
        action, info = agent.predict(obs, deterministic=True)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < env.action_space.n
        assert "log_prob" in info
        assert "value" in info
        env.close()
    
    def test_training_short(self):
        env = RLEnvironment("CartPole-v1")
        agent = PPOAgent(
            env=env,
            n_steps=128,
            batch_size=32,
            seed=42,
        )
        
        results = agent.train(
            total_timesteps=256,
            log_interval=100,
            progress_bar=False,
        )
        
        assert results["total_timesteps"] >= 256
        assert results["episodes"] > 0
        env.close()
    
    def test_save_and_load(self):
        env = RLEnvironment("CartPole-v1")
        agent = PPOAgent(env=env, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            agent.save(save_path)
            
            assert save_path.exists()
            
            new_agent = PPOAgent(env=env)
            new_agent.load(save_path)
            
            assert new_agent._total_timesteps == agent._total_timesteps
        
        env.close()
    
    def test_evaluate(self):
        env = RLEnvironment("CartPole-v1")
        agent = PPOAgent(env=env, seed=42)
        
        results = agent.evaluate(episodes=3, deterministic=True)
        
        assert "mean_reward" in results
        assert "std_reward" in results
        assert "min_reward" in results
        assert "max_reward" in results
        assert "mean_length" in results
        env.close()


class TestDQNAgent:
    def test_agent_creation(self):
        env = RLEnvironment("CartPole-v1")
        agent = DQNAgent(env=env, seed=42)
        
        assert agent.env == env
        assert agent.seed == 42
        assert agent.q_network is not None
        assert agent.target_network is not None
        env.close()
    
    def test_epsilon_decay(self):
        env = RLEnvironment("CartPole-v1")
        agent = DQNAgent(
            env=env,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=1000,
        )
        
        epsilon_0 = agent.get_epsilon(0)
        epsilon_500 = agent.get_epsilon(500)
        epsilon_1000 = agent.get_epsilon(1000)
        
        assert epsilon_0 == 1.0
        assert 0.1 < epsilon_500 < 1.0
        assert epsilon_1000 == 0.1
        env.close()
    
    def test_predict(self):
        env = RLEnvironment("CartPole-v1")
        agent = DQNAgent(env=env, seed=42)
        
        obs, _ = env.reset(seed=42)
        action, info = agent.predict(obs, deterministic=True)
        
        assert isinstance(action, np.ndarray)
        assert 0 <= action.item() < env.action_space.n
        assert "q_values" in info
        assert "max_q_value" in info
        env.close()
    
    def test_replay_buffer(self):
        env = RLEnvironment("CartPole-v1")
        agent = DQNAgent(env=env, buffer_size=100, seed=42)
        
        obs, _ = env.reset(seed=42)
        for _ in range(50):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            if done:
                obs, _ = env.reset()
        
        assert len(agent.replay_buffer) == 50
        
        batch = agent.replay_buffer.sample(32)
        assert len(batch) == 5
        assert batch[0].shape[0] == 32
        env.close()
    
    def test_save_and_load(self):
        env = RLEnvironment("CartPole-v1")
        agent = DQNAgent(env=env, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            agent.save(save_path)
            
            assert save_path.exists()
            
            new_agent = DQNAgent(env=env)
            new_agent.load(save_path)
            
            assert new_agent._total_timesteps == agent._total_timesteps
        
        env.close()
