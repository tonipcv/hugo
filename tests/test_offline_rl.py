import pytest
import numpy as np
from pathlib import Path
import tempfile

from rl_llm_toolkit.core.environment import RLEnvironment
from rl_llm_toolkit.agents.cql import CQLAgent
from rl_llm_toolkit.agents.iql import IQLAgent


class TestCQLAgent:
    def test_agent_creation(self):
        env = RLEnvironment("CartPole-v1")
        agent = CQLAgent(env=env, seed=42)
        
        assert agent.env == env
        assert agent.seed == 42
        assert agent.q_network is not None
        assert agent.target_network is not None
        env.close()
    
    def test_collect_dataset(self):
        env = RLEnvironment("CartPole-v1")
        agent = CQLAgent(env=env, seed=42)
        
        dataset = agent.collect_dataset(num_episodes=5, policy="random")
        
        assert len(dataset) > 0
        assert "obs" in dataset[0]
        assert "action" in dataset[0]
        assert "reward" in dataset[0]
        assert "next_obs" in dataset[0]
        assert "done" in dataset[0]
        env.close()
    
    def test_load_dataset(self):
        env = RLEnvironment("CartPole-v1")
        agent = CQLAgent(env=env, seed=42)
        
        dataset = agent.collect_dataset(num_episodes=5, policy="random")
        agent.load_dataset(dataset)
        
        assert len(agent.dataset) == len(dataset)
        env.close()
    
    def test_training_short(self):
        env = RLEnvironment("CartPole-v1")
        agent = CQLAgent(env=env, batch_size=32, seed=42)
        
        dataset = agent.collect_dataset(num_episodes=10, policy="random")
        agent.load_dataset(dataset)
        
        results = agent.train(
            total_timesteps=256,
            log_interval=1000,
            progress_bar=False,
        )
        
        assert results["total_timesteps"] >= 256
        assert results["updates"] > 0
        env.close()
    
    def test_save_and_load(self):
        env = RLEnvironment("CartPole-v1")
        agent = CQLAgent(env=env, seed=42)
        
        dataset = agent.collect_dataset(num_episodes=5, policy="random")
        agent.load_dataset(dataset)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            agent.save(save_path)
            
            assert save_path.exists()
            
            new_agent = CQLAgent(env=env)
            new_agent.load(save_path)
            
            assert new_agent._total_timesteps == agent._total_timesteps
        
        env.close()


class TestIQLAgent:
    def test_agent_creation(self):
        env = RLEnvironment("CartPole-v1")
        agent = IQLAgent(env=env, seed=42)
        
        assert agent.env == env
        assert agent.seed == 42
        assert agent.network is not None
        assert agent.target_network is not None
        env.close()
    
    def test_collect_dataset(self):
        env = RLEnvironment("CartPole-v1")
        agent = IQLAgent(env=env, seed=42)
        
        dataset = agent.collect_dataset(num_episodes=5, policy="random")
        
        assert len(dataset) > 0
        env.close()
    
    def test_training_short(self):
        env = RLEnvironment("CartPole-v1")
        agent = IQLAgent(env=env, batch_size=32, seed=42)
        
        dataset = agent.collect_dataset(num_episodes=10, policy="random")
        agent.load_dataset(dataset)
        
        results = agent.train(
            total_timesteps=256,
            log_interval=1000,
            progress_bar=False,
        )
        
        assert results["total_timesteps"] >= 256
        env.close()
    
    def test_predict(self):
        env = RLEnvironment("CartPole-v1")
        agent = IQLAgent(env=env, seed=42)
        
        obs, _ = env.reset(seed=42)
        action, info = agent.predict(obs, deterministic=True)
        
        assert isinstance(action, np.ndarray)
        assert 0 <= action.item() < env.action_space.n
        assert "logits" in info
        assert "v_value" in info
        env.close()
