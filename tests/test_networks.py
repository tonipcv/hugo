import pytest
import torch
import numpy as np
from gymnasium import spaces
from rl_llm_toolkit.agents.networks import ActorCriticNetwork, QNetwork


class TestActorCriticNetwork:
    def test_discrete_action_space(self):
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        
        network = ActorCriticNetwork(obs_space, action_space)
        
        obs = torch.randn(1, 4)
        action, log_prob, entropy, value = network.get_action_and_value(obs)
        
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1,)
    
    def test_continuous_action_space(self):
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        network = ActorCriticNetwork(obs_space, action_space)
        
        obs = torch.randn(1, 4)
        action, log_prob, entropy, value = network.get_action_and_value(obs)
        
        assert action.shape == (1, 2)
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1,)
    
    def test_deterministic_action(self):
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        
        network = ActorCriticNetwork(obs_space, action_space)
        
        obs = torch.randn(1, 4)
        action1, _, _, _ = network.get_action_and_value(obs, deterministic=True)
        action2, _, _, _ = network.get_action_and_value(obs, deterministic=True)
        
        assert torch.equal(action1, action2)
    
    def test_value_prediction(self):
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        
        network = ActorCriticNetwork(obs_space, action_space)
        
        obs = torch.randn(1, 4)
        value = network.get_value(obs)
        
        assert value.shape == (1,)
        assert isinstance(value.item(), float)


class TestQNetwork:
    def test_forward_pass(self):
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        
        network = QNetwork(obs_space, action_space)
        
        obs = torch.randn(1, 4)
        q_values = network(obs)
        
        assert q_values.shape == (1, 2)
    
    def test_batch_processing(self):
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(3)
        
        network = QNetwork(obs_space, action_space)
        
        obs = torch.randn(32, 4)
        q_values = network(obs)
        
        assert q_values.shape == (32, 3)
    
    def test_action_selection(self):
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        
        network = QNetwork(obs_space, action_space)
        
        obs = torch.randn(1, 4)
        q_values = network(obs)
        action = q_values.argmax(dim=1)
        
        assert action.shape == (1,)
        assert 0 <= action.item() < 2
