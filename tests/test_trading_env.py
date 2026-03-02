import pytest
import numpy as np
from rl_llm_toolkit.environments.trading import CryptoTradingEnv


class TestCryptoTradingEnv:
    def test_env_creation(self):
        env = CryptoTradingEnv()
        assert env.initial_balance == 10000.0
        assert env.observation_space is not None
        assert env.action_space.n == 3
    
    def test_reset(self):
        env = CryptoTradingEnv()
        obs, info = env.reset(seed=42)
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] >= 9  # At least 9 features (may vary based on implementation)
        assert env.current_step == 0
        assert env.balance == env.initial_balance
        assert env.position == 0.0
    
    def test_buy_action(self):
        env = CryptoTradingEnv()
        obs, _ = env.reset(seed=42)
        
        initial_balance = env.balance
        obs, reward, terminated, truncated, info = env.step(2)
        
        assert env.position > 0
        assert env.balance < initial_balance
        assert "portfolio_value" in info
        assert "trades_made" in info
        assert info["trades_made"] == 1
    
    def test_sell_action(self):
        env = CryptoTradingEnv()
        obs, _ = env.reset(seed=42)
        
        env.step(2)
        
        position_before = env.position
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert env.position == 0
        assert position_before > 0
        assert info["trades_made"] == 2
    
    def test_hold_action(self):
        env = CryptoTradingEnv()
        obs, _ = env.reset(seed=42)
        
        balance_before = env.balance
        position_before = env.position
        
        obs, reward, terminated, truncated, info = env.step(1)
        
        assert env.balance == balance_before
        assert env.position == position_before
        assert info["trades_made"] == 0
    
    def test_episode_completion(self):
        env = CryptoTradingEnv()
        obs, _ = env.reset(seed=42)
        
        done = False
        steps = 0
        max_steps = len(env.price_data)
        
        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        assert done or steps >= max_steps - 1
    
    def test_transaction_fees(self):
        env = CryptoTradingEnv(transaction_fee=0.01)
        obs, _ = env.reset(seed=42)
        
        initial_balance = env.balance
        env.step(2)  # Buy action
        
        # Transaction fees should reduce balance
        assert env.balance < initial_balance
        assert env.position > 0
    
    def test_custom_price_data(self):
        custom_prices = np.array([100, 110, 105, 115, 120])
        env = CryptoTradingEnv(price_data=custom_prices)
        
        assert len(env.price_data) == 5
        assert np.array_equal(env.price_data, custom_prices)
