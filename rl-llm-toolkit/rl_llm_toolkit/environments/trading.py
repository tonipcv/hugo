from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CryptoTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        price_data: Optional[np.ndarray] = None,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        max_position: float = 1.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        if price_data is None:
            self.price_data = self._generate_synthetic_prices(1000)
        else:
            self.price_data = price_data
        
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position = max_position
        self.render_mode = render_mode
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)
        
        self.reset()
    
    def _generate_synthetic_prices(self, length: int) -> np.ndarray:
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, length)
        prices = 100 * np.exp(np.cumsum(returns))
        return prices
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_profit = 0.0
        self.trades_made = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        window_size = 5
        start_idx = max(0, self.current_step - window_size)
        
        price_window = self.price_data[start_idx:self.current_step + 1]
        if len(price_window) < window_size + 1:
            price_window = np.pad(
                price_window,
                (window_size + 1 - len(price_window), 0),
                mode='edge'
            )
        
        returns = np.diff(price_window) / price_window[:-1]
        
        current_price = self.price_data[self.current_step]
        portfolio_value = self.balance + self.position * current_price
        
        obs = np.array([
            current_price / 100.0,
            self.balance / self.initial_balance,
            self.position / self.max_position,
            portfolio_value / self.initial_balance,
            *returns,
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self.price_data[self.current_step]
        prev_portfolio_value = self.balance + self.position * current_price
        
        if action == 0:
            if self.position > 0:
                self.balance += self.position * current_price * (1 - self.transaction_fee)
                self.position = 0
                self.trades_made += 1
        
        elif action == 1:
            pass
        
        elif action == 2:
            if self.position == 0 and self.balance > 0:
                shares_to_buy = (self.balance * 0.95) / current_price
                cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                if cost <= self.balance:
                    self.position = shares_to_buy
                    self.balance -= cost
                    self.trades_made += 1
        
        self.current_step += 1
        
        new_price = self.price_data[self.current_step]
        new_portfolio_value = self.balance + self.position * new_price
        
        reward = (new_portfolio_value - prev_portfolio_value) / self.initial_balance
        
        self.total_profit = new_portfolio_value - self.initial_balance
        
        terminated = self.current_step >= len(self.price_data) - 1
        truncated = new_portfolio_value <= 0
        
        info = {
            "portfolio_value": new_portfolio_value,
            "balance": self.balance,
            "position": self.position,
            "total_profit": self.total_profit,
            "trades_made": self.trades_made,
            "price": new_price,
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self) -> None:
        if self.render_mode == "human":
            current_price = self.price_data[self.current_step]
            portfolio_value = self.balance + self.position * current_price
            print(
                f"Step: {self.current_step} | "
                f"Price: ${current_price:.2f} | "
                f"Balance: ${self.balance:.2f} | "
                f"Position: {self.position:.4f} | "
                f"Portfolio: ${portfolio_value:.2f} | "
                f"Profit: ${self.total_profit:.2f}"
            )
