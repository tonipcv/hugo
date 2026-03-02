from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StockTradingEnv(gym.Env):
    """
    Advanced stock trading environment with realistic features:
    - Multiple stocks portfolio
    - Transaction costs and slippage
    - Market indicators (RSI, MACD, Bollinger Bands)
    - Risk management (stop-loss, position limits)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        stock_data: Optional[np.ndarray] = None,
        num_stocks: int = 5,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_position_per_stock: float = 0.3,
        lookback_window: int = 20,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.num_stocks = num_stocks
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_per_stock = max_position_per_stock
        self.lookback_window = lookback_window
        self.render_mode = render_mode
        
        if stock_data is None:
            self.stock_data = self._generate_synthetic_data(1000, num_stocks)
        else:
            self.stock_data = stock_data
        
        self.num_features = 4 + num_stocks * 5
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32
        )
        
        self.action_space = spaces.MultiDiscrete([3] * num_stocks)
        
        self.reset()
    
    def _generate_synthetic_data(self, length: int, num_stocks: int) -> np.ndarray:
        """Generate synthetic stock price data with realistic patterns."""
        np.random.seed(42)
        
        data = np.zeros((length, num_stocks))
        
        for i in range(num_stocks):
            trend = np.random.uniform(-0.0002, 0.0005)
            volatility = np.random.uniform(0.01, 0.03)
            
            returns = np.random.normal(trend, volatility, length)
            
            for j in range(1, length, 50):
                if np.random.random() < 0.3:
                    shock = np.random.uniform(-0.05, 0.05)
                    returns[j:j+10] += shock
            
            prices = 100 * np.exp(np.cumsum(returns))
            data[:, i] = prices
        
        return data
    
    def _calculate_indicators(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate technical indicators for a stock."""
        if len(prices) < self.lookback_window:
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'bb_position': 0.5,
                'momentum': 0.0,
            }
        
        recent_prices = prices[-self.lookback_window:]
        
        gains = np.maximum(np.diff(recent_prices), 0)
        losses = np.maximum(-np.diff(recent_prices), 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        ema_12 = np.mean(recent_prices[-12:]) if len(recent_prices) >= 12 else recent_prices[-1]
        ema_26 = np.mean(recent_prices[-26:]) if len(recent_prices) >= 26 else recent_prices[-1]
        macd = ema_12 - ema_26
        
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        if upper_band - lower_band > 0:
            bb_position = (recent_prices[-1] - lower_band) / (upper_band - lower_band)
        else:
            bb_position = 0.5
        
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] != 0 else 0
        
        return {
            'rsi': rsi,
            'macd': macd / recent_prices[-1] if recent_prices[-1] != 0 else 0,
            'bb_position': bb_position,
            'momentum': momentum,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.positions = np.zeros(self.num_stocks)
        self.total_profit = 0.0
        self.trades_made = 0
        self.portfolio_values = [self.initial_balance]
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with market data and portfolio state."""
        current_prices = self.stock_data[self.current_step]
        portfolio_value = self.balance + np.sum(self.positions * current_prices)
        
        obs = [
            self.balance / self.initial_balance,
            portfolio_value / self.initial_balance,
            np.sum(self.positions * current_prices) / portfolio_value if portfolio_value > 0 else 0,
            self.trades_made / 100.0,
        ]
        
        for i in range(self.num_stocks):
            price_history = self.stock_data[max(0, self.current_step - self.lookback_window):self.current_step + 1, i]
            indicators = self._calculate_indicators(price_history)
            
            obs.extend([
                current_prices[i] / 100.0,
                self.positions[i] / (self.initial_balance * self.max_position_per_stock / current_prices[i]) if current_prices[i] > 0 else 0,
                indicators['rsi'] / 100.0,
                indicators['macd'],
                indicators['momentum'],
            ])
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute trading actions.
        Actions: 0 = Sell, 1 = Hold, 2 = Buy for each stock
        """
        current_prices = self.stock_data[self.current_step]
        prev_portfolio_value = self.balance + np.sum(self.positions * current_prices)
        
        for i, act in enumerate(action):
            price = current_prices[i]
            
            if act == 0 and self.positions[i] > 0:
                sell_amount = self.positions[i]
                effective_price = price * (1 - self.slippage)
                proceeds = sell_amount * effective_price * (1 - self.transaction_cost)
                
                self.balance += proceeds
                self.positions[i] = 0
                self.trades_made += 1
            
            elif act == 2:
                max_shares = (self.balance * self.max_position_per_stock) / price
                
                if max_shares > 0 and self.balance > price:
                    buy_amount = min(max_shares, self.balance * 0.95 / price)
                    effective_price = price * (1 + self.slippage)
                    cost = buy_amount * effective_price * (1 + self.transaction_cost)
                    
                    if cost <= self.balance:
                        self.positions[i] += buy_amount
                        self.balance -= cost
                        self.trades_made += 1
        
        self.current_step += 1
        
        new_prices = self.stock_data[self.current_step]
        new_portfolio_value = self.balance + np.sum(self.positions * new_prices)
        
        self.portfolio_values.append(new_portfolio_value)
        
        reward = (new_portfolio_value - prev_portfolio_value) / self.initial_balance
        
        sharpe_ratio = self._calculate_sharpe_ratio()
        reward += sharpe_ratio * 0.1
        
        self.total_profit = new_portfolio_value - self.initial_balance
        
        terminated = self.current_step >= len(self.stock_data) - 1
        truncated = new_portfolio_value <= self.initial_balance * 0.5
        
        info = {
            "portfolio_value": new_portfolio_value,
            "balance": self.balance,
            "positions": self.positions.copy(),
            "total_profit": self.total_profit,
            "trades_made": self.trades_made,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self._calculate_max_drawdown(),
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of portfolio returns."""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown of portfolio."""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        portfolio_array = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - running_max) / running_max
        
        return np.min(drawdown)
    
    def render(self) -> None:
        if self.render_mode == "human":
            current_prices = self.stock_data[self.current_step]
            portfolio_value = self.balance + np.sum(self.positions * current_prices)
            
            print(f"Step: {self.current_step} | Portfolio: ${portfolio_value:.2f} | "
                  f"Balance: ${self.balance:.2f} | Profit: ${self.total_profit:.2f} | "
                  f"Trades: {self.trades_made}")
