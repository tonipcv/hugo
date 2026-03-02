#!/usr/bin/env python3
"""
Advanced Trading Example with RL-LLM Toolkit

Demonstrates:
- Stock trading environment with multiple stocks
- PPO agent training
- Performance visualization
- Risk metrics calculation
- Portfolio optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rl_llm_toolkit import RLEnvironment, PPOAgent
from rl_llm_toolkit.environments.stock_trading import StockTradingEnv


def generate_synthetic_stock_data(num_stocks=5, num_days=252):
    """Generate synthetic stock price data."""
    np.random.seed(42)
    
    # Starting prices
    start_prices = np.random.uniform(50, 200, num_stocks)
    
    # Generate returns with different volatilities
    volatilities = np.random.uniform(0.01, 0.03, num_stocks)
    drifts = np.random.uniform(-0.0002, 0.0005, num_stocks)
    
    prices = np.zeros((num_days, num_stocks))
    prices[0] = start_prices
    
    for day in range(1, num_days):
        returns = np.random.normal(drifts, volatilities)
        prices[day] = prices[day-1] * (1 + returns)
    
    return prices


def calculate_portfolio_metrics(portfolio_values):
    """Calculate portfolio performance metrics."""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Sharpe ratio (annualized)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    # Maximum drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax
    max_drawdown = np.min(drawdown)
    
    # Total return
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252)
    
    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "volatility": volatility,
        "final_value": portfolio_values[-1]
    }


def plot_trading_results(portfolio_values, stock_prices, trades, save_path=None):
    """Plot trading results."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Portfolio value over time
    axes[0].plot(portfolio_values, linewidth=2)
    axes[0].set_title("Portfolio Value Over Time", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Trading Day")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].grid(True, alpha=0.3)
    
    # Stock prices
    for i in range(stock_prices.shape[1]):
        axes[1].plot(stock_prices[:, i], label=f"Stock {i+1}", alpha=0.7)
    axes[1].set_title("Stock Prices", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Trading Day")
    axes[1].set_ylabel("Price ($)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax * 100
    axes[2].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
    axes[2].plot(drawdown, color='red', linewidth=2)
    axes[2].set_title("Portfolio Drawdown", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Trading Day")
    axes[2].set_ylabel("Drawdown (%)")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    print("=" * 60)
    print("Advanced Stock Trading with RL-LLM Toolkit")
    print("=" * 60)
    
    # Generate synthetic stock data
    print("\n1. Generating synthetic stock data...")
    num_stocks = 5
    stock_prices = generate_synthetic_stock_data(num_stocks=num_stocks, num_days=252)
    print(f"   Generated {num_stocks} stocks with 252 trading days")
    
    # Create trading environment
    print("\n2. Creating stock trading environment...")
    env = StockTradingEnv(
        price_data=stock_prices,
        initial_balance=100000.0,
        transaction_cost=0.001,  # 0.1% transaction cost
        slippage=0.0005,  # 0.05% slippage
        max_position_per_stock=0.3,  # Max 30% per stock
    )
    print(f"   Initial balance: ${env.initial_balance:,.2f}")
    print(f"   Transaction cost: {env.transaction_cost*100:.2f}%")
    print(f"   Slippage: {env.slippage*100:.3f}%")
    
    # Create and train PPO agent
    print("\n3. Training PPO agent...")
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        seed=42
    )
    
    results = agent.train(
        total_timesteps=50000,
        log_interval=5000,
        progress_bar=True
    )
    
    print(f"\n   Training completed!")
    print(f"   Total timesteps: {results['total_timesteps']}")
    print(f"   Episodes: {len(results['stats']['episode_rewards'])}")
    
    # Evaluate agent
    print("\n4. Evaluating trained agent...")
    eval_results = agent.evaluate(episodes=10, deterministic=True)
    
    print(f"\n   Evaluation Results:")
    print(f"   Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"   Std Reward: {eval_results['std_reward']:.2f}")
    print(f"   Min Reward: {eval_results['min_reward']:.2f}")
    print(f"   Max Reward: {eval_results['max_reward']:.2f}")
    
    # Run a single episode to collect detailed metrics
    print("\n5. Running detailed evaluation episode...")
    obs, _ = env.reset(seed=42)
    done = False
    portfolio_values = [env.initial_balance]
    trades = []
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        portfolio_values.append(info['portfolio_value'])
        if info.get('trades_made', 0) > 0:
            trades.append({
                'step': env.current_step,
                'action': action,
                'portfolio_value': info['portfolio_value']
            })
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(np.array(portfolio_values))
    
    print("\n" + "=" * 60)
    print("Portfolio Performance Metrics")
    print("=" * 60)
    print(f"Initial Value:    ${env.initial_balance:,.2f}")
    print(f"Final Value:      ${metrics['final_value']:,.2f}")
    print(f"Total Return:     {metrics['total_return']*100:+.2f}%")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']*100:.2f}%")
    print(f"Volatility (Ann): {metrics['volatility']*100:.2f}%")
    print(f"Total Trades:     {len(trades)}")
    print("=" * 60)
    
    # Save model
    model_dir = Path("models/trading")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "stock_trading_ppo.pt"
    agent.save(model_path)
    print(f"\n✅ Model saved to {model_path}")
    
    # Plot results
    print("\n6. Generating visualizations...")
    plot_path = Path("results/stock_trading_results.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_trading_results(
        np.array(portfolio_values),
        stock_prices,
        trades,
        save_path=plot_path
    )
    
    # Compare with buy-and-hold strategy
    print("\n7. Comparing with buy-and-hold strategy...")
    equal_weight_portfolio = np.mean(stock_prices / stock_prices[0], axis=1) * env.initial_balance
    bh_metrics = calculate_portfolio_metrics(equal_weight_portfolio)
    
    print("\nBuy-and-Hold Strategy:")
    print(f"  Final Value:  ${bh_metrics['final_value']:,.2f}")
    print(f"  Total Return: {bh_metrics['total_return']*100:+.2f}%")
    print(f"  Sharpe Ratio: {bh_metrics['sharpe_ratio']:.3f}")
    
    print("\nRL Agent vs Buy-and-Hold:")
    print(f"  Return Difference: {(metrics['total_return'] - bh_metrics['total_return'])*100:+.2f}%")
    print(f"  Sharpe Improvement: {metrics['sharpe_ratio'] - bh_metrics['sharpe_ratio']:+.3f}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Advanced Trading Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
