from rl_llm_toolkit import PPOAgent
from rl_llm_toolkit.core.environment import RLEnvironment
from rl_llm_toolkit.environments.stock_trading import StockTradingEnv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("="*60)
    print("Advanced Stock Trading Environment Example")
    print("="*60)
    
    print("\n1. Creating stock trading environment...")
    
    num_stocks = 5
    env_raw = StockTradingEnv(
        num_stocks=num_stocks,
        initial_balance=100000.0,
        transaction_cost=0.001,
        slippage=0.0005,
        max_position_per_stock=0.3,
        lookback_window=20,
    )
    
    env = RLEnvironment.from_gym_env(env_raw)
    
    print(f"Environment created:")
    print(f"  Number of stocks: {num_stocks}")
    print(f"  Initial balance: $100,000")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    print("\n2. Training PPO agent...")
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        seed=42,
    )
    
    results = agent.train(
        total_timesteps=100000,
        log_interval=10,
        eval_interval=20000,
        eval_episodes=5,
        progress_bar=True,
    )
    
    print(f"\n✅ Training complete!")
    print(f"Total episodes: {results['episodes']}")
    print(f"Total timesteps: {results['total_timesteps']}")
    
    print("\n3. Evaluating trained agent...")
    eval_results = agent.evaluate(episodes=10, deterministic=True)
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"  Std Reward: {eval_results['std_reward']:.2f}")
    print(f"  Min Reward: {eval_results['min_reward']:.2f}")
    print(f"  Max Reward: {eval_results['max_reward']:.2f}")
    
    print("\n4. Running detailed episode...")
    obs, _ = env.reset(seed=42)
    done = False
    step = 0
    episode_info = []
    
    while not done and step < 200:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_info.append({
            "step": step,
            "portfolio_value": info.get("portfolio_value", 0),
            "balance": info.get("balance", 0),
            "total_profit": info.get("total_profit", 0),
            "trades_made": info.get("trades_made", 0),
        })
        
        if step % 50 == 0:
            print(f"\nStep {step}:")
            print(f"  Portfolio Value: ${info.get('portfolio_value', 0):.2f}")
            print(f"  Cash Balance: ${info.get('balance', 0):.2f}")
            print(f"  Total Profit: ${info.get('total_profit', 0):.2f}")
            print(f"  Trades Made: {info.get('trades_made', 0)}")
            print(f"  Sharpe Ratio: {info.get('sharpe_ratio', 0):.3f}")
        
        step += 1
    
    print(f"\n5. Visualizing trading performance...")
    
    portfolio_values = [e["portfolio_value"] for e in episode_info]
    profits = [e["total_profit"] for e in episode_info]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(portfolio_values, linewidth=2)
    ax1.axhline(y=100000, color='r', linestyle='--', label='Initial Balance')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Value Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(profits, linewidth=2, color='green')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Profit/Loss ($)')
    ax2.set_title('Cumulative Profit/Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path("./outputs/stock_trading")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "trading_performance.png", dpi=150)
    print(f"✅ Chart saved to {output_dir / 'trading_performance.png'}")
    
    model_path = output_dir / "stock_trading_agent.pt"
    agent.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    final_info = episode_info[-1]
    print(f"\n{'='*60}")
    print("Final Episode Summary")
    print(f"{'='*60}")
    print(f"Initial Balance: $100,000.00")
    print(f"Final Portfolio Value: ${final_info['portfolio_value']:.2f}")
    print(f"Total Profit: ${final_info['total_profit']:.2f}")
    print(f"Return: {(final_info['total_profit'] / 100000) * 100:.2f}%")
    print(f"Total Trades: {final_info['trades_made']}")
    
    env.close()


if __name__ == "__main__":
    main()
