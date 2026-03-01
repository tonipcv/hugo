from rl_llm_toolkit.environments.trading import CryptoTradingEnv
from rl_llm_toolkit import PPOAgent
from pathlib import Path
import numpy as np


def main():
    print("Training PPO agent on Crypto Trading Environment")
    
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 2000)
    price_data = 100 * np.exp(np.cumsum(returns))
    
    env = CryptoTradingEnv(
        price_data=price_data,
        initial_balance=10000.0,
        transaction_fee=0.001,
    )
    
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        seed=42,
    )
    
    print("Starting training...")
    results = agent.train(
        total_timesteps=100000,
        log_interval=10,
        eval_interval=10000,
        eval_episodes=5,
    )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Total Episodes: {results['episodes']}")
    
    model_path = Path("./outputs/crypto_trading/model.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(episodes=10, deterministic=True)
    print(f"Mean Reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    
    print("\nRunning sample episode with rendering...")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    for _ in range(min(50, len(price_data) - 1)):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
        
        if done:
            break
    
    print(f"\nFinal Portfolio Value: ${info['portfolio_value']:.2f}")
    print(f"Total Profit: ${info['total_profit']:.2f}")
    print(f"Return: {(info['total_profit'] / 10000.0) * 100:.2f}%")


if __name__ == "__main__":
    main()
