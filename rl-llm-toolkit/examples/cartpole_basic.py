from rl_llm_toolkit import RLEnvironment, PPOAgent
from pathlib import Path


def main():
    print("Training PPO agent on CartPole-v1")
    
    env = RLEnvironment("CartPole-v1")
    
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
        eval_interval=10000,
        eval_episodes=5,
    )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Total Episodes: {results['episodes']}")
    print(f"Total Timesteps: {results['total_timesteps']}")
    
    model_path = Path("./outputs/cartpole_basic/model.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(episodes=10, deterministic=True)
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Min/Max Reward: {eval_results['min_reward']:.2f} / {eval_results['max_reward']:.2f}")
    
    env.close()


if __name__ == "__main__":
    main()
