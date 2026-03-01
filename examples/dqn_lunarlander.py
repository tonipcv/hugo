from rl_llm_toolkit import RLEnvironment, DQNAgent
from pathlib import Path


def main():
    print("Training DQN agent on LunarLander-v2")
    
    env = RLEnvironment("LunarLander-v2")
    
    agent = DQNAgent(
        env=env,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=32,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=50000,
        target_update_freq=1000,
        learning_starts=10000,
        seed=42,
    )
    
    results = agent.train(
        total_timesteps=200000,
        log_interval=5000,
        eval_interval=20000,
        eval_episodes=10,
    )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Total Episodes: {results['episodes']}")
    print(f"Total Timesteps: {results['total_timesteps']}")
    
    model_path = Path("./outputs/lunarlander_dqn/model.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(episodes=20, deterministic=True)
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Min/Max Reward: {eval_results['min_reward']:.2f} / {eval_results['max_reward']:.2f}")
    
    env.close()


if __name__ == "__main__":
    main()
