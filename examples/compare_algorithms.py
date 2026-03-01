from rl_llm_toolkit import RLEnvironment, PPOAgent, DQNAgent
from pathlib import Path
import numpy as np
import time


def train_and_evaluate(agent_class, env_name, timesteps, seed=42):
    print(f"\nTraining {agent_class.__name__} on {env_name}...")
    
    env = RLEnvironment(env_name)
    
    if agent_class == PPOAgent:
        agent = agent_class(
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            seed=seed,
        )
    else:
        agent = agent_class(
            env=env,
            learning_rate=1e-4,
            batch_size=32,
            seed=seed,
        )
    
    start_time = time.time()
    
    results = agent.train(
        total_timesteps=timesteps,
        log_interval=100,
        progress_bar=True,
    )
    
    training_time = time.time() - start_time
    
    eval_results = agent.evaluate(episodes=20, deterministic=True)
    
    env.close()
    
    return {
        "algorithm": agent_class.__name__,
        "training_time": training_time,
        "total_episodes": results["episodes"],
        "mean_reward": eval_results["mean_reward"],
        "std_reward": eval_results["std_reward"],
        "min_reward": eval_results["min_reward"],
        "max_reward": eval_results["max_reward"],
    }


def main():
    print("="*60)
    print("Algorithm Comparison Benchmark")
    print("="*60)
    
    env_name = "CartPole-v1"
    timesteps = 50000
    
    algorithms = [PPOAgent, DQNAgent]
    results = []
    
    for algo in algorithms:
        result = train_and_evaluate(algo, env_name, timesteps)
        results.append(result)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['algorithm']}:")
        print(f"  Training Time: {result['training_time']:.1f}s")
        print(f"  Episodes: {result['total_episodes']}")
        print(f"  Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Min/Max: {result['min_reward']:.2f} / {result['max_reward']:.2f}")
    
    best_algo = max(results, key=lambda x: x['mean_reward'])
    print(f"\n🏆 Best Algorithm: {best_algo['algorithm']} "
          f"(Mean Reward: {best_algo['mean_reward']:.2f})")
    
    fastest_algo = min(results, key=lambda x: x['training_time'])
    print(f"⚡ Fastest Training: {fastest_algo['algorithm']} "
          f"({fastest_algo['training_time']:.1f}s)")


if __name__ == "__main__":
    main()
