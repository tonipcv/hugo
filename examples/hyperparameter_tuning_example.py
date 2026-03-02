#!/usr/bin/env python3
"""
Hyperparameter Tuning Example with RL-LLM Toolkit

Demonstrates:
- Grid search for hyperparameters
- Random search optimization
- Performance comparison across configurations
- Best hyperparameter selection
"""

import numpy as np
from pathlib import Path
from itertools import product
import json

from rl_llm_toolkit import RLEnvironment, PPOAgent, DQNAgent
from rl_llm_toolkit.benchmarks import PerformanceMetrics


def grid_search_ppo(env_name="CartPole-v1", total_timesteps=20000):
    """Perform grid search for PPO hyperparameters."""
    print("=" * 60)
    print("Grid Search for PPO Hyperparameters")
    print("=" * 60)
    
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'n_steps': [512, 1024, 2048],
        'batch_size': [32, 64],
        'gamma': [0.95, 0.99],
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    print(f"\nTesting {len(combinations)} hyperparameter combinations...")
    print(f"Environment: {env_name}")
    print(f"Training timesteps: {total_timesteps}")
    
    results = []
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        
        print(f"\n[{i}/{len(combinations)}] Testing: {params}")
        
        # Create environment and agent
        env = RLEnvironment(env_name)
        agent = PPOAgent(
            env=env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            gamma=params['gamma'],
            seed=42
        )
        
        # Train
        train_results = agent.train(
            total_timesteps=total_timesteps,
            progress_bar=False
        )
        
        # Evaluate
        eval_results = agent.evaluate(episodes=10, deterministic=True)
        
        # Calculate metrics
        episode_rewards = train_results['stats']['episode_rewards']
        stability = PerformanceMetrics.calculate_stability(episode_rewards, window=10)
        
        result = {
            'params': params,
            'mean_reward': eval_results['mean_reward'],
            'std_reward': eval_results['std_reward'],
            'stability': stability,
            'training_episodes': len(episode_rewards)
        }
        
        results.append(result)
        
        print(f"   Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"   Stability: {result['stability']:.4f}")
        
        env.close()
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['mean_reward'])
    
    print("\n" + "=" * 60)
    print("Best Hyperparameters Found")
    print("=" * 60)
    print(f"Parameters: {best_result['params']}")
    print(f"Mean Reward: {best_result['mean_reward']:.2f} ± {best_result['std_reward']:.2f}")
    print(f"Stability: {best_result['stability']:.4f}")
    
    return results, best_result


def random_search_dqn(env_name="CartPole-v1", n_trials=10, total_timesteps=20000):
    """Perform random search for DQN hyperparameters."""
    print("\n" + "=" * 60)
    print("Random Search for DQN Hyperparameters")
    print("=" * 60)
    
    print(f"\nTesting {n_trials} random configurations...")
    print(f"Environment: {env_name}")
    print(f"Training timesteps: {total_timesteps}")
    
    results = []
    
    for trial in range(1, n_trials + 1):
        # Sample random hyperparameters
        params = {
            'learning_rate': np.random.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
            'buffer_size': np.random.choice([10000, 50000, 100000]),
            'batch_size': np.random.choice([32, 64, 128]),
            'gamma': np.random.uniform(0.95, 0.995),
            'epsilon_decay': np.random.choice([500, 1000, 2000]),
        }
        
        print(f"\n[{trial}/{n_trials}] Testing: {params}")
        
        # Create environment and agent
        env = RLEnvironment(env_name)
        agent = DQNAgent(
            env=env,
            learning_rate=params['learning_rate'],
            buffer_size=params['buffer_size'],
            batch_size=params['batch_size'],
            gamma=params['gamma'],
            epsilon_decay=params['epsilon_decay'],
            seed=42
        )
        
        # Train
        train_results = agent.train(
            total_timesteps=total_timesteps,
            progress_bar=False
        )
        
        # Evaluate
        eval_results = agent.evaluate(episodes=10, deterministic=True)
        
        # Calculate metrics
        episode_rewards = train_results['stats']['episode_rewards']
        stability = PerformanceMetrics.calculate_stability(episode_rewards, window=10)
        
        result = {
            'params': params,
            'mean_reward': eval_results['mean_reward'],
            'std_reward': eval_results['std_reward'],
            'stability': stability,
            'training_episodes': len(episode_rewards)
        }
        
        results.append(result)
        
        print(f"   Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"   Stability: {result['stability']:.4f}")
        
        env.close()
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['mean_reward'])
    
    print("\n" + "=" * 60)
    print("Best Hyperparameters Found")
    print("=" * 60)
    print(f"Parameters: {best_result['params']}")
    print(f"Mean Reward: {best_result['mean_reward']:.2f} ± {best_result['std_reward']:.2f}")
    print(f"Stability: {best_result['stability']:.4f}")
    
    return results, best_result


def compare_configurations(results, algorithm_name):
    """Compare and visualize different configurations."""
    print("\n" + "=" * 60)
    print(f"{algorithm_name} Configuration Comparison")
    print("=" * 60)
    
    # Sort by mean reward
    sorted_results = sorted(results, key=lambda x: x['mean_reward'], reverse=True)
    
    print("\nTop 5 Configurations:")
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"   Stability: {result['stability']:.4f}")
        print(f"   Params: {result['params']}")
    
    # Calculate statistics
    mean_rewards = [r['mean_reward'] for r in results]
    print(f"\nOverall Statistics:")
    print(f"  Best:    {np.max(mean_rewards):.2f}")
    print(f"  Worst:   {np.min(mean_rewards):.2f}")
    print(f"  Mean:    {np.mean(mean_rewards):.2f}")
    print(f"  Std:     {np.std(mean_rewards):.2f}")


def save_results(results, best_result, filename):
    """Save tuning results to file."""
    output = {
        'all_results': results,
        'best_configuration': best_result
    }
    
    results_dir = Path("results/hyperparameter_tuning")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = results_dir / filename
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    output = convert_types(output)
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to {filepath}")


def main():
    print("=" * 60)
    print("Hyperparameter Tuning Example")
    print("=" * 60)
    
    # Grid search for PPO
    ppo_results, ppo_best = grid_search_ppo(
        env_name="CartPole-v1",
        total_timesteps=20000
    )
    
    compare_configurations(ppo_results, "PPO")
    save_results(ppo_results, ppo_best, "ppo_grid_search.json")
    
    # Random search for DQN
    dqn_results, dqn_best = random_search_dqn(
        env_name="CartPole-v1",
        n_trials=10,
        total_timesteps=20000
    )
    
    compare_configurations(dqn_results, "DQN")
    save_results(dqn_results, dqn_best, "dqn_random_search.json")
    
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Complete!")
    print("=" * 60)
    
    print("\nKey Findings:")
    print(f"  PPO Best: {ppo_best['mean_reward']:.2f} with {ppo_best['params']}")
    print(f"  DQN Best: {dqn_best['mean_reward']:.2f} with {dqn_best['params']}")


if __name__ == "__main__":
    main()
