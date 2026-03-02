#!/usr/bin/env python3
"""
Advanced Visualization Example for RL-LLM Toolkit

Demonstrates:
- Training curve visualization
- Performance comparison plots
- Learning dynamics analysis
- Multi-agent visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rl_llm_toolkit import RLEnvironment, PPOAgent, DQNAgent
from rl_llm_toolkit.benchmarks import PerformanceMetrics


def plot_training_curves(agents_data, save_path=None):
    """Plot training curves for multiple agents."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    for agent_name, data in agents_data.items():
        rewards = data['episode_rewards']
        axes[0, 0].plot(rewards, label=agent_name, alpha=0.7)
        
        # Moving average
        window = min(20, len(rewards) // 10)
        if window > 0:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(rewards)), ma, 
                          linestyle='--', linewidth=2)
    
    axes[0, 0].set_title("Episode Rewards", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    for agent_name, data in agents_data.items():
        lengths = data['episode_lengths']
        axes[0, 1].plot(lengths, label=agent_name, alpha=0.7)
    
    axes[0, 1].set_title("Episode Lengths", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    for agent_name, data in agents_data.items():
        if 'learning_rates' in data:
            axes[1, 0].plot(data['learning_rates'], label=agent_name, alpha=0.7)
    
    axes[1, 0].set_title("Learning Rate Schedule", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Update Step")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Loss curves
    for agent_name, data in agents_data.items():
        if 'losses' in data:
            axes[1, 1].plot(data['losses'], label=agent_name, alpha=0.7)
    
    axes[1, 1].set_title("Training Loss", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Update Step")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_performance_comparison(results, save_path=None):
    """Plot performance comparison across algorithms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    algorithms = list(results.keys())
    mean_rewards = [results[alg]['mean_reward'] for alg in algorithms]
    std_rewards = [results[alg]['std_reward'] for alg in algorithms]
    
    # Bar plot with error bars
    x_pos = np.arange(len(algorithms))
    axes[0].bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(algorithms, rotation=45)
    axes[0].set_title("Mean Reward Comparison", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Mean Reward")
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Sample efficiency
    sample_efficiency = [results[alg].get('sample_efficiency', 0) for alg in algorithms]
    axes[1].bar(x_pos, sample_efficiency, alpha=0.7, color='green')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(algorithms, rotation=45)
    axes[1].set_title("Sample Efficiency", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Steps to Threshold")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Stability
    stability = [results[alg].get('stability', 0) for alg in algorithms]
    axes[2].bar(x_pos, stability, alpha=0.7, color='orange')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(algorithms, rotation=45)
    axes[2].set_title("Training Stability", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Stability Score")
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_learning_dynamics(timesteps, rewards, values, entropies, save_path=None):
    """Plot detailed learning dynamics."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Rewards over time
    axes[0].plot(timesteps, rewards, linewidth=2, color='blue')
    axes[0].fill_between(timesteps, 
                         rewards - np.std(rewards) * 0.5,
                         rewards + np.std(rewards) * 0.5,
                         alpha=0.3, color='blue')
    axes[0].set_title("Reward Progression", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)
    
    # Value estimates
    axes[1].plot(timesteps, values, linewidth=2, color='green')
    axes[1].set_title("Value Function Estimates", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.3)
    
    # Policy entropy
    axes[2].plot(timesteps, entropies, linewidth=2, color='red')
    axes[2].set_title("Policy Entropy (Exploration)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Entropy")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning dynamics saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    print("=" * 60)
    print("Advanced Visualization Example")
    print("=" * 60)
    
    # Train multiple agents
    print("\n1. Training multiple agents for comparison...")
    env = RLEnvironment("CartPole-v1")
    
    agents_data = {}
    
    # Train PPO
    print("\n   Training PPO...")
    ppo_agent = PPOAgent(env=env, learning_rate=3e-4, seed=42)
    ppo_results = ppo_agent.train(total_timesteps=20000, progress_bar=True)
    
    agents_data['PPO'] = {
        'episode_rewards': ppo_results['stats']['episode_rewards'],
        'episode_lengths': ppo_results['stats']['episode_lengths'],
        'losses': ppo_results['stats'].get('losses', []),
    }
    
    # Train DQN
    print("\n   Training DQN...")
    dqn_agent = DQNAgent(env=env, learning_rate=1e-4, seed=42)
    dqn_results = dqn_agent.train(total_timesteps=20000, progress_bar=True)
    
    agents_data['DQN'] = {
        'episode_rewards': dqn_results['stats']['episode_rewards'],
        'episode_lengths': dqn_results['stats']['episode_lengths'],
        'losses': dqn_results['stats'].get('losses', []),
    }
    
    # Plot training curves
    print("\n2. Generating training curve visualizations...")
    results_dir = Path("results/visualizations")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plot_training_curves(
        agents_data,
        save_path=results_dir / "training_curves.png"
    )
    
    # Evaluate agents
    print("\n3. Evaluating agents...")
    ppo_eval = ppo_agent.evaluate(episodes=20, deterministic=True)
    dqn_eval = dqn_agent.evaluate(episodes=20, deterministic=True)
    
    # Calculate performance metrics
    print("\n4. Calculating performance metrics...")
    
    ppo_rewards = agents_data['PPO']['episode_rewards']
    dqn_rewards = agents_data['DQN']['episode_rewards']
    
    ppo_stability = PerformanceMetrics.calculate_stability(ppo_rewards, window=10)
    dqn_stability = PerformanceMetrics.calculate_stability(dqn_rewards, window=10)
    
    ppo_asymptotic = PerformanceMetrics.calculate_asymptotic_performance(ppo_rewards, window=20)
    dqn_asymptotic = PerformanceMetrics.calculate_asymptotic_performance(dqn_rewards, window=20)
    
    comparison_results = {
        'PPO': {
            'mean_reward': ppo_eval['mean_reward'],
            'std_reward': ppo_eval['std_reward'],
            'stability': ppo_stability,
            'asymptotic_performance': ppo_asymptotic,
        },
        'DQN': {
            'mean_reward': dqn_eval['mean_reward'],
            'std_reward': dqn_eval['std_reward'],
            'stability': dqn_stability,
            'asymptotic_performance': dqn_asymptotic,
        }
    }
    
    # Plot performance comparison
    print("\n5. Generating performance comparison plots...")
    plot_performance_comparison(
        comparison_results,
        save_path=results_dir / "performance_comparison.png"
    )
    
    # Generate learning dynamics plot
    print("\n6. Generating learning dynamics visualization...")
    
    # Simulate learning dynamics data
    timesteps = np.linspace(0, 20000, len(ppo_rewards))
    values = np.cumsum(ppo_rewards) / (np.arange(len(ppo_rewards)) + 1)
    entropies = np.exp(-np.linspace(0, 3, len(ppo_rewards)))
    
    plot_learning_dynamics(
        timesteps,
        np.array(ppo_rewards),
        values,
        entropies,
        save_path=results_dir / "learning_dynamics.png"
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    
    for alg_name, metrics in comparison_results.items():
        print(f"\n{alg_name}:")
        print(f"  Mean Reward:     {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Stability:       {metrics['stability']:.4f}")
        print(f"  Asymptotic Perf: {metrics['asymptotic_performance']:.2f}")
    
    print("\n" + "=" * 60)
    print("Visualizations saved to:", results_dir)
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
