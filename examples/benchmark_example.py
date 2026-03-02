from rl_llm_toolkit import RLEnvironment, PPOAgent, DQNAgent
from rl_llm_toolkit.agents.cql import CQLAgent
from rl_llm_toolkit.benchmarks import BenchmarkSuite, PerformanceMetrics
from pathlib import Path


def create_ppo_agent(env, seed, **kwargs):
    """Factory function for PPO agent."""
    return PPOAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        seed=seed,
        **kwargs
    )


def create_dqn_agent(env, seed, **kwargs):
    """Factory function for DQN agent."""
    return DQNAgent(
        env=env,
        learning_rate=1e-4,
        buffer_size=50000,
        batch_size=32,
        seed=seed,
        **kwargs
    )


def main():
    print("="*60)
    print("Comprehensive Benchmark Suite")
    print("="*60)
    
    suite = BenchmarkSuite(output_dir=Path("./benchmark_results"))
    
    print("\n1. Benchmarking PPO on CartPole-v1...")
    ppo_result = suite.run_benchmark(
        agent_factory=create_ppo_agent,
        env_name="CartPole-v1",
        num_seeds=3,
        total_timesteps=50000,
        eval_episodes=20,
    )
    
    print("\n2. Benchmarking DQN on CartPole-v1...")
    dqn_result = suite.run_benchmark(
        agent_factory=create_dqn_agent,
        env_name="CartPole-v1",
        num_seeds=3,
        total_timesteps=50000,
        eval_episodes=20,
    )
    
    print("\n3. Comparing algorithms...")
    comparison = suite.compare_algorithms(
        agent_factories={
            "PPO": create_ppo_agent,
            "DQN": create_dqn_agent,
        },
        env_names=["CartPole-v1"],
        num_seeds=2,
        total_timesteps=30000,
    )
    
    print("\n" + "="*60)
    print("Algorithm Comparison Summary")
    print("="*60)
    
    for key, result in comparison.items():
        agg = result["aggregated"]
        print(f"\n{key}:")
        print(f"  Mean Reward: {agg['mean_reward']:.2f} ± {agg['std_reward']:.2f}")
        print(f"  Training Time: {agg['mean_training_time']:.1f}s")
    
    print("\n4. Calculating performance metrics...")
    
    for result in suite.results:
        if result["env_name"] == "CartPole-v1":
            seed_results = result["seed_results"]
            
            rewards = [r["eval_mean_reward"] for r in seed_results]
            
            robustness = PerformanceMetrics.calculate_robustness(
                seed_results,
                metric_key="eval_mean_reward"
            )
            
            print(f"\n{result['algorithm']} Robustness Metrics:")
            print(f"  Mean: {robustness['mean']:.2f}")
            print(f"  Std: {robustness['std']:.2f}")
            print(f"  Range: {robustness['range']:.2f}")
            print(f"  CV: {robustness['coefficient_of_variation']:.3f}")
    
    print("\n5. Profiling performance...")
    env = RLEnvironment("CartPole-v1")
    agent = PPOAgent(env=env, seed=42)
    
    agent.train(total_timesteps=10000, progress_bar=False)
    
    profile = suite.profile_performance(
        agent=agent,
        env_name="CartPole-v1",
        num_steps=5000,
    )
    
    print("\nPerformance Profile:")
    print(f"  FPS: {profile['fps']:.1f}")
    print(f"  Memory: {profile['initial_memory_mb']:.1f} MB → {profile['final_memory_mb']:.1f} MB")
    print(f"  Increase: {profile['memory_increase_mb']:.1f} MB")
    
    env.close()
    
    print("\n6. Exporting results...")
    json_path = suite.export_results("benchmark_results.json", format="json")
    md_path = suite.export_results("benchmark_results.md", format="markdown")
    
    print(f"\n✅ Benchmark complete!")
    print(f"   JSON: {json_path}")
    print(f"   Markdown: {md_path}")


if __name__ == "__main__":
    main()
