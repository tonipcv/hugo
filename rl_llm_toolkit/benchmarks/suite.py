from typing import Dict, List, Any, Optional, Callable
import time
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from rl_llm_toolkit.core.environment import RLEnvironment
from rl_llm_toolkit.agents.base import BaseAgent


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for RL algorithms.
    
    Features:
    - Standard environment benchmarks
    - Algorithm comparison
    - Performance profiling
    - Reproducibility tracking
    - Result visualization
    """
    
    STANDARD_ENVS = {
        "classic_control": [
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1",
        ],
        "box2d": [
            "LunarLander-v2",
            "BipedalWalker-v3",
        ],
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            output_dir = Path("./benchmark_results")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def run_benchmark(
        self,
        agent_factory: Callable,
        env_name: str,
        num_seeds: int = 5,
        total_timesteps: int = 100000,
        eval_episodes: int = 20,
        **agent_kwargs
    ) -> Dict[str, Any]:
        """
        Run benchmark for a specific agent-environment combination.
        
        Args:
            agent_factory: Function that creates agent given (env, seed, **kwargs)
            env_name: Environment name
            num_seeds: Number of random seeds to test
            total_timesteps: Training timesteps per seed
            eval_episodes: Evaluation episodes per seed
            **agent_kwargs: Additional arguments for agent creation
            
        Returns:
            Benchmark results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking on {env_name}")
        print(f"{'='*60}")
        
        seed_results = []
        
        for seed in tqdm(range(num_seeds), desc=f"Seeds for {env_name}"):
            env = RLEnvironment(env_name)
            
            start_time = time.time()
            
            agent = agent_factory(env=env, seed=seed, **agent_kwargs)
            
            train_results = agent.train(
                total_timesteps=total_timesteps,
                log_interval=10000,
                progress_bar=False,
            )
            
            training_time = time.time() - start_time
            
            eval_results = agent.evaluate(
                episodes=eval_episodes,
                deterministic=True,
            )
            
            stats = agent.get_training_stats()
            
            seed_results.append({
                "seed": seed,
                "training_time": training_time,
                "total_episodes": train_results.get("episodes", 0),
                "eval_mean_reward": eval_results["mean_reward"],
                "eval_std_reward": eval_results["std_reward"],
                "eval_min_reward": eval_results["min_reward"],
                "eval_max_reward": eval_results["max_reward"],
                "final_episode_rewards": stats["stats"]["episode_rewards"][-10:] if stats["stats"]["episode_rewards"] else [],
            })
            
            env.close()
        
        aggregated = self._aggregate_results(seed_results)
        
        result = {
            "env_name": env_name,
            "algorithm": agent_factory.__name__ if hasattr(agent_factory, '__name__') else str(agent_factory),
            "num_seeds": num_seeds,
            "total_timesteps": total_timesteps,
            "agent_kwargs": agent_kwargs,
            "seed_results": seed_results,
            "aggregated": aggregated,
        }
        
        self.results.append(result)
        
        self._print_summary(result)
        
        return result
    
    def _aggregate_results(self, seed_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate results across seeds."""
        eval_rewards = [r["eval_mean_reward"] for r in seed_results]
        training_times = [r["training_time"] for r in seed_results]
        
        return {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "min_reward": np.min(eval_rewards),
            "max_reward": np.max(eval_rewards),
            "median_reward": np.median(eval_rewards),
            "mean_training_time": np.mean(training_times),
            "std_training_time": np.std(training_times),
        }
    
    def _print_summary(self, result: Dict[str, Any]) -> None:
        """Print benchmark summary."""
        agg = result["aggregated"]
        
        print(f"\nResults for {result['env_name']}:")
        print(f"  Mean Reward: {agg['mean_reward']:.2f} ± {agg['std_reward']:.2f}")
        print(f"  Median Reward: {agg['median_reward']:.2f}")
        print(f"  Range: [{agg['min_reward']:.2f}, {agg['max_reward']:.2f}]")
        print(f"  Training Time: {agg['mean_training_time']:.1f}s ± {agg['std_training_time']:.1f}s")
    
    def compare_algorithms(
        self,
        agent_factories: Dict[str, Callable],
        env_names: List[str],
        num_seeds: int = 3,
        total_timesteps: int = 50000,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple algorithms across environments.
        
        Returns:
            Dictionary mapping (algorithm, env) to results
        """
        comparison = {}
        
        for env_name in env_names:
            for algo_name, agent_factory in agent_factories.items():
                print(f"\nTesting {algo_name} on {env_name}...")
                
                result = self.run_benchmark(
                    agent_factory=agent_factory,
                    env_name=env_name,
                    num_seeds=num_seeds,
                    total_timesteps=total_timesteps,
                )
                
                comparison[f"{algo_name}_{env_name}"] = result
        
        return comparison
    
    def export_results(
        self,
        filename: str = "benchmark_results.json",
        format: str = "json",
    ) -> Path:
        """Export benchmark results to file."""
        output_path = self.output_dir / filename
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
        
        elif format == "markdown":
            self._export_markdown(output_path)
        
        print(f"\n✅ Results exported to {output_path}")
        return output_path
    
    def _export_markdown(self, output_path: Path) -> None:
        """Export results as markdown table."""
        with open(output_path, "w") as f:
            f.write("# Benchmark Results\n\n")
            
            f.write("| Environment | Algorithm | Mean Reward | Std | Training Time |\n")
            f.write("|-------------|-----------|-------------|-----|---------------|\n")
            
            for result in self.results:
                agg = result["aggregated"]
                f.write(
                    f"| {result['env_name']} | {result['algorithm']} | "
                    f"{agg['mean_reward']:.2f} | {agg['std_reward']:.2f} | "
                    f"{agg['mean_training_time']:.1f}s |\n"
                )
    
    def profile_performance(
        self,
        agent: BaseAgent,
        env_name: str,
        num_steps: int = 10000,
    ) -> Dict[str, Any]:
        """
        Profile agent performance metrics.
        
        Returns:
            Performance metrics including FPS, memory usage, etc.
        """
        env = RLEnvironment(env_name)
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        
        obs, _ = env.reset(seed=42)
        for _ in range(num_steps):
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        elapsed_time = time.time() - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        fps = num_steps / elapsed_time
        memory_increase = final_memory - initial_memory
        
        env.close()
        
        return {
            "fps": fps,
            "steps_per_second": fps,
            "total_time": elapsed_time,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase,
        }
