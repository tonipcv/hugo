import pytest
import numpy as np
from pathlib import Path
import tempfile

from rl_llm_toolkit import RLEnvironment, PPOAgent
from rl_llm_toolkit.benchmarks import BenchmarkSuite, PerformanceMetrics


class TestBenchmarkSuite:
    def test_suite_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = BenchmarkSuite(output_dir=Path(tmpdir))
            assert suite.output_dir.exists()
    
    def test_run_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = BenchmarkSuite(output_dir=Path(tmpdir))
            
            def create_agent(env, seed):
                return PPOAgent(env=env, learning_rate=3e-4, seed=seed)
            
            result = suite.run_benchmark(
                agent_factory=create_agent,
                env_name="CartPole-v1",
                num_seeds=2,
                total_timesteps=5000,
                eval_episodes=3,
            )
            
            assert "env_name" in result
            assert "aggregated" in result
            assert "seed_results" in result
            assert len(result["seed_results"]) == 2
    
    def test_export_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = BenchmarkSuite(output_dir=Path(tmpdir))
            
            suite.results = [{
                "env_name": "CartPole-v1",
                "algorithm": "PPO",
                "aggregated": {
                    "mean_reward": 450.0,
                    "std_reward": 25.0,
                    "mean_training_time": 10.0,
                }
            }]
            
            json_path = suite.export_results("test.json", format="json")
            assert json_path.exists()
            
            md_path = suite.export_results("test.md", format="markdown")
            assert md_path.exists()


class TestPerformanceMetrics:
    def test_calculate_stability(self):
        rewards = [100, 105, 95, 110, 90, 100, 105]
        stability = PerformanceMetrics.calculate_stability(rewards, window=5)
        assert isinstance(stability, float)
        assert stability >= 0
    
    def test_calculate_asymptotic_performance(self):
        rewards = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        asymptotic = PerformanceMetrics.calculate_asymptotic_performance(rewards, window=3)
        assert asymptotic > 80
    
    def test_calculate_sample_efficiency(self):
        rewards = [10, 20, 30, 40, 50]
        timesteps = [1000, 2000, 3000, 4000, 5000]
        
        steps = PerformanceMetrics.calculate_sample_efficiency(
            rewards, threshold=35, timesteps=timesteps
        )
        assert steps == 4000
    
    def test_calculate_robustness(self):
        seed_results = [
            {"mean_reward": 450},
            {"mean_reward": 475},
            {"mean_reward": 460},
        ]
        
        robustness = PerformanceMetrics.calculate_robustness(
            seed_results, metric_key="mean_reward"
        )
        
        assert "mean" in robustness
        assert "std" in robustness
        assert "coefficient_of_variation" in robustness
        assert robustness["mean"] > 450
