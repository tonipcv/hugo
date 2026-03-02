from typing import List, Dict, Any
import numpy as np


class PerformanceMetrics:
    """
    Calculate various performance metrics for RL agents.
    """
    
    @staticmethod
    def calculate_sample_efficiency(
        rewards: List[float],
        threshold: float,
        timesteps: List[int],
    ) -> int:
        """
        Calculate sample efficiency: timesteps needed to reach threshold.
        
        Returns:
            Number of timesteps to reach threshold, or -1 if not reached
        """
        for i, reward in enumerate(rewards):
            if reward >= threshold:
                return timesteps[i] if i < len(timesteps) else i
        return -1
    
    @staticmethod
    def calculate_stability(rewards: List[float], window: int = 10) -> float:
        """
        Calculate training stability using coefficient of variation.
        
        Lower values indicate more stable training.
        """
        if len(rewards) < window:
            return 0.0
        
        recent_rewards = rewards[-window:]
        mean = np.mean(recent_rewards)
        std = np.std(recent_rewards)
        
        if mean == 0:
            return 0.0
        
        return std / abs(mean)
    
    @staticmethod
    def calculate_asymptotic_performance(
        rewards: List[float],
        window: int = 20,
    ) -> float:
        """
        Calculate asymptotic performance (final performance).
        """
        if len(rewards) < window:
            return np.mean(rewards) if rewards else 0.0
        
        return np.mean(rewards[-window:])
    
    @staticmethod
    def calculate_area_under_curve(
        rewards: List[float],
        normalize: bool = True,
    ) -> float:
        """
        Calculate area under the learning curve.
        
        Higher values indicate better overall performance during training.
        """
        auc = np.trapz(rewards)
        
        if normalize and len(rewards) > 0:
            auc = auc / len(rewards)
        
        return auc
    
    @staticmethod
    def calculate_regret(
        rewards: List[float],
        optimal_reward: float,
    ) -> float:
        """
        Calculate cumulative regret compared to optimal policy.
        """
        return sum(optimal_reward - r for r in rewards)
    
    @staticmethod
    def calculate_success_rate(
        episode_infos: List[Dict[str, Any]],
        success_key: str = "is_success",
    ) -> float:
        """
        Calculate success rate from episode info dictionaries.
        """
        if not episode_infos:
            return 0.0
        
        successes = sum(1 for info in episode_infos if info.get(success_key, False))
        return successes / len(episode_infos)
    
    @staticmethod
    def calculate_convergence_speed(
        rewards: List[float],
        threshold: float = 0.95,
    ) -> int:
        """
        Calculate convergence speed: episodes to reach threshold of max reward.
        """
        if not rewards:
            return -1
        
        max_reward = max(rewards)
        target = threshold * max_reward
        
        for i, reward in enumerate(rewards):
            if reward >= target:
                return i
        
        return -1
    
    @staticmethod
    def calculate_robustness(
        seed_results: List[Dict[str, float]],
        metric_key: str = "mean_reward",
    ) -> Dict[str, float]:
        """
        Calculate robustness metrics across multiple seeds.
        """
        values = [r[metric_key] for r in seed_results]
        
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "range": np.max(values) - np.min(values),
            "coefficient_of_variation": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
        }
