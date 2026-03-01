from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path
from rl_llm_toolkit.core.environment import RLEnvironment
from rl_llm_toolkit.rewards.llm_shaper import LLMRewardShaper


class BaseAgent(ABC):
    def __init__(
        self,
        env: RLEnvironment,
        reward_shaper: Optional[LLMRewardShaper] = None,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.reward_shaper = reward_shaper
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        self._total_timesteps = 0
        self._episode_count = 0
        self._training_stats: Dict[str, list] = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
        }

    @abstractmethod
    def train(self, total_timesteps: int, **kwargs: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        pass

    def evaluate(
        self,
        episodes: int = 10,
        render: bool = False,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    self.env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
        }

    def get_training_stats(self) -> Dict[str, Any]:
        return {
            "total_timesteps": self._total_timesteps,
            "episode_count": self._episode_count,
            "stats": self._training_stats,
        }
