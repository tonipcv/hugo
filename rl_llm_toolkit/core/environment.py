from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RLEnvironment:
    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        **kwargs: Any
    ):
        self.env_id = env_id
        self.render_mode = render_mode
        self.env = gym.make(env_id, render_mode=render_mode, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._current_episode_reward = 0.0
        self._current_episode_length = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._current_episode_length > 0:
            self._episode_rewards.append(self._current_episode_reward)
            self._episode_lengths.append(self._current_episode_length)
        
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._current_episode_reward += reward
        self._current_episode_length += 1
        
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def render(self) -> Optional[np.ndarray]:
        return self.env.render()

    @property
    def episode_statistics(self) -> Dict[str, float]:
        if not self._episode_rewards:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_length": 0.0,
                "total_episodes": 0,
            }
        
        return {
            "mean_reward": np.mean(self._episode_rewards),
            "std_reward": np.std(self._episode_rewards),
            "mean_length": np.mean(self._episode_lengths),
            "total_episodes": len(self._episode_rewards),
        }

    def get_state_description(self, obs: np.ndarray) -> str:
        if isinstance(self.observation_space, spaces.Box):
            if len(obs.shape) == 1:
                return f"State vector with {len(obs)} dimensions: {obs.tolist()}"
            else:
                return f"State array with shape {obs.shape}"
        elif isinstance(self.observation_space, spaces.Discrete):
            return f"Discrete state: {obs}"
        else:
            return f"State: {obs}"

    def get_action_description(self, action: Union[int, np.ndarray]) -> str:
        if isinstance(self.action_space, spaces.Discrete):
            return f"Action {action} (discrete)"
        elif isinstance(self.action_space, spaces.Box):
            if isinstance(action, np.ndarray):
                return f"Action vector: {action.tolist()}"
            return f"Action: {action}"
        else:
            return f"Action: {action}"
