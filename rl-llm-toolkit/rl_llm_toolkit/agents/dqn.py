from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import deque
import random
from tqdm import tqdm

from rl_llm_toolkit.agents.base import BaseAgent
from rl_llm_toolkit.agents.networks import QNetwork
from rl_llm_toolkit.core.environment import RLEnvironment
from rl_llm_toolkit.rewards.llm_shaper import LLMRewardShaper


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        env: RLEnvironment,
        reward_shaper: Optional[LLMRewardShaper] = None,
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        batch_size: int = 32,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10000,
        target_update_freq: int = 1000,
        learning_starts: int = 1000,
        train_freq: int = 4,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env, reward_shaper, seed)
        
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.q_network = QNetwork(
            env.observation_space,
            env.action_space,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        
        self.target_network = QNetwork(
            env.observation_space,
            env.action_space,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        if seed is not None:
            random.seed(seed)

    def get_epsilon(self, timestep: int) -> float:
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * min(
            timestep / self.epsilon_decay, 1.0
        )
        return epsilon

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 1000,
        eval_interval: int = 5000,
        eval_episodes: int = 5,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        obs, _ = self.env.reset(seed=self.seed)
        episode_reward = 0.0
        episode_length = 0
        
        pbar = tqdm(total=total_timesteps, disable=not progress_bar, desc="Training DQN")
        
        for timestep in range(total_timesteps):
            epsilon = self.get_epsilon(timestep)
            
            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    q_values = self.q_network(obs_tensor)
                    action = q_values.argmax(dim=1).item()
            
            next_obs, env_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            if self.reward_shaper is not None:
                reward, shaping_info = self.reward_shaper.shape_reward(
                    env_reward, obs, action, next_obs,
                    context={"steps": episode_length}
                )
            else:
                reward = env_reward
            
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += env_reward
            episode_length += 1
            self._total_timesteps += 1
            
            if done:
                self._training_stats["episode_rewards"].append(episode_reward)
                self._training_stats["episode_lengths"].append(episode_length)
                self._episode_count += 1
                
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
            
            if (
                timestep >= self.learning_starts
                and timestep % self.train_freq == 0
                and len(self.replay_buffer) >= self.batch_size
            ):
                loss = self._update_q_network()
                self._training_stats["losses"].append(loss)
            
            if timestep % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            if (timestep + 1) % log_interval == 0:
                self._log_training_progress(timestep + 1, total_timesteps, epsilon)
            
            if eval_interval > 0 and (timestep + 1) % eval_interval == 0:
                eval_stats = self.evaluate(episodes=eval_episodes, deterministic=True)
                tqdm.write(f"Evaluation: Mean Reward = {eval_stats['mean_reward']:.2f}")
            
            pbar.update(1)
        
        pbar.close()
        
        return {
            "total_timesteps": self._total_timesteps,
            "episodes": self._episode_count,
            "final_stats": self.get_training_stats(),
        }

    def _update_q_network(self) -> float:
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = (
            self.replay_buffer.sample(self.batch_size)
        )
        
        obs_tensor = torch.from_numpy(obs_batch).float().to(self.device)
        action_tensor = torch.from_numpy(action_batch).long().to(self.device)
        reward_tensor = torch.from_numpy(reward_batch).to(self.device)
        next_obs_tensor = torch.from_numpy(next_obs_batch).float().to(self.device)
        done_tensor = torch.from_numpy(done_batch).to(self.device)
        
        current_q_values = self.q_network(obs_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_tensor).max(dim=1)[0]
            target_q_values = reward_tensor + (1 - done_tensor) * self.gamma * next_q_values
        
        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        return loss.item()

    def _log_training_progress(
        self, timestep: int, total_timesteps: int, epsilon: float
    ) -> None:
        recent_rewards = self._training_stats["episode_rewards"][-100:]
        if recent_rewards:
            mean_reward = np.mean(recent_rewards)
            recent_losses = self._training_stats["losses"][-100:]
            mean_loss = np.mean(recent_losses) if recent_losses else 0.0
            tqdm.write(
                f"Timestep {timestep}/{total_timesteps} | "
                f"Episodes: {self._episode_count} | "
                f"Mean Reward (100 ep): {mean_reward:.2f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Loss: {mean_loss:.4f}"
            )

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            
            if deterministic:
                action = q_values.argmax(dim=1).item()
            else:
                epsilon = self.epsilon_end
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = q_values.argmax(dim=1).item()
            
            info = {
                "q_values": q_values.cpu().numpy()[0],
                "max_q_value": q_values.max().item(),
            }
            
            return np.array(action), info

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self._training_stats,
            "total_timesteps": self._total_timesteps,
            "episode_count": self._episode_count,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "target_update_freq": self.target_update_freq,
            }
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._training_stats = checkpoint["training_stats"]
        self._total_timesteps = checkpoint["total_timesteps"]
        self._episode_count = checkpoint["episode_count"]
