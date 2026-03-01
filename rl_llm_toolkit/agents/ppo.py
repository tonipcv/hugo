from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from rl_llm_toolkit.agents.base import BaseAgent
from rl_llm_toolkit.agents.networks import ActorCriticNetwork
from rl_llm_toolkit.core.environment import RLEnvironment
from rl_llm_toolkit.rewards.llm_shaper import LLMRewardShaper


class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        
        self.observations = torch.zeros((buffer_size,) + obs_shape, dtype=torch.float32)
        self.actions = torch.zeros(buffer_size, dtype=torch.long)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32)
        
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        self.observations[self.pos] = torch.from_numpy(obs)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> None:
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values

    def get(self, batch_size: int):
        indices = np.random.permutation(self.buffer_size)
        
        for start_idx in range(0, self.buffer_size, batch_size):
            end_idx = min(start_idx + batch_size, self.buffer_size)
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                "observations": self.observations[batch_indices].to(self.device),
                "actions": self.actions[batch_indices].to(self.device),
                "old_log_probs": self.log_probs[batch_indices].to(self.device),
                "advantages": self.advantages[batch_indices].to(self.device),
                "returns": self.returns[batch_indices].to(self.device),
            }

    def reset(self) -> None:
        self.pos = 0
        self.full = False


class PPOAgent(BaseAgent):
    def __init__(
        self,
        env: RLEnvironment,
        reward_shaper: Optional[LLMRewardShaper] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env, reward_shaper, seed)
        
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.network = ActorCriticNetwork(
            env.observation_space,
            env.action_space,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        
        obs_shape = env.observation_space.shape
        self.rollout_buffer = RolloutBuffer(n_steps, obs_shape, self.device)

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 10,
        eval_interval: int = 5000,
        eval_episodes: int = 5,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        obs, _ = self.env.reset(seed=self.seed)
        episode_reward = 0.0
        episode_length = 0
        
        num_updates = total_timesteps // self.n_steps
        
        pbar = tqdm(total=total_timesteps, disable=not progress_bar, desc="Training PPO")
        
        for update in range(num_updates):
            for step in range(self.n_steps):
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()[0]
                    log_prob = log_prob.cpu().item()
                    value = value.cpu().item()
                
                next_obs, env_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                if self.reward_shaper is not None:
                    reward, shaping_info = self.reward_shaper.shape_reward(
                        env_reward, obs, action, next_obs,
                        context={"steps": episode_length}
                    )
                else:
                    reward = env_reward
                
                self.rollout_buffer.add(obs, action, reward, value, log_prob, done)
                
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
                
                pbar.update(1)
            
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                last_value = self.network.get_value(obs_tensor).cpu().item()
            
            self.rollout_buffer.compute_returns_and_advantages(
                last_value, self.gamma, self.gae_lambda
            )
            
            train_stats = self._update_policy()
            self._training_stats["losses"].append(train_stats["total_loss"])
            
            if (update + 1) % log_interval == 0:
                self._log_training_progress(update + 1, num_updates, train_stats)
            
            if eval_interval > 0 and (update + 1) % (eval_interval // self.n_steps) == 0:
                eval_stats = self.evaluate(episodes=eval_episodes, deterministic=True)
                tqdm.write(f"Evaluation: Mean Reward = {eval_stats['mean_reward']:.2f}")
        
        pbar.close()
        
        return {
            "total_timesteps": self._total_timesteps,
            "episodes": self._episode_count,
            "final_stats": self.get_training_stats(),
        }

    def _update_policy(self) -> Dict[str, float]:
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                _, log_prob, entropy, value = self.network.get_action_and_value(
                    batch["observations"], batch["actions"]
                )
                
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                ratio = torch.exp(log_prob - batch["old_log_probs"])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.functional.mse_loss(value, batch["returns"])
                
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())
        
        self.rollout_buffer.reset()
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "total_loss": np.mean(total_losses),
        }

    def _log_training_progress(
        self, update: int, total_updates: int, train_stats: Dict[str, float]
    ) -> None:
        recent_rewards = self._training_stats["episode_rewards"][-100:]
        if recent_rewards:
            mean_reward = np.mean(recent_rewards)
            tqdm.write(
                f"Update {update}/{total_updates} | "
                f"Episodes: {self._episode_count} | "
                f"Mean Reward (100 ep): {mean_reward:.2f} | "
                f"Policy Loss: {train_stats['policy_loss']:.4f} | "
                f"Value Loss: {train_stats['value_loss']:.4f}"
            )

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.network.get_action_and_value(
                obs_tensor, deterministic=deterministic
            )
            action = action.cpu().numpy()[0]
            
            info = {
                "log_prob": log_prob.cpu().item(),
                "value": value.cpu().item(),
            }
            
            return action, info

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self._training_stats,
            "total_timesteps": self._total_timesteps,
            "episode_count": self._episode_count,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "n_steps": self.n_steps,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_range": self.clip_range,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm,
            }
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._training_stats = checkpoint["training_stats"]
        self._total_timesteps = checkpoint["total_timesteps"]
        self._episode_count = checkpoint["episode_count"]
