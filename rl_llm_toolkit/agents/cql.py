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


class CQLAgent(BaseAgent):
    """
    Conservative Q-Learning (CQL) for Offline RL.
    
    CQL learns from a fixed dataset without environment interaction,
    using conservative Q-value estimates to avoid overestimation.
    """
    
    def __init__(
        self,
        env: RLEnvironment,
        reward_shaper: Optional[LLMRewardShaper] = None,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 1.0,
        cql_weight: float = 1.0,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env, reward_shaper, seed)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.cql_weight = cql_weight
        
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
        
        self.dataset = []
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def load_dataset(self, dataset: list) -> None:
        """Load offline dataset for training."""
        self.dataset = dataset
        print(f"Loaded dataset with {len(dataset)} transitions")

    def collect_dataset(
        self,
        num_episodes: int = 100,
        policy: str = "random",
        epsilon: float = 0.3,
    ) -> list:
        """Collect dataset using specified policy."""
        dataset = []
        
        for ep in tqdm(range(num_episodes), desc="Collecting dataset"):
            obs, _ = self.env.reset(seed=self.seed + ep if self.seed else None)
            done = False
            
            while not done:
                if policy == "random":
                    action = self.env.action_space.sample()
                elif policy == "epsilon_greedy":
                    if random.random() < epsilon:
                        action = self.env.action_space.sample()
                    else:
                        with torch.no_grad():
                            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                            q_values = self.q_network(obs_tensor)
                            action = q_values.argmax(dim=1).item()
                else:
                    raise ValueError(f"Unknown policy: {policy}")
                
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                dataset.append({
                    'obs': obs.copy(),
                    'action': action,
                    'reward': reward,
                    'next_obs': next_obs.copy(),
                    'done': done,
                })
                
                obs = next_obs
        
        self.dataset = dataset
        return dataset

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 1000,
        eval_interval: int = 5000,
        eval_episodes: int = 10,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """Train CQL agent on offline dataset."""
        
        if not self.dataset:
            raise ValueError("No dataset loaded. Use load_dataset() or collect_dataset() first.")
        
        num_updates = total_timesteps // self.batch_size
        
        pbar = tqdm(total=num_updates, disable=not progress_bar, desc="Training CQL")
        
        for update in range(num_updates):
            loss_dict = self._update_q_network()
            self._training_stats["losses"].append(loss_dict["total_loss"])
            
            self._soft_update_target()
            
            self._total_timesteps += self.batch_size
            
            if (update + 1) % log_interval == 0:
                self._log_training_progress(update + 1, num_updates, loss_dict)
            
            if eval_interval > 0 and (update + 1) % (eval_interval // self.batch_size) == 0:
                eval_stats = self.evaluate(episodes=eval_episodes, deterministic=True)
                tqdm.write(f"Evaluation: Mean Reward = {eval_stats['mean_reward']:.2f}")
            
            pbar.update(1)
        
        pbar.close()
        
        return {
            "total_timesteps": self._total_timesteps,
            "updates": num_updates,
            "final_stats": self.get_training_stats(),
        }

    def _update_q_network(self) -> Dict[str, float]:
        """Perform one CQL update step."""
        batch = random.sample(self.dataset, self.batch_size)
        
        obs_batch = np.array([t['obs'] for t in batch])
        action_batch = np.array([t['action'] for t in batch])
        reward_batch = np.array([t['reward'] for t in batch], dtype=np.float32)
        next_obs_batch = np.array([t['next_obs'] for t in batch])
        done_batch = np.array([t['done'] for t in batch], dtype=np.float32)
        
        obs_tensor = torch.from_numpy(obs_batch).float().to(self.device)
        action_tensor = torch.from_numpy(action_batch).long().to(self.device)
        reward_tensor = torch.from_numpy(reward_batch).to(self.device)
        next_obs_tensor = torch.from_numpy(next_obs_batch).float().to(self.device)
        done_tensor = torch.from_numpy(done_batch).to(self.device)
        
        current_q_values = self.q_network(obs_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_tensor).max(dim=1)[0]
            target_q_values = reward_tensor + (1 - done_tensor) * self.gamma * next_q_values
        
        bellman_loss = nn.functional.mse_loss(current_q_values, target_q_values)
        
        all_q_values = self.q_network(obs_tensor)
        logsumexp = torch.logsumexp(all_q_values, dim=1)
        cql_loss = (logsumexp - current_q_values).mean()
        
        total_loss = bellman_loss + self.cql_weight * cql_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        return {
            "bellman_loss": bellman_loss.item(),
            "cql_loss": cql_loss.item(),
            "total_loss": total_loss.item(),
        }

    def _soft_update_target(self) -> None:
        """Soft update of target network."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _log_training_progress(
        self, update: int, total_updates: int, loss_dict: Dict[str, float]
    ) -> None:
        tqdm.write(
            f"Update {update}/{total_updates} | "
            f"Bellman Loss: {loss_dict['bellman_loss']:.4f} | "
            f"CQL Loss: {loss_dict['cql_loss']:.4f} | "
            f"Total Loss: {loss_dict['total_loss']:.4f}"
        )

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
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
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": self.alpha,
                "cql_weight": self.cql_weight,
            }
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._training_stats = checkpoint["training_stats"]
        self._total_timesteps = checkpoint["total_timesteps"]
