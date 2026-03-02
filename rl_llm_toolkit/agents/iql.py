from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import random
from tqdm import tqdm

from rl_llm_toolkit.agents.base import BaseAgent
from rl_llm_toolkit.core.environment import RLEnvironment
from rl_llm_toolkit.rewards.llm_shaper import LLMRewardShaper


class IQLNetwork(nn.Module):
    """Implicit Q-Learning network with separate V and Q functions."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        
        self.q_network = self._build_network(obs_dim + action_dim, 1, hidden_sizes)
        self.v_network = self._build_network(obs_dim, 1, hidden_sizes)
        self.policy_network = self._build_network(obs_dim, action_dim, hidden_sizes)
    
    def _build_network(self, input_dim: int, output_dim: int, hidden_sizes: Tuple[int, ...]) -> nn.Module:
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_dim))
        
        return nn.Sequential(*layers)
    
    def get_q_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value for state-action pair."""
        x = torch.cat([obs, action], dim=-1)
        return self.q_network(x)
    
    def get_v_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get V-value for state."""
        return self.v_network(obs)
    
    def get_action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Get policy logits for state."""
        return self.policy_network(obs)


class IQLAgent(BaseAgent):
    """
    Implicit Q-Learning (IQL) for Offline RL.
    
    IQL avoids explicit policy constraints by using expectile regression
    for value learning, making it more stable for offline learning.
    """
    
    def __init__(
        self,
        env: RLEnvironment,
        reward_shaper: Optional[LLMRewardShaper] = None,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        temperature: float = 3.0,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env, reward_shaper, seed)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = env.action_space.n
        
        self.network = IQLNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target_network = IQLNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        
        self.q_optimizer = optim.Adam(self.network.q_network.parameters(), lr=learning_rate)
        self.v_optimizer = optim.Adam(self.network.v_network.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.network.policy_network.parameters(), lr=learning_rate)
        
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
    ) -> list:
        """Collect dataset using specified policy."""
        dataset = []
        
        for ep in tqdm(range(num_episodes), desc="Collecting dataset"):
            obs, _ = self.env.reset(seed=self.seed + ep if self.seed else None)
            done = False
            
            while not done:
                if policy == "random":
                    action = self.env.action_space.sample()
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
        """Train IQL agent on offline dataset."""
        
        if not self.dataset:
            raise ValueError("No dataset loaded. Use load_dataset() or collect_dataset() first.")
        
        num_updates = total_timesteps // self.batch_size
        
        pbar = tqdm(total=num_updates, disable=not progress_bar, desc="Training IQL")
        
        for update in range(num_updates):
            loss_dict = self._update_networks()
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

    def _expectile_loss(self, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """Asymmetric squared loss for expectile regression."""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def _update_networks(self) -> Dict[str, float]:
        """Perform one IQL update step."""
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
        
        action_one_hot = torch.nn.functional.one_hot(action_tensor, self.env.action_space.n).float()
        
        with torch.no_grad():
            next_v = self.target_network.get_v_value(next_obs_tensor).squeeze(-1)
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * next_v
        
        current_q = self.network.get_q_value(obs_tensor, action_one_hot).squeeze(-1)
        q_loss = nn.functional.mse_loss(current_q, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        with torch.no_grad():
            q_value = self.network.get_q_value(obs_tensor, action_one_hot).squeeze(-1)
        
        v_value = self.network.get_v_value(obs_tensor).squeeze(-1)
        v_loss = self._expectile_loss(q_value - v_value, self.expectile).mean()
        
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        with torch.no_grad():
            v = self.network.get_v_value(obs_tensor).squeeze(-1)
            q = self.network.get_q_value(obs_tensor, action_one_hot).squeeze(-1)
            advantage = q - v
            exp_advantage = torch.exp(advantage / self.temperature)
            exp_advantage = torch.clamp(exp_advantage, max=100.0)
        
        logits = self.network.get_action_logits(obs_tensor)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        policy_loss = -(exp_advantage * action_log_probs).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        total_loss = q_loss + v_loss + policy_loss
        
        return {
            "q_loss": q_loss.item(),
            "v_loss": v_loss.item(),
            "policy_loss": policy_loss.item(),
            "total_loss": total_loss.item(),
        }

    def _soft_update_target(self) -> None:
        """Soft update of target network."""
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _log_training_progress(
        self, update: int, total_updates: int, loss_dict: Dict[str, float]
    ) -> None:
        tqdm.write(
            f"Update {update}/{total_updates} | "
            f"Q Loss: {loss_dict['q_loss']:.4f} | "
            f"V Loss: {loss_dict['v_loss']:.4f} | "
            f"Policy Loss: {loss_dict['policy_loss']:.4f}"
        )

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            logits = self.network.get_action_logits(obs_tensor)
            
            if deterministic:
                action = logits.argmax(dim=1).item()
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            
            info = {
                "logits": logits.cpu().numpy()[0],
                "v_value": self.network.get_v_value(obs_tensor).cpu().item(),
            }
            
            return np.array(action), info

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "v_optimizer_state_dict": self.v_optimizer.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "training_stats": self._training_stats,
            "total_timesteps": self._total_timesteps,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "tau": self.tau,
                "expectile": self.expectile,
                "temperature": self.temperature,
            }
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.v_optimizer.load_state_dict(checkpoint["v_optimizer_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self._training_stats = checkpoint["training_stats"]
        self._total_timesteps = checkpoint["total_timesteps"]
