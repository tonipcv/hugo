from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import deque
import random
from tqdm import tqdm

from rl_llm_toolkit.agents.base import BaseAgent
from rl_llm_toolkit.multiagent.environment import MultiAgentEnv


class ActorNetwork(nn.Module):
    """Actor network for MADDPG."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (128, 128)):
        super().__init__()
        
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class CriticNetwork(nn.Module):
    """Centralized critic network for MADDPG."""
    
    def __init__(
        self,
        total_obs_dim: int,
        total_action_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128)
    ):
        super().__init__()
        
        layers = []
        prev_size = total_obs_dim + total_action_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return self.network(x)


class MADDPGAgent(BaseAgent):
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG).
    
    Features:
    - Centralized training, decentralized execution
    - Each agent has its own actor-critic
    - Critic uses global information during training
    """
    
    def __init__(
        self,
        env: MultiAgentEnv,
        num_agents: int,
        learning_rate_actor: float = 1e-4,
        learning_rate_critic: float = 1e-3,
        buffer_size: int = 100000,
        batch_size: int = 64,
        gamma: float = 0.95,
        tau: float = 0.01,
        noise_scale: float = 0.1,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env, None, seed)
        
        self.num_agents = num_agents
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        obs_dim = int(np.prod(env.agent_observation_space.shape))
        action_dim = int(np.prod(env.agent_action_space.shape))
        
        total_obs_dim = obs_dim * num_agents
        total_action_dim = action_dim * num_agents
        
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for _ in range(num_agents):
            actor = ActorNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
            critic = CriticNetwork(total_obs_dim, total_action_dim, hidden_sizes).to(self.device)
            
            target_actor = ActorNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
            target_critic = CriticNetwork(total_obs_dim, total_action_dim, hidden_sizes).to(self.device)
            
            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=learning_rate_actor))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=learning_rate_critic))
        
        self.replay_buffer = deque(maxlen=buffer_size)
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 1000,
        eval_interval: int = 5000,
        eval_episodes: int = 10,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """Train MADDPG agents."""
        
        observations, _ = self.env.reset(seed=self.seed)
        episode_rewards = [0.0] * self.num_agents
        episode_length = 0
        
        pbar = tqdm(total=total_timesteps, disable=not progress_bar, desc="Training MADDPG")
        
        for timestep in range(total_timesteps):
            actions = {}
            
            for i in range(self.num_agents):
                obs = observations[f"agent_{i}"]
                action = self._get_action(i, obs, add_noise=True)
                actions[f"agent_{i}"] = action
            
            next_observations, rewards, terminated, truncated, info = self.env.step(actions)
            
            transition = {
                'observations': observations.copy(),
                'actions': actions.copy(),
                'rewards': rewards.copy(),
                'next_observations': next_observations.copy(),
                'terminated': terminated.copy(),
            }
            self.replay_buffer.append(transition)
            
            for i in range(self.num_agents):
                episode_rewards[i] += rewards[f"agent_{i}"]
            episode_length += 1
            
            observations = next_observations
            
            done = any(terminated.values()) or any(truncated.values())
            if done:
                for i in range(self.num_agents):
                    if f"agent_{i}_rewards" not in self._training_stats:
                        self._training_stats[f"agent_{i}_rewards"] = []
                    self._training_stats[f"agent_{i}_rewards"].append(episode_rewards[i])
                
                self._episode_count += 1
                observations, _ = self.env.reset()
                episode_rewards = [0.0] * self.num_agents
                episode_length = 0
            
            if len(self.replay_buffer) >= self.batch_size:
                loss_dict = self._update_networks()
                if "losses" not in self._training_stats:
                    self._training_stats["losses"] = []
                self._training_stats["losses"].append(loss_dict["total_loss"])
            
            self._total_timesteps += 1
            
            if (timestep + 1) % log_interval == 0:
                self._log_training_progress(timestep + 1, total_timesteps)
            
            pbar.update(1)
        
        pbar.close()
        
        return {
            "total_timesteps": self._total_timesteps,
            "episodes": self._episode_count,
            "final_stats": self.get_training_stats(),
        }

    def _get_action(self, agent_id: int, obs: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Get action for a specific agent."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action = self.actors[agent_id](obs_tensor).cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action

    def _update_networks(self) -> Dict[str, float]:
        """Update all agent networks."""
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        
        for agent_id in range(self.num_agents):
            obs_batch = []
            action_batch = []
            reward_batch = []
            next_obs_batch = []
            done_batch = []
            
            all_obs_batch = []
            all_actions_batch = []
            all_next_obs_batch = []
            
            for transition in batch:
                obs_batch.append(transition['observations'][f'agent_{agent_id}'])
                action_batch.append(transition['actions'][f'agent_{agent_id}'])
                reward_batch.append(transition['rewards'][f'agent_{agent_id}'])
                next_obs_batch.append(transition['next_observations'][f'agent_{agent_id}'])
                done_batch.append(transition['terminated'][f'agent_{agent_id}'])
                
                all_obs = np.concatenate([
                    transition['observations'][f'agent_{i}'] for i in range(self.num_agents)
                ])
                all_actions = np.concatenate([
                    transition['actions'][f'agent_{i}'] for i in range(self.num_agents)
                ])
                all_next_obs = np.concatenate([
                    transition['next_observations'][f'agent_{i}'] for i in range(self.num_agents)
                ])
                
                all_obs_batch.append(all_obs)
                all_actions_batch.append(all_actions)
                all_next_obs_batch.append(all_next_obs)
            
            obs_tensor = torch.from_numpy(np.array(obs_batch)).float().to(self.device)
            action_tensor = torch.from_numpy(np.array(action_batch)).float().to(self.device)
            reward_tensor = torch.from_numpy(np.array(reward_batch)).float().to(self.device)
            next_obs_tensor = torch.from_numpy(np.array(next_obs_batch)).float().to(self.device)
            done_tensor = torch.from_numpy(np.array(done_batch)).float().to(self.device)
            
            all_obs_tensor = torch.from_numpy(np.array(all_obs_batch)).float().to(self.device)
            all_actions_tensor = torch.from_numpy(np.array(all_actions_batch)).float().to(self.device)
            all_next_obs_tensor = torch.from_numpy(np.array(all_next_obs_batch)).float().to(self.device)
            
            with torch.no_grad():
                next_actions_list = []
                for i in range(self.num_agents):
                    next_obs_i = all_next_obs_tensor[:, i*obs_tensor.shape[1]:(i+1)*obs_tensor.shape[1]]
                    next_action_i = self.target_actors[i](next_obs_i)
                    next_actions_list.append(next_action_i)
                
                next_actions = torch.cat(next_actions_list, dim=-1)
                target_q = self.target_critics[agent_id](all_next_obs_tensor, next_actions)
                target_q = reward_tensor.unsqueeze(1) + (1 - done_tensor.unsqueeze(1)) * self.gamma * target_q
            
            current_q = self.critics[agent_id](all_obs_tensor, all_actions_tensor)
            critic_loss = nn.functional.mse_loss(current_q, target_q)
            
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_id].step()
            
            current_actions_list = []
            for i in range(self.num_agents):
                obs_i = all_obs_tensor[:, i*obs_tensor.shape[1]:(i+1)*obs_tensor.shape[1]]
                if i == agent_id:
                    action_i = self.actors[i](obs_i)
                else:
                    with torch.no_grad():
                        action_i = self.actors[i](obs_i)
                current_actions_list.append(action_i)
            
            current_actions = torch.cat(current_actions_list, dim=-1)
            actor_loss = -self.critics[agent_id](all_obs_tensor, current_actions).mean()
            
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_id].step()
            
            self._soft_update(self.actors[agent_id], self.target_actors[agent_id])
            self._soft_update(self.critics[agent_id], self.target_critics[agent_id])
            
            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
        
        return {
            "critic_loss": total_critic_loss / self.num_agents,
            "actor_loss": total_actor_loss / self.num_agents,
            "total_loss": (total_critic_loss + total_actor_loss) / self.num_agents,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update of target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _log_training_progress(self, timestep: int, total_timesteps: int) -> None:
        """Log training progress."""
        if self._episode_count > 0:
            recent_rewards = []
            for i in range(self.num_agents):
                key = f"agent_{i}_rewards"
                if key in self._training_stats and self._training_stats[key]:
                    recent = self._training_stats[key][-10:]
                    recent_rewards.append(np.mean(recent))
            
            if recent_rewards:
                tqdm.write(
                    f"Timestep {timestep}/{total_timesteps} | "
                    f"Episodes: {self._episode_count} | "
                    f"Avg Rewards: {[f'{r:.2f}' for r in recent_rewards]}"
                )

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action for single agent (agent 0 by default)."""
        action = self._get_action(0, observation, add_noise=not deterministic)
        return action, {}

    def save(self, path: Path) -> None:
        """Save all agent models."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "num_agents": self.num_agents,
            "training_stats": self._training_stats,
            "total_timesteps": self._total_timesteps,
            "episode_count": self._episode_count,
        }
        
        for i in range(self.num_agents):
            checkpoint[f"actor_{i}"] = self.actors[i].state_dict()
            checkpoint[f"critic_{i}"] = self.critics[i].state_dict()
            checkpoint[f"target_actor_{i}"] = self.target_actors[i].state_dict()
            checkpoint[f"target_critic_{i}"] = self.target_critics[i].state_dict()
        
        torch.save(checkpoint, path)

    def load(self, path: Path) -> None:
        """Load all agent models."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint[f"actor_{i}"])
            self.critics[i].load_state_dict(checkpoint[f"critic_{i}"])
            self.target_actors[i].load_state_dict(checkpoint[f"target_actor_{i}"])
            self.target_critics[i].load_state_dict(checkpoint[f"target_critic_{i}"])
        
        self._training_stats = checkpoint["training_stats"]
        self._total_timesteps = checkpoint["total_timesteps"]
        self._episode_count = checkpoint["episode_count"]
