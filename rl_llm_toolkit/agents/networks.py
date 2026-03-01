from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "tanh",
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        if isinstance(observation_space, spaces.Box):
            input_dim = int(np.prod(observation_space.shape))
        elif isinstance(observation_space, spaces.Discrete):
            input_dim = observation_space.n
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")
        
        if isinstance(action_space, spaces.Discrete):
            output_dim = action_space.n
            self.is_discrete = True
        elif isinstance(action_space, spaces.Box):
            output_dim = int(np.prod(action_space.shape))
            self.is_discrete = False
        else:
            raise ValueError(f"Unsupported action space: {action_space}")
        
        activation_fn = nn.Tanh if activation == "tanh" else nn.ReLU
        
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation_fn(),
            ])
            prev_size = hidden_size
        
        self.shared_net = nn.Sequential(*layers)
        
        self.policy_head = nn.Linear(prev_size, output_dim)
        self.value_head = nn.Linear(prev_size, 1)
        
        if not self.is_discrete:
            self.log_std = nn.Parameter(torch.zeros(output_dim))
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_net(obs)
        return self.policy_head(features), self.value_head(features)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        
        if self.is_discrete:
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            if action is None:
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = dist.sample()
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            mean = logits
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            
            if action is None:
                if deterministic:
                    action = mean
                else:
                    action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(obs)
        return value.squeeze(-1)


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        activation: str = "relu",
    ):
        super().__init__()
        
        if isinstance(observation_space, spaces.Box):
            input_dim = int(np.prod(observation_space.shape))
        elif isinstance(observation_space, spaces.Discrete):
            input_dim = observation_space.n
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")
        
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError("QNetwork only supports discrete action spaces")
        
        output_dim = action_space.n
        
        activation_fn = nn.ReLU if activation == "relu" else nn.Tanh
        
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation_fn(),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
