from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiAgentEnv(gym.Env):
    """
    Base class for multi-agent environments.
    
    Supports:
    - Multiple agents with independent or shared observations
    - Cooperative, competitive, or mixed scenarios
    - Communication between agents
    """
    
    def __init__(
        self,
        num_agents: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        scenario: str = "cooperative",
    ):
        super().__init__()
        
        self.num_agents = num_agents
        self.agent_observation_space = observation_space
        self.agent_action_space = action_space
        self.scenario = scenario
        
        self.observation_space = spaces.Dict({
            f"agent_{i}": observation_space for i in range(num_agents)
        })
        
        self.action_space = spaces.Dict({
            f"agent_{i}": action_space for i in range(num_agents)
        })
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and return initial observations for all agents."""
        raise NotImplementedError
    
    def step(
        self,
        actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any]
    ]:
        """
        Execute actions for all agents.
        
        Returns:
            observations: Dict mapping agent_id to observation
            rewards: Dict mapping agent_id to reward
            terminated: Dict mapping agent_id to terminated flag
            truncated: Dict mapping agent_id to truncated flag
            info: Dict with additional information
        """
        raise NotImplementedError


class CooperativeNavigationEnv(MultiAgentEnv):
    """
    Cooperative navigation task where agents must reach goals while avoiding collisions.
    """
    
    def __init__(
        self,
        num_agents: int = 3,
        num_landmarks: int = 3,
        world_size: float = 2.0,
        agent_size: float = 0.05,
        landmark_size: float = 0.1,
    ):
        obs_dim = 2 + 2 + (num_agents - 1) * 2 + num_landmarks * 2
        
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        super().__init__(num_agents, observation_space, action_space, "cooperative")
        
        self.num_landmarks = num_landmarks
        self.world_size = world_size
        self.agent_size = agent_size
        self.landmark_size = landmark_size
        
        self.agent_positions = np.zeros((num_agents, 2))
        self.agent_velocities = np.zeros((num_agents, 2))
        self.landmark_positions = np.zeros((num_landmarks, 2))
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        
        self.agent_positions = np.random.uniform(
            -self.world_size, self.world_size, (self.num_agents, 2)
        )
        self.agent_velocities = np.zeros((self.num_agents, 2))
        
        self.landmark_positions = np.random.uniform(
            -self.world_size, self.world_size, (self.num_landmarks, 2)
        )
        
        observations = {}
        for i in range(self.num_agents):
            observations[f"agent_{i}"] = self._get_observation(i)
        
        return observations, {}
    
    def step(
        self,
        actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any]
    ]:
        for i in range(self.num_agents):
            action = actions[f"agent_{i}"]
            
            self.agent_velocities[i] = action * 0.1
            self.agent_positions[i] += self.agent_velocities[i]
            
            self.agent_positions[i] = np.clip(
                self.agent_positions[i],
                -self.world_size,
                self.world_size
            )
        
        observations = {}
        rewards = {}
        terminated = {}
        truncated = {}
        
        for i in range(self.num_agents):
            observations[f"agent_{i}"] = self._get_observation(i)
            
            reward = self._compute_reward(i)
            rewards[f"agent_{i}"] = reward
            
            terminated[f"agent_{i}"] = False
            truncated[f"agent_{i}"] = False
        
        info = {
            "collisions": self._count_collisions(),
            "coverage": self._compute_coverage(),
        }
        
        return observations, rewards, terminated, truncated, info
    
    def _get_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a specific agent."""
        obs = []
        
        obs.extend(self.agent_positions[agent_id])
        obs.extend(self.agent_velocities[agent_id])
        
        for j in range(self.num_agents):
            if j != agent_id:
                relative_pos = self.agent_positions[j] - self.agent_positions[agent_id]
                obs.extend(relative_pos)
        
        for landmark_pos in self.landmark_positions:
            relative_pos = landmark_pos - self.agent_positions[agent_id]
            obs.extend(relative_pos)
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self, agent_id: int) -> float:
        """Compute reward for cooperative task."""
        reward = 0.0
        
        min_dist = float('inf')
        for landmark_pos in self.landmark_positions:
            dist = np.linalg.norm(self.agent_positions[agent_id] - landmark_pos)
            min_dist = min(min_dist, dist)
        
        reward -= min_dist
        
        for j in range(self.num_agents):
            if j != agent_id:
                dist = np.linalg.norm(self.agent_positions[agent_id] - self.agent_positions[j])
                if dist < 2 * self.agent_size:
                    reward -= 1.0
        
        return reward
    
    def _count_collisions(self) -> int:
        """Count number of agent collisions."""
        collisions = 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(self.agent_positions[i] - self.agent_positions[j])
                if dist < 2 * self.agent_size:
                    collisions += 1
        return collisions
    
    def _compute_coverage(self) -> float:
        """Compute how well landmarks are covered by agents."""
        total_coverage = 0.0
        
        for landmark_pos in self.landmark_positions:
            min_dist = float('inf')
            for agent_pos in self.agent_positions:
                dist = np.linalg.norm(agent_pos - landmark_pos)
                min_dist = min(min_dist, dist)
            
            coverage = max(0, 1.0 - min_dist / self.world_size)
            total_coverage += coverage
        
        return total_coverage / self.num_landmarks
    
    def render(self) -> None:
        """Render the environment (placeholder)."""
        print(f"Agents: {self.agent_positions}")
        print(f"Landmarks: {self.landmark_positions}")
        print(f"Collisions: {self._count_collisions()}")
        print(f"Coverage: {self._compute_coverage():.2f}")
