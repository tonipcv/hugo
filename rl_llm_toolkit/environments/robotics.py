"""Robotics environments for RL-LLM Toolkit."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class SimpleReacherEnv(gym.Env):
    """
    Simple 2D robotic arm reaching environment.
    
    The agent controls a 2-link robotic arm to reach a target position.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        link_lengths: Tuple[float, float] = (1.0, 1.0),
        target_distance_threshold: float = 0.1,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.link_lengths = np.array(link_lengths)
        self.target_threshold = target_distance_threshold
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Action space: joint velocities for 2 joints
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: [joint_angles (2), target_position (2), end_effector_position (2)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Random initial joint angles
        self.joint_angles = self.np_random.uniform(-np.pi, np.pi, size=2)
        
        # Random target position within reachable area
        max_reach = np.sum(self.link_lengths)
        angle = self.np_random.uniform(0, 2 * np.pi)
        distance = self.np_random.uniform(0.5, max_reach * 0.9)
        
        self.target_position = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle)
        ])
        
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def _get_end_effector_position(self) -> np.ndarray:
        """Calculate end effector position using forward kinematics."""
        x1 = self.link_lengths[0] * np.cos(self.joint_angles[0])
        y1 = self.link_lengths[0] * np.sin(self.joint_angles[0])
        
        x2 = x1 + self.link_lengths[1] * np.cos(self.joint_angles[0] + self.joint_angles[1])
        y2 = y1 + self.link_lengths[1] * np.sin(self.joint_angles[0] + self.joint_angles[1])
        
        return np.array([x2, y2])
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        end_effector_pos = self._get_end_effector_position()
        
        obs = np.concatenate([
            self.joint_angles,
            self.target_position,
            end_effector_pos
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Apply action (joint velocities)
        action = np.clip(action, -1.0, 1.0)
        self.joint_angles += action * 0.1  # Scale velocities
        
        # Normalize angles to [-pi, pi]
        self.joint_angles = np.arctan2(np.sin(self.joint_angles), np.cos(self.joint_angles))
        
        # Calculate distance to target
        end_effector_pos = self._get_end_effector_position()
        distance = np.linalg.norm(end_effector_pos - self.target_position)
        
        # Reward: negative distance + bonus for reaching target
        reward = -distance
        
        if distance < self.target_threshold:
            reward += 10.0  # Bonus for reaching target
            terminated = True
        else:
            terminated = False
        
        # Penalty for large actions (energy efficiency)
        reward -= 0.01 * np.sum(np.square(action))
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        info = {
            "distance_to_target": distance,
            "end_effector_position": end_effector_pos,
            "target_reached": distance < self.target_threshold
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Joint angles: {self.joint_angles}")
            print(f"End effector: {self._get_end_effector_position()}")
            print(f"Target: {self.target_position}")
            print(f"Distance: {np.linalg.norm(self._get_end_effector_position() - self.target_position):.3f}")


class GridWorldEnv(gym.Env):
    """
    Simple grid world navigation environment.
    
    The agent navigates a grid to reach a goal while avoiding obstacles.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        grid_size: int = 10,
        num_obstacles: int = 10,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Action space: 4 directions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: agent position (2) + goal position (2) + grid (flattened)
        self.observation_space = spaces.Box(
            low=0,
            high=max(grid_size, 2),  # 0=empty, 1=obstacle, 2=goal
            shape=(4 + grid_size * grid_size,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place obstacles
        obstacle_positions = set()
        while len(obstacle_positions) < self.num_obstacles:
            pos = (
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size)
            )
            obstacle_positions.add(pos)
        
        for pos in obstacle_positions:
            self.grid[pos] = 1
        
        # Place agent (avoid obstacles)
        while True:
            self.agent_pos = np.array([
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size)
            ])
            if self.grid[tuple(self.agent_pos)] == 0:
                break
        
        # Place goal (avoid obstacles and agent)
        while True:
            self.goal_pos = np.array([
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size)
            ])
            if (self.grid[tuple(self.goal_pos)] == 0 and 
                not np.array_equal(self.goal_pos, self.agent_pos)):
                break
        
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.concatenate([
            self.agent_pos.astype(np.float32),
            self.goal_pos.astype(np.float32),
            self.grid.flatten().astype(np.float32)
        ])
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        moves = {
            0: np.array([-1, 0]),  # up
            1: np.array([1, 0]),   # down
            2: np.array([0, -1]),  # left
            3: np.array([0, 1])    # right
        }
        
        new_pos = self.agent_pos + moves[action]
        
        # Check boundaries
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            
            # Check obstacles
            if self.grid[tuple(new_pos)] == 0:
                self.agent_pos = new_pos
                collision = False
            else:
                collision = True
        else:
            collision = True
        
        # Calculate reward
        distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 100.0
            terminated = True
        else:
            reward = -0.1  # Small step penalty
            if collision:
                reward -= 1.0  # Collision penalty
            terminated = False
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        info = {
            "distance_to_goal": distance,
            "collision": collision,
            "goal_reached": np.array_equal(self.agent_pos, self.goal_pos)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            grid_display = self.grid.copy()
            grid_display[tuple(self.agent_pos)] = 2  # Agent
            grid_display[tuple(self.goal_pos)] = 3   # Goal
            
            print(f"\nStep: {self.current_step}")
            print("Grid (0=empty, 1=obstacle, 2=agent, 3=goal):")
            print(grid_display)
            print(f"Distance to goal: {np.linalg.norm(self.agent_pos - self.goal_pos):.2f}")
