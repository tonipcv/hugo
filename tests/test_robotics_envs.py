"""Tests for robotics environments."""
import pytest
import numpy as np

from rl_llm_toolkit.environments.robotics import SimpleReacherEnv, GridWorldEnv


class TestSimpleReacherEnv:
    def test_env_creation(self):
        env = SimpleReacherEnv()
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (6,)
    
    def test_reset(self):
        env = SimpleReacherEnv()
        obs, info = env.reset(seed=42)
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (6,)
        assert env.current_step == 0
    
    def test_step(self):
        env = SimpleReacherEnv()
        obs, _ = env.reset(seed=42)
        
        action = np.array([0.5, -0.3])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "distance_to_target" in info
        assert "end_effector_position" in info
        assert "target_reached" in info
    
    def test_forward_kinematics(self):
        env = SimpleReacherEnv(link_lengths=(1.0, 1.0))
        env.reset(seed=42)
        
        # Set known joint angles
        env.joint_angles = np.array([0.0, 0.0])
        end_effector = env._get_end_effector_position()
        
        # Should be at (2.0, 0.0) when both angles are 0
        assert np.allclose(end_effector, [2.0, 0.0], atol=0.01)
    
    def test_target_reached(self):
        env = SimpleReacherEnv(target_distance_threshold=0.5)
        env.reset(seed=42)
        
        # Manually set end effector close to target
        env.joint_angles = np.array([0.0, 0.0])
        env.target_position = env._get_end_effector_position() + np.array([0.1, 0.1])
        
        action = np.array([0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should be close enough to target
        assert info["distance_to_target"] < 0.5
    
    def test_episode_truncation(self):
        env = SimpleReacherEnv(max_steps=10)
        env.reset(seed=42)
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
        assert truncated or terminated


class TestGridWorldEnv:
    def test_env_creation(self):
        env = GridWorldEnv(grid_size=10)
        assert env.action_space.n == 4
        assert env.observation_space.shape == (4 + 10 * 10,)
    
    def test_reset(self):
        env = GridWorldEnv(grid_size=10, num_obstacles=5)
        obs, info = env.reset(seed=42)
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4 + 10 * 10,)
        assert env.current_step == 0
        
        # Check that agent and goal are not on obstacles
        assert env.grid[tuple(env.agent_pos)] == 0
        assert env.grid[tuple(env.goal_pos)] == 0
    
    def test_step(self):
        env = GridWorldEnv(grid_size=10)
        obs, _ = env.reset(seed=42)
        
        action = 0  # up
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "distance_to_goal" in info
        assert "collision" in info
        assert "goal_reached" in info
    
    def test_movement(self):
        env = GridWorldEnv(grid_size=10, num_obstacles=0)
        env.reset(seed=42)
        
        # Place agent away from boundaries
        env.agent_pos = np.array([5, 5])
        initial_pos = env.agent_pos.copy()
        
        # Move up
        env.step(0)
        assert env.agent_pos[0] == initial_pos[0] - 1
        
        # Move down
        env.step(1)
        assert np.array_equal(env.agent_pos, initial_pos)
        
        # Move left
        env.step(2)
        assert env.agent_pos[1] == initial_pos[1] - 1
        
        # Move right
        env.step(3)
        assert np.array_equal(env.agent_pos, initial_pos)
    
    def test_boundary_collision(self):
        env = GridWorldEnv(grid_size=5, num_obstacles=0)
        env.reset(seed=42)
        
        # Move agent to corner
        env.agent_pos = np.array([0, 0])
        
        # Try to move up (out of bounds)
        obs, reward, terminated, truncated, info = env.step(0)
        
        # Agent should stay in place
        assert np.array_equal(env.agent_pos, [0, 0])
        assert info["collision"]
    
    def test_obstacle_collision(self):
        env = GridWorldEnv(grid_size=10, num_obstacles=5)
        env.reset(seed=42)
        
        # Find an obstacle
        obstacle_pos = np.argwhere(env.grid == 1)[0]
        
        # Place agent next to obstacle
        if obstacle_pos[0] > 0:
            env.agent_pos = obstacle_pos - np.array([1, 0])
            
            # Try to move into obstacle
            obs, reward, terminated, truncated, info = env.step(1)  # down
            
            # Agent should not move into obstacle
            assert not np.array_equal(env.agent_pos, obstacle_pos)
    
    def test_goal_reached(self):
        env = GridWorldEnv(grid_size=10, num_obstacles=0)
        env.reset(seed=42)
        
        # Place agent next to goal
        env.agent_pos = env.goal_pos - np.array([1, 0])
        
        # Move to goal
        obs, reward, terminated, truncated, info = env.step(1)  # down
        
        assert np.array_equal(env.agent_pos, env.goal_pos)
        assert info["goal_reached"]
        assert terminated
        assert reward > 50  # Should get large reward
    
    def test_episode_truncation(self):
        env = GridWorldEnv(grid_size=10, max_steps=10)
        env.reset(seed=42)
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
        assert truncated or terminated
    
    def test_observation_structure(self):
        env = GridWorldEnv(grid_size=5)
        obs, _ = env.reset(seed=42)
        
        # First 2 elements: agent position
        assert np.array_equal(obs[:2], env.agent_pos)
        
        # Next 2 elements: goal position
        assert np.array_equal(obs[2:4], env.goal_pos)
        
        # Remaining elements: flattened grid
        assert np.array_equal(obs[4:], env.grid.flatten())
