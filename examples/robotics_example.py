#!/usr/bin/env python3
"""
Robotics Example with RL-LLM Toolkit

Demonstrates:
- Custom robotics environments
- 2D robotic arm control
- Grid world navigation
- PPO training for continuous and discrete control
"""

import numpy as np
from pathlib import Path

from rl_llm_toolkit import PPOAgent
from rl_llm_toolkit.core.environment import RLEnvironment
from rl_llm_toolkit.environments.robotics import SimpleReacherEnv, GridWorldEnv


def train_reacher():
    """Train agent on robotic arm reaching task."""
    print("=" * 60)
    print("Training Robotic Arm Reaching")
    print("=" * 60)
    
    # Create environment
    env = SimpleReacherEnv(
        link_lengths=(1.0, 1.0),
        target_distance_threshold=0.1,
        max_steps=200
    )
    
    print(f"\nEnvironment: 2-link robotic arm")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Wrap in RLEnvironment
    wrapped_env = RLEnvironment(env)
    
    # Create agent
    agent = PPOAgent(
        env=wrapped_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        seed=42
    )
    
    # Train
    print("\nTraining agent...")
    results = agent.train(
        total_timesteps=100000,
        log_interval=10000,
        progress_bar=True
    )
    
    print(f"\nTraining completed!")
    print(f"Total timesteps: {results['total_timesteps']}")
    
    # Evaluate
    print("\nEvaluating agent...")
    eval_results = agent.evaluate(episodes=10, deterministic=True)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"Std Reward: {eval_results['std_reward']:.2f}")
    print(f"Success Rate: {eval_results.get('success_rate', 'N/A')}")
    
    # Save model
    model_dir = Path("models/robotics")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "reacher_ppo.pt"
    agent.save(model_path)
    print(f"\n✅ Model saved to {model_path}")
    
    # Demonstrate learned policy
    print("\nDemonstrating learned policy...")
    obs, _ = env.reset(seed=42)
    
    for step in range(50):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            print(f"Step {step}: Distance to target = {info['distance_to_target']:.3f}")
        
        if terminated or truncated:
            if info['target_reached']:
                print(f"✅ Target reached in {step+1} steps!")
            break
    
    env.close()
    wrapped_env.close()


def train_gridworld():
    """Train agent on grid world navigation."""
    print("\n" + "=" * 60)
    print("Training Grid World Navigation")
    print("=" * 60)
    
    # Create environment
    env = GridWorldEnv(
        grid_size=10,
        num_obstacles=15,
        max_steps=100
    )
    
    print(f"\nEnvironment: {env.grid_size}x{env.grid_size} grid world")
    print(f"Obstacles: {env.num_obstacles}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Wrap in RLEnvironment
    wrapped_env = RLEnvironment(env)
    
    # Create agent
    agent = PPOAgent(
        env=wrapped_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        seed=42
    )
    
    # Train
    print("\nTraining agent...")
    results = agent.train(
        total_timesteps=100000,
        log_interval=10000,
        progress_bar=True
    )
    
    print(f"\nTraining completed!")
    print(f"Total timesteps: {results['total_timesteps']}")
    
    # Evaluate
    print("\nEvaluating agent...")
    eval_results = agent.evaluate(episodes=20, deterministic=True)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"Std Reward: {eval_results['std_reward']:.2f}")
    
    # Calculate success rate
    success_count = 0
    for _ in range(20):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if info['goal_reached']:
                success_count += 1
                break
    
    success_rate = success_count / 20
    print(f"Success Rate: {success_rate*100:.1f}%")
    
    # Save model
    model_dir = Path("models/robotics")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "gridworld_ppo.pt"
    agent.save(model_path)
    print(f"\n✅ Model saved to {model_path}")
    
    # Demonstrate learned policy
    print("\nDemonstrating learned policy...")
    obs, _ = env.reset(seed=42)
    env.render()
    
    for step in range(50):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            env.render()
        
        if terminated or truncated:
            if info['goal_reached']:
                print(f"\n✅ Goal reached in {step+1} steps!")
                env.render()
            break
    
    env.close()
    wrapped_env.close()


def main():
    print("=" * 60)
    print("Robotics Environments Example")
    print("=" * 60)
    
    # Train on reacher task
    train_reacher()
    
    # Train on grid world
    train_gridworld()
    
    print("\n" + "=" * 60)
    print("Robotics Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
