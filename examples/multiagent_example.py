from rl_llm_toolkit.multiagent import MADDPGAgent, MultiAgentEnv
from rl_llm_toolkit.multiagent.environment import CooperativeNavigationEnv
from pathlib import Path
import numpy as np


def main():
    print("="*60)
    print("Multi-Agent RL Example: MADDPG")
    print("="*60)
    
    num_agents = 3
    env = CooperativeNavigationEnv(
        num_agents=num_agents,
        num_landmarks=3,
        world_size=2.0,
    )
    
    print(f"\nEnvironment: Cooperative Navigation")
    print(f"  Number of agents: {num_agents}")
    print(f"  Number of landmarks: 3")
    print(f"  Observation space: {env.agent_observation_space}")
    print(f"  Action space: {env.agent_action_space}")
    
    print("\nTraining MADDPG agents...")
    agent = MADDPGAgent(
        env=env,
        num_agents=num_agents,
        learning_rate_actor=1e-4,
        learning_rate_critic=1e-3,
        buffer_size=100000,
        batch_size=64,
        gamma=0.95,
        tau=0.01,
        noise_scale=0.1,
        seed=42,
    )
    
    results = agent.train(
        total_timesteps=100000,
        log_interval=5000,
        progress_bar=True,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total timesteps: {results['total_timesteps']}")
    print(f"Total episodes: {results['episodes']}")
    
    print("\nEvaluating trained agents...")
    observations, _ = env.reset(seed=42)
    episode_rewards = [0.0] * num_agents
    done = False
    steps = 0
    max_steps = 100
    
    while not done and steps < max_steps:
        actions = {}
        for i in range(num_agents):
            obs = observations[f"agent_{i}"]
            action, _ = agent._get_action(i, obs, add_noise=False)
            actions[f"agent_{i}"] = action
        
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        for i in range(num_agents):
            episode_rewards[i] += rewards[f"agent_{i}"]
        
        done = any(terminated.values()) or any(truncated.values())
        steps += 1
        
        if steps % 20 == 0:
            print(f"\nStep {steps}:")
            print(f"  Collisions: {info['collisions']}")
            print(f"  Coverage: {info['coverage']:.2%}")
            print(f"  Rewards: {[f'{r:.2f}' for r in episode_rewards]}")
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    for i in range(num_agents):
        print(f"Agent {i}: Total Reward = {episode_rewards[i]:.2f}")
    
    print(f"\nFinal Coverage: {info['coverage']:.2%}")
    print(f"Final Collisions: {info['collisions']}")
    
    model_path = Path("./models/multiagent/maddpg_navigation.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    print(f"\n✅ Model saved to {model_path}")


if __name__ == "__main__":
    main()
