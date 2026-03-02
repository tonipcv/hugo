from rl_llm_toolkit import RLEnvironment, PPOAgent, DQNAgent
from rl_llm_toolkit.collaboration import CollaborationSession, SharedReplayBuffer
from pathlib import Path
import time


def train_collaborative_agent(
    agent_id: str,
    agent_type: str,
    session: CollaborationSession,
    shared_buffer: SharedReplayBuffer,
    timesteps: int = 20000,
):
    """Train agent with collaborative experience sharing."""
    print(f"\n{'='*60}")
    print(f"Training {agent_id} ({agent_type})")
    print(f"{'='*60}")
    
    env = RLEnvironment("CartPole-v1")
    
    if agent_type == "PPO":
        agent = PPOAgent(env=env, learning_rate=3e-4, seed=42)
    else:
        agent = DQNAgent(env=env, learning_rate=1e-4, seed=42)
    
    session.join(
        participant_id=agent_id,
        agent_type=agent_type,
        metadata={"learning_rate": agent.learning_rate},
    )
    
    obs, _ = env.reset(seed=42)
    episode_experiences = []
    episode_reward = 0
    
    for step in range(timesteps):
        action, _ = agent.predict(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        experience = {
            "obs": obs.tolist(),
            "action": int(action),
            "reward": float(reward),
            "next_obs": next_obs.tolist(),
            "done": bool(terminated or truncated),
        }
        episode_experiences.append(experience)
        episode_reward += reward
        
        obs = next_obs
        
        if terminated or truncated:
            shared_buffer.add(episode_experiences, contributor_id=agent_id)
            session.share_experience(agent_id, episode_experiences)
            
            if step % 5000 == 0 and step > 0:
                shared_exp = shared_buffer.sample(
                    batch_size=100,
                    exclude_contributor=agent_id,
                )
                
                if shared_exp:
                    print(f"  {agent_id}: Received {len(shared_exp)} shared experiences")
            
            episode_experiences = []
            episode_reward = 0
            obs, _ = env.reset()
    
    results = agent.train(total_timesteps=timesteps, progress_bar=False)
    eval_results = agent.evaluate(episodes=10, deterministic=True)
    
    model_path = Path(f"./models/collab/{agent_id}.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    
    session.share_model_update(
        participant_id=agent_id,
        model_path=model_path,
        metrics={
            "mean_reward": eval_results["mean_reward"],
            "std_reward": eval_results["std_reward"],
        },
    )
    
    print(f"\n{agent_id} Results:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"  Episodes: {results.get('episodes', 0)}")
    
    env.close()
    
    return eval_results


def main():
    print("="*60)
    print("Collaborative Training Example")
    print("="*60)
    
    session = CollaborationSession(
        session_id="demo_session_001",
    )
    
    shared_buffer = SharedReplayBuffer(
        capacity=50000,
        min_size=1000,
    )
    
    print("\nStarting collaborative training with 3 agents...")
    
    agents = [
        ("agent_ppo_1", "PPO"),
        ("agent_ppo_2", "PPO"),
        ("agent_dqn_1", "DQN"),
    ]
    
    results = {}
    
    for agent_id, agent_type in agents:
        result = train_collaborative_agent(
            agent_id=agent_id,
            agent_type=agent_type,
            session=session,
            shared_buffer=shared_buffer,
            timesteps=15000,
        )
        results[agent_id] = result
        
        time.sleep(1)
    
    print("\n" + "="*60)
    print("Collaboration Session Summary")
    print("="*60)
    
    session_stats = session.get_session_stats()
    print(f"\nSession ID: {session_stats['session_id']}")
    print(f"Participants: {session_stats['num_participants']}")
    print(f"Total Experiences Shared: {session_stats['total_experiences']}")
    print(f"Total Model Updates: {session_stats['total_updates']}")
    
    buffer_stats = shared_buffer.get_stats()
    print(f"\nShared Buffer Statistics:")
    print(f"  Size: {buffer_stats['size']}/{buffer_stats['capacity']}")
    print(f"  Total Added: {buffer_stats['total_added']}")
    print(f"  Total Sampled: {buffer_stats['total_sampled']}")
    print(f"  Contributors: {buffer_stats['num_contributors']}")
    
    best_model = session.get_best_model(metric="mean_reward")
    if best_model:
        print(f"\nBest Model:")
        print(f"  Participant: {best_model['participant_id']}")
        print(f"  Mean Reward: {best_model['metrics']['mean_reward']:.2f}")
    
    print("\n" + "="*60)
    print("Individual Results")
    print("="*60)
    
    for agent_id, result in results.items():
        print(f"\n{agent_id}:")
        print(f"  Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    
    print("\n✅ Collaborative training complete!")


if __name__ == "__main__":
    main()
