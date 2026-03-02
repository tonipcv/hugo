from rl_llm_toolkit import RLEnvironment
from rl_llm_toolkit.agents.cql import CQLAgent
from rl_llm_toolkit.agents.iql import IQLAgent
from pathlib import Path
import numpy as np


def main():
    print("="*60)
    print("Offline RL Example: CQL and IQL")
    print("="*60)
    
    env = RLEnvironment("CartPole-v1")
    
    print("\n1. Collecting offline dataset...")
    cql_agent = CQLAgent(env=env, seed=42)
    
    dataset = cql_agent.collect_dataset(
        num_episodes=100,
        policy="random",
    )
    
    print(f"Dataset collected: {len(dataset)} transitions")
    
    rewards = [t['reward'] for t in dataset]
    print(f"Dataset statistics:")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    print(f"  Total reward: {np.sum(rewards):.2f}")
    
    print("\n2. Training CQL agent on offline data...")
    cql_agent = CQLAgent(
        env=env,
        learning_rate=3e-4,
        batch_size=256,
        cql_weight=1.0,
        seed=42,
    )
    cql_agent.load_dataset(dataset)
    
    cql_results = cql_agent.train(
        total_timesteps=50000,
        log_interval=5000,
        eval_interval=10000,
        eval_episodes=10,
        progress_bar=True,
    )
    
    print("\n3. Evaluating CQL agent...")
    cql_eval = cql_agent.evaluate(episodes=20, deterministic=True)
    print(f"CQL Mean Reward: {cql_eval['mean_reward']:.2f} ± {cql_eval['std_reward']:.2f}")
    
    print("\n4. Training IQL agent on same dataset...")
    iql_agent = IQLAgent(
        env=env,
        learning_rate=3e-4,
        batch_size=256,
        expectile=0.7,
        seed=42,
    )
    iql_agent.load_dataset(dataset)
    
    iql_results = iql_agent.train(
        total_timesteps=50000,
        log_interval=5000,
        eval_interval=10000,
        eval_episodes=10,
        progress_bar=True,
    )
    
    print("\n5. Evaluating IQL agent...")
    iql_eval = iql_agent.evaluate(episodes=20, deterministic=True)
    print(f"IQL Mean Reward: {iql_eval['mean_reward']:.2f} ± {iql_eval['std_reward']:.2f}")
    
    print("\n" + "="*60)
    print("Comparison: Offline RL vs Random Policy")
    print("="*60)
    print(f"Random Policy (dataset): ~{np.mean(rewards):.2f}")
    print(f"CQL Agent: {cql_eval['mean_reward']:.2f}")
    print(f"IQL Agent: {iql_eval['mean_reward']:.2f}")
    
    improvement_cql = cql_eval['mean_reward'] - np.mean(rewards)
    improvement_iql = iql_eval['mean_reward'] - np.mean(rewards)
    
    print(f"\nImprovement:")
    print(f"  CQL: +{improvement_cql:.2f} ({improvement_cql/np.mean(rewards)*100:.1f}%)")
    print(f"  IQL: +{improvement_iql:.2f} ({improvement_iql/np.mean(rewards)*100:.1f}%)")
    
    model_dir = Path("./models/offline_rl")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    cql_agent.save(model_dir / "cql_cartpole.pt")
    iql_agent.save(model_dir / "iql_cartpole.pt")
    print(f"\n✅ Models saved to {model_dir}")
    
    env.close()


if __name__ == "__main__":
    main()
