from rl_llm_toolkit import RLEnvironment, PPOAgent
from rl_llm_toolkit.integrations import HuggingFaceHub
from rl_llm_toolkit.integrations.leaderboard import Leaderboard
from pathlib import Path
import os


def main():
    print("="*60)
    print("Hugging Face Integration Example")
    print("="*60)
    
    env = RLEnvironment("CartPole-v1")
    
    print("\n1. Training agent...")
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        seed=42,
    )
    
    results = agent.train(
        total_timesteps=50000,
        log_interval=10,
        progress_bar=True,
    )
    
    print("\n2. Evaluating agent...")
    eval_results = agent.evaluate(episodes=20, deterministic=True)
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    
    model_path = Path("./models/hf_example/cartpole_ppo.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    
    print("\n3. Uploading to Hugging Face Hub...")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️  HF_TOKEN not set. Skipping upload.")
        print("To upload, set HF_TOKEN environment variable:")
        print("  export HF_TOKEN='your_token_here'")
    else:
        hub = HuggingFaceHub(token=hf_token)
        
        metadata = {
            "algorithm": "PPO",
            "env_name": "CartPole-v1",
            "total_timesteps": results["total_timesteps"],
            "mean_reward": eval_results["mean_reward"],
            "std_reward": eval_results["std_reward"],
            "eval_results": eval_results,
            "hyperparameters": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "gamma": 0.99,
            }
        }
        
        try:
            url = hub.upload_model(
                model_path=model_path,
                repo_id="username/cartpole-ppo-example",
                metadata=metadata,
                commit_message="Upload CartPole PPO model",
                private=False,
            )
            print(f"✅ Model uploaded: {url}")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
    
    print("\n4. Adding to local leaderboard...")
    leaderboard = Leaderboard()
    
    submission_id = leaderboard.submit(
        env_name="CartPole-v1",
        algorithm="PPO",
        mean_reward=eval_results["mean_reward"],
        std_reward=eval_results["std_reward"],
        total_timesteps=results["total_timesteps"],
        hyperparameters={
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
        },
        model_id="username/cartpole-ppo-example",
        username="demo_user",
    )
    
    print(f"Submission ID: {submission_id}")
    
    print("\n5. Viewing leaderboard...")
    top_submissions = leaderboard.get_leaderboard("CartPole-v1", limit=5)
    
    print("\nTop 5 CartPole-v1 Submissions:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Algorithm':<12} {'Mean Reward':<15} {'User':<15} {'Date':<12}")
    print("-" * 80)
    
    for i, entry in enumerate(top_submissions, 1):
        print(
            f"{i:<6} {entry['algorithm']:<12} "
            f"{entry['mean_reward']:.2f} ± {entry['std_reward']:.2f}  "
            f"{entry['username']:<15} {entry['submission_date'][:10]:<12}"
        )
    
    print("\n6. Comparing algorithms...")
    comparison = leaderboard.compare_algorithms("CartPole-v1")
    
    print("\nAlgorithm Comparison:")
    print("-" * 60)
    for algo, stats in comparison.items():
        print(f"{algo}:")
        print(f"  Best: {stats['best_reward']:.2f}")
        print(f"  Average: {stats['avg_reward']:.2f}")
        print(f"  Submissions: {stats['num_submissions']}")
    
    print("\n7. Exporting leaderboard...")
    export_path = Path("./outputs/leaderboard_cartpole.md")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    leaderboard.export_leaderboard(
        env_name="CartPole-v1",
        output_path=export_path,
        format="markdown",
    )
    
    print(f"✅ Leaderboard exported to {export_path}")
    
    env.close()


if __name__ == "__main__":
    main()
