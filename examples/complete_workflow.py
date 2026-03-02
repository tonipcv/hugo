"""
Complete Workflow Example: End-to-End RL Pipeline

This example demonstrates a complete workflow from training to deployment:
1. Train multiple algorithms
2. Benchmark and compare
3. Upload best model to HuggingFace
4. Submit to leaderboard
5. Save for production deployment
"""

from rl_llm_toolkit import RLEnvironment, PPOAgent, DQNAgent
from rl_llm_toolkit.agents.cql import CQLAgent
from rl_llm_toolkit.benchmarks import BenchmarkSuite, PerformanceMetrics
from rl_llm_toolkit.integrations import HuggingFaceHub
from rl_llm_toolkit.integrations.leaderboard import Leaderboard
from pathlib import Path
import json
import os


def create_ppo_agent(env, seed, **kwargs):
    return PPOAgent(env=env, learning_rate=3e-4, seed=seed, **kwargs)


def create_dqn_agent(env, seed, **kwargs):
    return DQNAgent(env=env, learning_rate=1e-4, seed=seed, **kwargs)


def main():
    print("="*80)
    print("COMPLETE RL WORKFLOW: TRAINING TO DEPLOYMENT")
    print("="*80)
    
    env_name = "CartPole-v1"
    output_dir = Path("./complete_workflow_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: BENCHMARK MULTIPLE ALGORITHMS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: BENCHMARKING ALGORITHMS")
    print("="*80)
    
    suite = BenchmarkSuite(output_dir=output_dir / "benchmarks")
    
    algorithms = {
        "PPO": create_ppo_agent,
        "DQN": create_dqn_agent,
    }
    
    comparison = suite.compare_algorithms(
        agent_factories=algorithms,
        env_names=[env_name],
        num_seeds=3,
        total_timesteps=50000,
    )
    
    # Find best algorithm
    best_algo = None
    best_reward = float('-inf')
    
    for key, result in comparison.items():
        reward = result["aggregated"]["mean_reward"]
        if reward > best_reward:
            best_reward = reward
            best_algo = result["algorithm"]
    
    print(f"\n✅ Best Algorithm: {best_algo} with {best_reward:.2f} mean reward")
    
    # ========================================================================
    # STEP 2: TRAIN BEST MODEL WITH MORE TIMESTEPS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: TRAINING PRODUCTION MODEL")
    print("="*80)
    
    env = RLEnvironment(env_name)
    
    if best_algo == "PPO":
        production_agent = PPOAgent(
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            seed=42,
        )
    else:
        production_agent = DQNAgent(
            env=env,
            learning_rate=1e-4,
            buffer_size=50000,
            batch_size=32,
            seed=42,
        )
    
    print(f"Training {best_algo} for production...")
    train_results = production_agent.train(
        total_timesteps=100000,
        log_interval=10,
        progress_bar=True,
    )
    
    print(f"\n✅ Training complete: {train_results['total_timesteps']} timesteps")
    
    # ========================================================================
    # STEP 3: COMPREHENSIVE EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: EVALUATING PRODUCTION MODEL")
    print("="*80)
    
    eval_results = production_agent.evaluate(episodes=50, deterministic=True)
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Min Reward: {eval_results['min_reward']:.2f}")
    print(f"  Max Reward: {eval_results['max_reward']:.2f}")
    
    # Calculate additional metrics
    stats = production_agent.get_training_stats()
    episode_rewards = stats["stats"]["episode_rewards"]
    
    if episode_rewards:
        stability = PerformanceMetrics.calculate_stability(episode_rewards, window=10)
        asymptotic = PerformanceMetrics.calculate_asymptotic_performance(episode_rewards, window=20)
        
        print(f"\nTraining Metrics:")
        print(f"  Stability (CV): {stability:.3f}")
        print(f"  Asymptotic Performance: {asymptotic:.2f}")
    
    # ========================================================================
    # STEP 4: SAVE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: SAVING MODEL")
    print("="*80)
    
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"{best_algo.lower()}_{env_name.lower()}_production.pt"
    production_agent.save(model_path)
    
    print(f"✅ Model saved to {model_path}")
    
    # ========================================================================
    # STEP 5: SUBMIT TO LEADERBOARD
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: SUBMITTING TO LEADERBOARD")
    print("="*80)
    
    leaderboard = Leaderboard(db_path=output_dir / "leaderboard.db")
    
    submission_id = leaderboard.submit(
        env_name=env_name,
        algorithm=best_algo,
        mean_reward=eval_results["mean_reward"],
        std_reward=eval_results["std_reward"],
        total_timesteps=train_results["total_timesteps"],
        hyperparameters={
            "learning_rate": production_agent.learning_rate,
            "seed": 42,
        },
        username="production_pipeline",
    )
    
    print(f"✅ Submitted to leaderboard (ID: {submission_id})")
    
    # View leaderboard
    top_5 = leaderboard.get_leaderboard(env_name, limit=5)
    print(f"\nTop 5 on {env_name}:")
    for i, entry in enumerate(top_5, 1):
        print(f"  {i}. {entry['algorithm']}: {entry['mean_reward']:.2f}")
    
    # ========================================================================
    # STEP 6: UPLOAD TO HUGGING FACE (OPTIONAL)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: HUGGING FACE UPLOAD (OPTIONAL)")
    print("="*80)
    
    hf_token = os.getenv("HF_TOKEN")
    
    if hf_token:
        hub = HuggingFaceHub(token=hf_token)
        
        metadata = {
            "algorithm": best_algo,
            "env_name": env_name,
            "total_timesteps": train_results["total_timesteps"],
            "mean_reward": eval_results["mean_reward"],
            "std_reward": eval_results["std_reward"],
            "eval_results": eval_results,
            "hyperparameters": {
                "learning_rate": production_agent.learning_rate,
            },
            "benchmark_results": {
                "num_seeds": 3,
                "comparison": {k: v["aggregated"] for k, v in comparison.items()},
            }
        }
        
        try:
            url = hub.upload_model(
                model_path=model_path,
                repo_id=f"rl-toolkit/{best_algo.lower()}-{env_name.lower()}",
                metadata=metadata,
                commit_message=f"Production {best_algo} model for {env_name}",
            )
            print(f"✅ Model uploaded to Hugging Face: {url}")
        except Exception as e:
            print(f"⚠️  HF upload failed: {e}")
    else:
        print("⚠️  HF_TOKEN not set, skipping upload")
        print("   Set HF_TOKEN environment variable to enable upload")
    
    # ========================================================================
    # STEP 7: EXPORT RESULTS AND METADATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: EXPORTING RESULTS")
    print("="*80)
    
    # Export benchmark results
    suite.export_results("benchmark_results.json", format="json")
    suite.export_results("benchmark_results.md", format="markdown")
    
    # Export leaderboard
    leaderboard.export_leaderboard(
        env_name=env_name,
        output_path=output_dir / "leaderboard.md",
        format="markdown",
    )
    
    # Create deployment metadata
    deployment_metadata = {
        "model_path": str(model_path),
        "algorithm": best_algo,
        "environment": env_name,
        "performance": {
            "mean_reward": eval_results["mean_reward"],
            "std_reward": eval_results["std_reward"],
            "min_reward": eval_results["min_reward"],
            "max_reward": eval_results["max_reward"],
        },
        "training": {
            "total_timesteps": train_results["total_timesteps"],
            "episodes": train_results.get("episodes", 0),
        },
        "hyperparameters": {
            "learning_rate": production_agent.learning_rate,
            "seed": 42,
        },
        "benchmark_rank": 1,  # Assuming best in benchmark
        "leaderboard_id": submission_id,
    }
    
    metadata_path = output_dir / "deployment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(deployment_metadata, f, indent=2)
    
    print(f"✅ Deployment metadata saved to {metadata_path}")
    
    # ========================================================================
    # STEP 8: GENERATE DEPLOYMENT INSTRUCTIONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: DEPLOYMENT INSTRUCTIONS")
    print("="*80)
    
    deployment_instructions = f"""
# Deployment Instructions for {best_algo} on {env_name}

## Model Information
- **Algorithm**: {best_algo}
- **Environment**: {env_name}
- **Performance**: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}
- **Model Path**: {model_path}

## Quick Start

### Load Model
```python
from rl_llm_toolkit import RLEnvironment, {best_algo}Agent

env = RLEnvironment("{env_name}")
agent = {best_algo}Agent(env=env)
agent.load("{model_path}")

# Make predictions
obs, _ = env.reset()
action, info = agent.predict(obs, deterministic=True)
```

### API Server
```python
from fastapi import FastAPI
from rl_llm_toolkit import RLEnvironment, {best_algo}Agent
import numpy as np

app = FastAPI()
env = RLEnvironment("{env_name}")
agent = {best_algo}Agent(env=env)
agent.load("{model_path}")

@app.post("/predict")
async def predict(observation: list[float]):
    obs = np.array(observation)
    action, _ = agent.predict(obs, deterministic=True)
    return {{"action": int(action)}}
```

### Docker Deployment
```bash
docker build -t rl-model:latest .
docker run -p 8000:8000 -v {model_path}:/models/agent.pt rl-model:latest
```

## Performance Benchmarks
- Training Time: ~{train_results.get('total_timesteps', 0) / 1000:.1f}k timesteps
- Evaluation Episodes: 50
- Success Rate: {(eval_results['mean_reward'] / 500) * 100:.1f}%

## Monitoring
- Track prediction latency
- Monitor model confidence
- Log prediction distribution
- Set up alerts for performance degradation

## Next Steps
1. Deploy to staging environment
2. Run integration tests
3. Monitor performance metrics
4. Deploy to production
5. Set up A/B testing
"""
    
    instructions_path = output_dir / "DEPLOYMENT.md"
    with open(instructions_path, "w") as f:
        f.write(deployment_instructions)
    
    print(f"✅ Deployment instructions saved to {instructions_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Benchmarked {len(algorithms)} algorithms")
    print(f"  ✅ Best algorithm: {best_algo}")
    print(f"  ✅ Production model trained: {train_results['total_timesteps']} timesteps")
    print(f"  ✅ Performance: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  ✅ Model saved: {model_path}")
    print(f"  ✅ Leaderboard submission: #{submission_id}")
    print(f"  ✅ Results exported to: {output_dir}")
    
    print(f"\n📁 Output Files:")
    print(f"  - Model: {model_path}")
    print(f"  - Metadata: {metadata_path}")
    print(f"  - Instructions: {instructions_path}")
    print(f"  - Benchmarks: {output_dir / 'benchmarks'}")
    print(f"  - Leaderboard: {output_dir / 'leaderboard.md'}")
    
    print(f"\n🚀 Ready for deployment!")
    
    env.close()


if __name__ == "__main__":
    main()
