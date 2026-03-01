from rl_llm_toolkit import RLEnvironment, PPOAgent, OllamaBackend, LLMRewardShaper
from rl_llm_toolkit.utils import Logger, plot_training_curves
from pathlib import Path
import numpy as np


def main():
    print("Advanced Training Example with Full Logging")
    
    env = RLEnvironment("CartPole-v1")
    
    llm = OllamaBackend(model="llama3", temperature=0.3)
    reward_shaper = LLMRewardShaper(
        llm_backend=llm,
        prompt_template="cartpole",
        llm_weight=0.3,
        env_weight=0.7,
    )
    
    logger = Logger(
        log_dir=Path("./logs"),
        experiment_name="cartpole_advanced",
        use_tensorboard=True,
        use_wandb=False,
    )
    
    agent = PPOAgent(
        env=env,
        reward_shaper=reward_shaper,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        seed=42,
    )
    
    logger.log_hyperparameters({
        "algorithm": "PPO",
        "env": "CartPole-v1",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "llm_weight": 0.3,
        "env_weight": 0.7,
    })
    
    print("Starting training...")
    
    num_updates = 50
    for update in range(num_updates):
        agent.train(
            total_timesteps=2048,
            log_interval=1,
            progress_bar=False,
        )
        
        stats = agent.get_training_stats()
        
        if stats["stats"]["episode_rewards"]:
            recent_rewards = stats["stats"]["episode_rewards"][-10:]
            mean_reward = np.mean(recent_rewards)
            
            logger.log_metrics({
                "mean_reward": mean_reward,
                "episodes": stats["episode_count"],
                "timesteps": stats["total_timesteps"],
            }, step=update)
            
            print(f"Update {update+1}/{num_updates} | Mean Reward: {mean_reward:.2f}")
        
        if (update + 1) % 10 == 0:
            eval_results = agent.evaluate(episodes=5, deterministic=True)
            logger.log_metrics({
                "eval_mean_reward": eval_results["mean_reward"],
                "eval_std_reward": eval_results["std_reward"],
            }, step=update)
            print(f"Evaluation: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    
    model_path = Path("./outputs/cartpole_advanced/model.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    
    stats = agent.get_training_stats()
    plot_training_curves(
        {
            "Episode Rewards": stats["stats"]["episode_rewards"],
            "Episode Lengths": stats["stats"]["episode_lengths"],
        },
        save_path=Path("./outputs/cartpole_advanced/training_curves.png"),
        title="CartPole Training Progress"
    )
    
    shaper_stats = reward_shaper.get_statistics()
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Total Episodes: {stats['episode_count']}")
    print(f"Total Timesteps: {stats['total_timesteps']}")
    print(f"\nLLM Statistics:")
    print(f"  Cache Hit Rate: {shaper_stats['cache_hit_rate']:.2%}")
    print(f"  Total LLM Calls: {shaper_stats['cache_misses']}")
    print(f"  Total Tokens: {shaper_stats['llm_stats']['total_tokens']}")
    print(f"  Avg Tokens/Call: {shaper_stats['llm_stats']['avg_tokens_per_request']:.1f}")
    
    logger.close()
    env.close()


if __name__ == "__main__":
    main()
