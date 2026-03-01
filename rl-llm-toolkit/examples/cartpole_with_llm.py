from rl_llm_toolkit import RLEnvironment, PPOAgent, OllamaBackend, LLMRewardShaper
from pathlib import Path


def main():
    print("Training PPO agent on CartPole-v1 with LLM reward shaping")
    print("Note: This requires Ollama running locally with llama3 model")
    print("Install: https://ollama.ai/")
    print("Run: ollama pull llama3")
    print()
    
    env = RLEnvironment("CartPole-v1")
    
    try:
        llm = OllamaBackend(
            model="llama3",
            base_url="http://localhost:11434",
            temperature=0.3,
            max_tokens=10,
        )
        
        reward_shaper = LLMRewardShaper(
            llm_backend=llm,
            prompt_template="cartpole",
            llm_weight=0.3,
            env_weight=0.7,
            use_cache=True,
            cache_size=10000,
        )
        
        print("LLM backend initialized successfully!")
        
    except Exception as e:
        print(f"Warning: Could not initialize LLM backend: {e}")
        print("Falling back to training without LLM reward shaping")
        reward_shaper = None
    
    agent = PPOAgent(
        env=env,
        reward_shaper=reward_shaper,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        seed=42,
    )
    
    results = agent.train(
        total_timesteps=50000,
        log_interval=5,
        eval_interval=10000,
        eval_episodes=5,
    )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    if reward_shaper:
        stats = reward_shaper.get_statistics()
        print(f"\nLLM Reward Shaping Statistics:")
        print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
        print(f"  Total LLM Calls: {stats['cache_misses']}")
        print(f"  Total Tokens Used: {stats['llm_stats']['total_tokens']}")
    
    model_path = Path("./outputs/cartpole_llm/model.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(episodes=10, deterministic=True)
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    
    env.close()


if __name__ == "__main__":
    main()
