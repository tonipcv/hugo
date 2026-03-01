# Getting Started Tutorial

This tutorial will walk you through the basics of RL-LLM Toolkit, from installation to training your first agent.

## Prerequisites

- Python 3.10 or higher
- Basic understanding of reinforcement learning concepts
- (Optional) CUDA-capable GPU for faster training

## Installation

### Option 1: PyPI (Recommended)

```bash
pip install rl-llm-toolkit
```

### Option 2: From Source

```bash
git clone https://github.com/yourusername/rl-llm-toolkit.git
cd rl-llm-toolkit
pip install -e ".[dev,examples,llm]"
```

### Option 3: Docker

```bash
docker pull rl-llm-toolkit:latest
docker run -it rl-llm-toolkit python examples/cartpole_basic.py
```

## Your First Training Session

### Step 1: Import the Library

```python
from rl_llm_toolkit import RLEnvironment, PPOAgent
```

### Step 2: Create an Environment

```python
env = RLEnvironment("CartPole-v1")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
```

### Step 3: Initialize an Agent

```python
agent = PPOAgent(
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    seed=42,
)
```

### Step 4: Train the Agent

```python
results = agent.train(
    total_timesteps=100000,
    log_interval=10,
    eval_interval=10000,
)

print(f"Training completed in {results['episodes']} episodes")
```

### Step 5: Evaluate Performance

```python
eval_results = agent.evaluate(episodes=10, deterministic=True)
print(f"Mean reward: {eval_results['mean_reward']:.2f}")
print(f"Std reward: {eval_results['std_reward']:.2f}")
```

### Step 6: Save the Model

```python
from pathlib import Path

model_path = Path("./models/my_first_agent.pt")
model_path.parent.mkdir(parents=True, exist_ok=True)
agent.save(model_path)
print(f"Model saved to {model_path}")
```

## Complete Example

Here's the full code:

```python
from rl_llm_toolkit import RLEnvironment, PPOAgent
from pathlib import Path

# Create environment
env = RLEnvironment("CartPole-v1")

# Create agent
agent = PPOAgent(
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    seed=42,
)

# Train
print("Starting training...")
results = agent.train(
    total_timesteps=100000,
    log_interval=10,
    eval_interval=10000,
)

# Evaluate
print("\nEvaluating agent...")
eval_results = agent.evaluate(episodes=10, deterministic=True)
print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

# Save
model_path = Path("./models/cartpole_agent.pt")
model_path.parent.mkdir(parents=True, exist_ok=True)
agent.save(model_path)
print(f"Model saved to {model_path}")

# Clean up
env.close()
```

## Adding LLM Reward Shaping

### Step 1: Set Up Ollama (Local LLM)

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3

# Verify it's running
curl http://localhost:11434/api/tags
```

### Step 2: Create LLM Backend

```python
from rl_llm_toolkit import OllamaBackend

llm = OllamaBackend(
    model="llama3",
    base_url="http://localhost:11434",
    temperature=0.3,
)
```

### Step 3: Create Reward Shaper

```python
from rl_llm_toolkit import LLMRewardShaper

reward_shaper = LLMRewardShaper(
    llm_backend=llm,
    prompt_template="cartpole",
    llm_weight=0.3,  # 30% LLM reward
    env_weight=0.7,  # 70% environment reward
    use_cache=True,
)
```

### Step 4: Train with LLM Shaping

```python
agent = PPOAgent(
    env=env,
    reward_shaper=reward_shaper,
    learning_rate=3e-4,
)

agent.train(total_timesteps=50000)

# Check LLM usage
stats = reward_shaper.get_statistics()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Total LLM calls: {stats['cache_misses']}")
```

## Using Different Algorithms

### DQN for Discrete Actions

```python
from rl_llm_toolkit import DQNAgent

env = RLEnvironment("LunarLander-v2")

agent = DQNAgent(
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,
    epsilon_decay=50000,
)

agent.train(total_timesteps=200000)
```

## Using the CLI

### Generate Config

```bash
rl-llm quickstart --env CartPole-v1 --algorithm ppo
```

### Train from Config

```bash
rl-llm train config_quickstart.yaml
```

### Evaluate Model

```bash
rl-llm evaluate ./outputs/model.pt CartPole-v1 --episodes 10
```

## Monitoring Training

### TensorBoard

```python
from rl_llm_toolkit.utils import Logger

logger = Logger(
    log_dir="./logs",
    experiment_name="my_experiment",
    use_tensorboard=True,
)

# Training loop
for step in range(1000):
    # ... training code ...
    logger.log_metrics({"reward": reward}, step=step)

logger.close()
```

Then view in browser:
```bash
tensorboard --logdir ./logs
```

### Weights & Biases

```python
logger = Logger(
    log_dir="./logs",
    use_wandb=True,
    wandb_config={
        "project": "rl-llm-toolkit",
        "entity": "your-username",
    }
)
```

## Common Patterns

### Hyperparameter Tuning

```python
learning_rates = [1e-4, 3e-4, 1e-3]
best_reward = -float('inf')
best_lr = None

for lr in learning_rates:
    agent = PPOAgent(env=env, learning_rate=lr)
    agent.train(total_timesteps=50000)
    
    eval_results = agent.evaluate(episodes=10)
    if eval_results['mean_reward'] > best_reward:
        best_reward = eval_results['mean_reward']
        best_lr = lr

print(f"Best learning rate: {best_lr}")
```

### Loading and Continuing Training

```python
# Load existing model
agent = PPOAgent(env=env)
agent.load("./models/checkpoint.pt")

# Continue training
agent.train(total_timesteps=50000)

# Save updated model
agent.save("./models/checkpoint_v2.pt")
```

### Custom Reward Templates

```python
from rl_llm_toolkit.rewards.prompts import PromptTemplate

custom_template = PromptTemplate(
    system_prompt="You are a reward designer for a custom task.",
    user_template=(
        "State: {state}\n"
        "Action: {action}\n"
        "Reward (-1 to 1):"
    )
)

reward_shaper = LLMRewardShaper(
    llm_backend=llm,
    prompt_template=custom_template,
)
```

## Troubleshooting

### Issue: Training is slow

**Solution**: Use GPU acceleration
```python
agent = PPOAgent(env=env, device="cuda")
```

### Issue: Out of memory

**Solution**: Reduce batch size
```python
agent = PPOAgent(env=env, batch_size=32, n_steps=1024)
```

### Issue: LLM connection error

**Solution**: Check Ollama is running
```bash
ollama serve
```

### Issue: Poor performance

**Solutions**:
1. Train longer: `total_timesteps=200000`
2. Tune hyperparameters
3. Try different algorithms
4. Adjust reward shaping weights

## Next Steps

- **Advanced Training**: Check `examples/advanced_training.py`
- **Custom Environments**: See `rl_llm_toolkit/environments/trading.py`
- **Algorithm Comparison**: Run `examples/compare_algorithms.py`
- **API Reference**: Read `docs/api_reference.md`
- **Architecture**: Understand the design in `docs/architecture.md`

## Resources

- [Examples Directory](../examples/)
- [API Documentation](api_reference.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [GitHub Issues](https://github.com/yourusername/rl-llm-toolkit/issues)

Happy training! 🚀
