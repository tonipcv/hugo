# Quick Start Guide

Get started with RL-LLM Toolkit in minutes!

## Installation

```bash
pip install hugo-rl-llm
```

For development installation:
```bash
git clone https://github.com/tonipcv/hugo.git
cd hugo
pip install -e ".[dev,examples,llm]"
```

## Your First Agent

### Basic PPO Training

```python
from rl_llm_toolkit import RLEnvironment, PPOAgent

# Create environment
env = RLEnvironment("CartPole-v1")

# Create agent
agent = PPOAgent(
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
)

# Train
results = agent.train(total_timesteps=100000)

# Evaluate
eval_results = agent.evaluate(episodes=10)
print(f"Mean Reward: {eval_results['mean_reward']:.2f}")

# Save model
agent.save("models/cartpole_ppo.pt")
```

### Training with LLM Reward Shaping

```python
from rl_llm_toolkit import (
    RLEnvironment, 
    PPOAgent, 
    OllamaBackend, 
    LLMRewardShaper
)

# Set up environment
env = RLEnvironment("CartPole-v1")

# Configure LLM backend (requires Ollama running locally)
llm = OllamaBackend(
    model="llama3",
    temperature=0.3,
)

# Create reward shaper
reward_shaper = LLMRewardShaper(
    llm_backend=llm,
    prompt_template="cartpole",
    llm_weight=0.3,  # 30% LLM, 70% environment reward
    env_weight=0.7,
)

# Train with LLM-shaped rewards
agent = PPOAgent(env=env, reward_shaper=reward_shaper)
agent.train(total_timesteps=50000)

# Check LLM usage statistics
stats = reward_shaper.get_statistics()
print(f"Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
print(f"Total Tokens: {stats['llm_stats']['total_tokens']}")
```

## Using the CLI

### Generate Configuration

```bash
hugo quickstart --env CartPole-v1 --algorithm ppo --llm
```

This creates `config_quickstart.yaml`:
```yaml
env_name: CartPole-v1
algorithm: ppo
training:
  total_timesteps: 100000
  learning_rate: 0.0003
  batch_size: 64
llm:
  provider: ollama
  model: llama3
reward_shaping:
  enabled: true
  llm_weight: 0.3
```

### Train from Config

```bash
hugo train config_quickstart.yaml --output-dir ./outputs
```

### Evaluate Model

```bash
hugo evaluate ./outputs/model.pt CartPole-v1 --episodes 10 --render
```

## DQN Example

```python
from rl_llm_toolkit import RLEnvironment, DQNAgent

env = RLEnvironment("LunarLander-v2")

agent = DQNAgent(
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,
    epsilon_decay=50000,
)

agent.train(total_timesteps=200000)
agent.save("models/lunarlander_dqn.pt")
```

## Custom Trading Environment

```python
from rl_llm_toolkit.environments.trading import CryptoTradingEnv
from rl_llm_toolkit import PPOAgent
import numpy as np

# Generate or load price data
price_data = np.loadtxt("btc_prices.csv")

# Create trading environment
env = CryptoTradingEnv(
    price_data=price_data,
    initial_balance=10000.0,
    transaction_fee=0.001,
)

# Train trading agent
agent = PPOAgent(env=env)
agent.train(total_timesteps=100000)

# Evaluate
results = agent.evaluate(episodes=10)
print(f"Average Return: {results['mean_reward']:.2%}")
```

## LLM Backend Options

### Ollama (Local)

```python
from rl_llm_toolkit.llm import OllamaBackend

llm = OllamaBackend(
    model="llama3",
    base_url="http://localhost:11434",
    temperature=0.7,
)
```

### OpenAI

```python
from rl_llm_toolkit.llm import OpenAIBackend
import os

llm = OpenAIBackend(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)
```

## Next Steps

- **Examples**: Check out `examples/` for complete training scripts
- **Documentation**: Read the full docs for advanced features
- **Tutorials**: Follow our Jupyter notebooks in `examples/notebooks/`
- **Community**: Join our Discord for help and discussions

## Common Issues

### Ollama Not Running

If you get connection errors with Ollama:
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3

# Verify it's running
curl http://localhost:11434/api/tags
```

### GPU Not Detected

```python
import torch
print(torch.cuda.is_available())  # Should be True

# Force CPU if needed
agent = PPOAgent(env=env, device="cpu")
```

### Out of Memory

Reduce batch size or n_steps:
```python
agent = PPOAgent(
    env=env,
    n_steps=1024,  # Default: 2048
    batch_size=32,  # Default: 64
)
```

## Performance Tips

1. **Use GPU**: Training is 10-50x faster on GPU
2. **Enable Caching**: LLM reward shaping with cache reduces API calls by 80%+
3. **Tune Hyperparameters**: Start with defaults, then optimize
4. **Monitor Training**: Use TensorBoard or Weights & Biases
5. **Vectorized Environments**: Use multiple parallel environments for faster training

```python
# Enable logging
from rl_llm_toolkit.utils import Logger

logger = Logger(
    log_dir="./logs",
    use_tensorboard=True,
    use_wandb=True,
)
```

Happy training! 🚀
