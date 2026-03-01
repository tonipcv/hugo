# Frequently Asked Questions

## General Questions

### What is RL-LLM Toolkit?

RL-LLM Toolkit is an open-source framework that integrates Reinforcement Learning with Large Language Models. It allows you to train RL agents with LLM-based reward shaping, reducing the cost and complexity of traditional RLHF approaches.

### Why use LLMs for reward shaping?

LLMs can simulate human feedback at a fraction of the cost of actual human labeling. They provide:
- Dense reward signals instead of sparse rewards
- Domain knowledge for better exploration
- Reduced training time in complex environments
- Cost savings of 50%+ compared to human RLHF

### What algorithms are supported?

Currently:
- **PPO** (Proximal Policy Optimization) - Best for continuous and discrete actions
- **DQN** (Deep Q-Network) - Best for discrete actions

Coming soon: A2C, SAC, TD3, and offline RL algorithms.

### Which LLM providers are supported?

- **Ollama** (local, free) - Recommended for development
- **OpenAI** (GPT-4, GPT-3.5) - Best quality
- **Anthropic** (Claude) - Coming soon

## Installation & Setup

### How do I install the toolkit?

```bash
pip install rl-llm-toolkit
```

For development:
```bash
git clone https://github.com/yourusername/rl-llm-toolkit.git
cd rl-llm-toolkit
pip install -e ".[dev,examples,llm]"
```

### Do I need a GPU?

No, but it's highly recommended. Training is 10-50x faster on GPU. The toolkit automatically detects and uses GPU if available.

### How do I use Ollama locally?

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3

# Verify
curl http://localhost:11434/api/tags
```

### Can I use this in Google Colab?

Yes! Install with:
```python
!pip install rl-llm-toolkit
```

GPU is automatically available in Colab.

## Training

### How long does training take?

Depends on the environment and algorithm:
- **CartPole**: 2-5 minutes (100k timesteps)
- **LunarLander**: 10-20 minutes (200k timesteps)
- **Custom Trading**: 15-30 minutes (100k timesteps)

With GPU, training is significantly faster.

### How do I know if training is working?

Monitor the mean reward over episodes. It should generally increase over time. Use:
```python
agent.train(total_timesteps=100000, log_interval=10)
```

### My agent isn't learning. What should I do?

1. **Check hyperparameters**: Start with defaults
2. **Train longer**: Try 2-5x more timesteps
3. **Verify environment**: Test with random actions
4. **Reduce complexity**: Start with simpler environments
5. **Check logs**: Look for NaN losses or errors

### How do I tune hyperparameters?

Start with defaults, then adjust:
- **Learning rate**: Try 1e-4, 3e-4, 1e-3
- **Batch size**: 32, 64, 128
- **Network size**: (64, 64), (128, 128), (256, 256)

Use grid search or tools like Optuna.

### Can I pause and resume training?

Yes:
```python
# Save checkpoint
agent.save("checkpoint.pt")

# Later, load and continue
agent = PPOAgent(env=env)
agent.load("checkpoint.pt")
agent.train(total_timesteps=50000)
```

## LLM Integration

### How much does LLM reward shaping cost?

With caching enabled (default), costs are minimal:
- **Ollama (local)**: Free
- **OpenAI GPT-4**: ~$0.50-2.00 per 100k timesteps
- **OpenAI GPT-3.5**: ~$0.05-0.20 per 100k timesteps

Cache hit rates typically exceed 80%.

### How does caching work?

The toolkit caches LLM responses based on state-action pairs. Identical transitions reuse cached rewards, reducing API calls by 80%+.

### Can I use custom prompts?

Yes:
```python
from rl_llm_toolkit.rewards.prompts import PromptTemplate

custom = PromptTemplate(
    system_prompt="Your system prompt here",
    user_template="State: {state}\nAction: {action}\nReward:"
)

reward_shaper = LLMRewardShaper(
    llm_backend=llm,
    prompt_template=custom,
)
```

### What if the LLM is slow?

1. **Use caching**: Enabled by default
2. **Use local LLM**: Ollama is faster than API calls
3. **Reduce max_tokens**: Set to 10-20 for reward generation
4. **Batch requests**: Coming in future versions

### How do I balance LLM vs environment rewards?

Adjust the weights:
```python
reward_shaper = LLMRewardShaper(
    llm_backend=llm,
    llm_weight=0.3,  # 30% LLM
    env_weight=0.7,  # 70% environment
)
```

Start with 0.3/0.7 and adjust based on results.

## Environments

### What environments are supported?

All Gymnasium environments:
- Classic Control (CartPole, MountainCar, etc.)
- Box2D (LunarLander, BipedalWalker, etc.)
- Atari (with additional dependencies)
- Custom environments

### How do I create a custom environment?

Inherit from `gym.Env`:
```python
import gymnasium as gym
from gymnasium import spaces

class MyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(...)
        self.action_space = spaces.Discrete(...)
    
    def reset(self, seed=None, options=None):
        # Return observation, info
        pass
    
    def step(self, action):
        # Return obs, reward, terminated, truncated, info
        pass
```

### Can I use multi-agent environments?

Not yet, but it's on the roadmap for v0.2.0.

### How do I use the trading environment?

```python
from rl_llm_toolkit.environments.trading import CryptoTradingEnv
import numpy as np

# Load your price data
prices = np.loadtxt("btc_prices.csv")

env = CryptoTradingEnv(
    price_data=prices,
    initial_balance=10000.0,
)

agent = PPOAgent(env=env)
agent.train(total_timesteps=100000)
```

## Performance

### How can I speed up training?

1. **Use GPU**: `agent = PPOAgent(env=env, device="cuda")`
2. **Increase batch size**: `batch_size=128`
3. **Vectorized environments**: Coming soon
4. **Reduce logging**: `log_interval=100`

### Why is my GPU not being used?

Check:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))
```

If False, reinstall PyTorch with CUDA support.

### How much memory do I need?

- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM
- **GPU**: 4GB+ VRAM for most tasks

### Can I train on multiple GPUs?

Not yet, but multi-GPU support is planned for v0.3.0.

## Deployment

### How do I deploy a trained model?

```python
# Save model
agent.save("production_model.pt")

# In production
agent = PPOAgent(env=env)
agent.load("production_model.pt")

# Get actions
obs, _ = env.reset()
action, _ = agent.predict(obs, deterministic=True)
```

### Can I export to ONNX?

Coming in v0.2.0. For now, use PyTorch models directly.

### How do I serve models via API?

Use FastAPI:
```python
from fastapi import FastAPI
import numpy as np

app = FastAPI()
agent = PPOAgent(env=env)
agent.load("model.pt")

@app.post("/predict")
def predict(observation: list):
    obs = np.array(observation)
    action, _ = agent.predict(obs, deterministic=True)
    return {"action": int(action)}
```

## Troubleshooting

### ImportError: No module named 'rl_llm_toolkit'

Install the package:
```bash
pip install rl-llm-toolkit
```

### CUDA out of memory

Reduce batch size:
```python
agent = PPOAgent(env=env, batch_size=32, n_steps=1024)
```

### Training diverges (NaN losses)

1. Reduce learning rate: `learning_rate=1e-4`
2. Clip gradients: `max_grad_norm=0.5` (default)
3. Check environment rewards are bounded

### LLM connection timeout

Increase timeout:
```python
llm = OllamaBackend(model="llama3", timeout=60)
```

### Tests failing

Run with verbose output:
```bash
pytest -v --tb=short
```

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. We welcome:
- Bug reports and fixes
- New algorithms
- Documentation improvements
- Example scripts
- Performance optimizations

### How do I report a bug?

Open an issue on GitHub with:
- Description of the bug
- Minimal reproduction code
- Environment details (OS, Python version, etc.)
- Error messages and stack traces

### Can I add a new LLM backend?

Yes! Inherit from `LLMBackend` and implement `generate()` and `generate_batch()`. See existing backends for examples.

## Community

### Where can I get help?

- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time chat and support
- **Discussions**: Questions and ideas
- **Twitter/X**: Updates and announcements

### Is there a roadmap?

Yes! See [README.md](../README.md) for the Now-Next-Later roadmap.

### How do I stay updated?

- Star the repo on GitHub
- Follow on Twitter/X
- Join the Discord community
- Subscribe to the newsletter

## License

### What license is this under?

MIT License - free for commercial and personal use.

### Can I use this in my company?

Yes! The MIT license allows commercial use.

### Do I need to cite this project?

Not required, but appreciated:
```
@software{rl_llm_toolkit,
  title = {RL-LLM Toolkit},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/rl-llm-toolkit}
}
```
