# Advanced Features Guide

This guide covers advanced features of RL-LLM Toolkit for power users and researchers.

## Table of Contents

1. [Offline Reinforcement Learning](#offline-rl)
2. [Multi-Agent Systems](#multi-agent)
3. [Hugging Face Integration](#huggingface)
4. [Community Leaderboards](#leaderboards)
5. [Collaborative Training](#collaboration)
6. [Visual Reward Shaping](#visual-rewards)
7. [Benchmarking Suite](#benchmarking)
8. [Advanced Trading Environments](#trading)

---

## Offline Reinforcement Learning {#offline-rl}

Train RL agents from fixed datasets without environment interaction.

### Conservative Q-Learning (CQL)

```python
from rl_llm_toolkit import RLEnvironment, CQLAgent

env = RLEnvironment("CartPole-v1")
agent = CQLAgent(env=env, cql_weight=1.0)

# Collect offline dataset
dataset = agent.collect_dataset(num_episodes=100, policy="random")

# Train on offline data
agent.train(total_timesteps=50000)

# Evaluate
results = agent.evaluate(episodes=20)
print(f"Mean reward: {results['mean_reward']:.2f}")
```

### Implicit Q-Learning (IQL)

```python
from rl_llm_toolkit.agents.iql import IQLAgent

agent = IQLAgent(
    env=env,
    expectile=0.7,
    temperature=3.0,
)

agent.load_dataset(dataset)
agent.train(total_timesteps=50000)
```

**Key Features:**
- Learn from suboptimal data
- No environment interaction needed
- Conservative value estimates
- Stable offline learning

**Use Cases:**
- Learning from logged data
- Safe policy improvement
- Batch RL scenarios
- Historical data utilization

---

## Multi-Agent Systems {#multi-agent}

Train multiple agents that interact and collaborate.

### MADDPG (Multi-Agent DDPG)

```python
from rl_llm_toolkit.multiagent import MADDPGAgent, CooperativeNavigationEnv

# Create multi-agent environment
env = CooperativeNavigationEnv(
    num_agents=3,
    num_landmarks=3,
)

# Train MADDPG
agent = MADDPGAgent(
    env=env,
    num_agents=3,
    learning_rate_actor=1e-4,
    learning_rate_critic=1e-3,
)

agent.train(total_timesteps=100000)
```

**Features:**
- Centralized training, decentralized execution
- Cooperative and competitive scenarios
- Communication between agents
- Scalable to many agents

**Custom Multi-Agent Environments:**

```python
from rl_llm_toolkit.multiagent import MultiAgentEnv
import gymnasium as gym

class MyMultiAgentEnv(MultiAgentEnv):
    def __init__(self, num_agents):
        obs_space = gym.spaces.Box(...)
        action_space = gym.spaces.Discrete(...)
        super().__init__(num_agents, obs_space, action_space)
    
    def reset(self, seed=None, options=None):
        # Return dict of observations
        return {f"agent_{i}": obs for i in range(self.num_agents)}, {}
    
    def step(self, actions):
        # Process actions, return observations, rewards, etc.
        return observations, rewards, terminated, truncated, info
```

---

## Hugging Face Integration {#huggingface}

Share and discover models on Hugging Face Hub.

### Upload Model

```python
from rl_llm_toolkit.integrations import HuggingFaceHub

hub = HuggingFaceHub(token="your_hf_token")

# Upload trained model
url = hub.upload_model(
    model_path="./models/my_agent.pt",
    repo_id="username/cartpole-ppo",
    metadata={
        "algorithm": "PPO",
        "env_name": "CartPole-v1",
        "mean_reward": 475.0,
        "hyperparameters": {...},
    },
)
```

### Download Model

```python
# Download pre-trained model
model_path = hub.download_model("username/cartpole-ppo")

# Load and use
agent = PPOAgent(env=env)
agent.load(model_path)
```

### Browse Models

```python
# List community models
models = hub.list_models(filter_tag="rl-llm-toolkit", limit=20)

for model in models:
    print(f"{model['id']}: {model['downloads']} downloads")
```

---

## Community Leaderboards {#leaderboards}

Track and compare agent performance.

### Submit Results

```python
from rl_llm_toolkit.integrations.leaderboard import Leaderboard

leaderboard = Leaderboard()

# Submit your results
submission_id = leaderboard.submit(
    env_name="CartPole-v1",
    algorithm="PPO",
    mean_reward=475.0,
    std_reward=25.0,
    total_timesteps=100000,
    hyperparameters={"lr": 3e-4},
    username="your_name",
)
```

### View Leaderboard

```python
# Get top submissions
top_10 = leaderboard.get_leaderboard("CartPole-v1", limit=10)

for i, entry in enumerate(top_10, 1):
    print(f"{i}. {entry['algorithm']}: {entry['mean_reward']:.2f}")
```

### Compare Algorithms

```python
comparison = leaderboard.compare_algorithms("CartPole-v1")

for algo, stats in comparison.items():
    print(f"{algo}: Best={stats['best_reward']:.2f}, Avg={stats['avg_reward']:.2f}")
```

### Export Leaderboard

```python
leaderboard.export_leaderboard(
    env_name="CartPole-v1",
    output_path="./leaderboard.md",
    format="markdown",
)
```

---

## Collaborative Training {#collaboration}

Train agents collaboratively with experience sharing.

### Create Session

```python
from rl_llm_toolkit.collaboration import CollaborationSession, SharedReplayBuffer

session = CollaborationSession(session_id="my_session")
shared_buffer = SharedReplayBuffer(capacity=100000)
```

### Join and Share

```python
# Agent 1
session.join("agent_1", "PPO", metadata={"lr": 3e-4})

# Collect and share experiences
experiences = [...]  # Your training data
session.share_experience("agent_1", experiences)
shared_buffer.add(experiences, contributor_id="agent_1")
```

### Use Shared Experiences

```python
# Agent 2 can use Agent 1's experiences
shared_exp = shared_buffer.sample(batch_size=64, exclude_contributor="agent_2")

# Or get from session
all_shared = session.get_shared_experiences("agent_2", max_count=1000)
```

### Share Model Updates

```python
session.share_model_update(
    participant_id="agent_1",
    model_path="./models/agent_1.pt",
    metrics={"mean_reward": 450.0},
)

# Get best model
best = session.get_best_model(metric="mean_reward")
print(f"Best model: {best['participant_id']} with {best['metrics']['mean_reward']:.2f}")
```

---

## Visual Reward Shaping {#visual-rewards}

Use vision models for reward shaping in visual environments.

### Setup

```python
from rl_llm_toolkit.vision import VideoReasoningBackend, VisualRewardShaper

video_backend = VideoReasoningBackend(
    model="gpt-4-vision",
    max_frames=4,
)

visual_shaper = VisualRewardShaper(
    video_backend=video_backend,
    visual_weight=0.3,
    env_weight=0.7,
)
```

### Use in Training

```python
# During training loop
shaped_reward, metadata = visual_shaper.shape_reward(
    env_reward=reward,
    before_frame=frame_before,
    after_frame=frame_after,
    action=action,
)

# Use shaped_reward for training
```

### Analyze Trajectories

```python
analysis = video_backend.analyze_trajectory(
    frames=[frame1, frame2, frame3],
    actions=[0, 1, 0],
    context="Navigate to goal",
)

print(f"Trajectory quality: {analysis['trajectory_quality']:.2f}")
print(f"Feedback: {analysis['feedback']}")
```

---

## Benchmarking Suite {#benchmarking}

Comprehensive performance evaluation.

### Run Benchmark

```python
from rl_llm_toolkit.benchmarks import BenchmarkSuite

suite = BenchmarkSuite(output_dir="./benchmarks")

# Benchmark single algorithm
result = suite.run_benchmark(
    agent_factory=lambda env, seed: PPOAgent(env=env, seed=seed),
    env_name="CartPole-v1",
    num_seeds=5,
    total_timesteps=100000,
)
```

### Compare Algorithms

```python
comparison = suite.compare_algorithms(
    agent_factories={
        "PPO": create_ppo,
        "DQN": create_dqn,
        "CQL": create_cql,
    },
    env_names=["CartPole-v1", "LunarLander-v2"],
    num_seeds=3,
)
```

### Performance Profiling

```python
profile = suite.profile_performance(
    agent=agent,
    env_name="CartPole-v1",
    num_steps=10000,
)

print(f"FPS: {profile['fps']:.1f}")
print(f"Memory: {profile['memory_increase_mb']:.1f} MB")
```

### Calculate Metrics

```python
from rl_llm_toolkit.benchmarks import PerformanceMetrics

# Sample efficiency
timesteps_to_solve = PerformanceMetrics.calculate_sample_efficiency(
    rewards=episode_rewards,
    threshold=475.0,
    timesteps=list(range(len(episode_rewards))),
)

# Training stability
stability = PerformanceMetrics.calculate_stability(episode_rewards, window=10)

# Robustness across seeds
robustness = PerformanceMetrics.calculate_robustness(
    seed_results=[{"mean_reward": 450}, {"mean_reward": 475}, ...],
    metric_key="mean_reward",
)
```

---

## Advanced Trading Environments {#trading}

Realistic financial trading simulation.

### Stock Trading Environment

```python
from rl_llm_toolkit.environments.stock_trading import StockTradingEnv
import numpy as np

# Load your price data
prices = np.loadtxt("stock_prices.csv")

env = StockTradingEnv(
    stock_data=prices,
    num_stocks=5,
    initial_balance=100000.0,
    transaction_cost=0.001,
    slippage=0.0005,
    max_position_per_stock=0.3,
)
```

**Features:**
- Multiple stock portfolio
- Technical indicators (RSI, MACD, Bollinger Bands)
- Transaction costs and slippage
- Position limits
- Sharpe ratio calculation
- Maximum drawdown tracking

### Train Trading Agent

```python
agent = PPOAgent(env=env, learning_rate=3e-4)
agent.train(total_timesteps=100000)

# Evaluate
results = agent.evaluate(episodes=10)
print(f"Mean return: {results['mean_reward']:.2%}")
```

### Custom Indicators

The environment automatically calculates:
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Price volatility bands
- **Momentum**: Price momentum

---

## Best Practices

### Offline RL
- Collect diverse datasets
- Use conservative algorithms (CQL, IQL)
- Validate on held-out data
- Monitor distribution shift

### Multi-Agent
- Start with simple scenarios
- Use centralized training
- Monitor agent interactions
- Balance cooperation vs competition

### Benchmarking
- Use multiple seeds (≥5)
- Report mean ± std
- Profile performance
- Track reproducibility

### Collaboration
- Share experiences regularly
- Monitor buffer diversity
- Track contributor statistics
- Verify model updates

## Troubleshooting

### Offline RL Issues
- **Poor performance**: Increase CQL weight or expectile
- **Slow training**: Reduce batch size or use smaller networks
- **Overfitting**: Add more diverse data

### Multi-Agent Issues
- **Agents not cooperating**: Adjust reward structure
- **Training unstable**: Reduce learning rates
- **Memory issues**: Reduce buffer size

### Integration Issues
- **HF upload fails**: Check token and repo permissions
- **Leaderboard conflicts**: Use unique config hashes
- **Collaboration sync**: Ensure shared storage access

## Next Steps

- Explore example scripts in `examples/`
- Join community discussions
- Contribute your models to HF Hub
- Share results on leaderboards
- Collaborate with other researchers
