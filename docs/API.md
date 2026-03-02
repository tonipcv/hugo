# API Reference

Complete API documentation for RL-LLM Toolkit v0.2.0

## Core Components

### RLEnvironment

Wrapper for Gymnasium environments with additional features.

```python
from rl_llm_toolkit import RLEnvironment

env = RLEnvironment("CartPole-v1")
obs, info = env.reset(seed=42)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

**Methods:**
- `reset(seed=None, options=None)` - Reset environment
- `step(action)` - Take action in environment
- `close()` - Close environment
- `render()` - Render environment (if supported)

---

## Agents

### PPOAgent

Proximal Policy Optimization agent.

```python
from rl_llm_toolkit import PPOAgent, RLEnvironment

env = RLEnvironment("CartPole-v1")
agent = PPOAgent(
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    seed=42
)

# Train
results = agent.train(total_timesteps=100000)

# Evaluate
eval_results = agent.evaluate(episodes=10, deterministic=True)

# Predict
action, info = agent.predict(observation, deterministic=True)

# Save/Load
agent.save("model.pt")
agent.load("model.pt")
```

**Parameters:**
- `env` (RLEnvironment): Training environment
- `learning_rate` (float): Learning rate (default: 3e-4)
- `n_steps` (int): Steps per update (default: 2048)
- `batch_size` (int): Minibatch size (default: 64)
- `n_epochs` (int): Optimization epochs (default: 10)
- `gamma` (float): Discount factor (default: 0.99)
- `gae_lambda` (float): GAE lambda (default: 0.95)
- `clip_range` (float): PPO clip range (default: 0.2)
- `seed` (int): Random seed

### DQNAgent

Deep Q-Network agent.

```python
from rl_llm_toolkit import DQNAgent, RLEnvironment

env = RLEnvironment("CartPole-v1")
agent = DQNAgent(
    env=env,
    learning_rate=1e-4,
    buffer_size=50000,
    batch_size=32,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=1000,
    target_update_freq=1000,
    seed=42
)
```

**Parameters:**
- `env` (RLEnvironment): Training environment
- `learning_rate` (float): Learning rate (default: 1e-4)
- `buffer_size` (int): Replay buffer size (default: 50000)
- `batch_size` (int): Minibatch size (default: 32)
- `gamma` (float): Discount factor (default: 0.99)
- `epsilon_start` (float): Initial epsilon (default: 1.0)
- `epsilon_end` (float): Final epsilon (default: 0.01)
- `epsilon_decay` (int): Epsilon decay steps (default: 1000)
- `target_update_freq` (int): Target network update frequency (default: 1000)

### CQLAgent

Conservative Q-Learning for offline RL.

```python
from rl_llm_toolkit.agents.cql import CQLAgent

agent = CQLAgent(
    env=env,
    learning_rate=3e-4,
    batch_size=256,
    cql_weight=1.0,
    seed=42
)

# Collect dataset
dataset = agent.collect_dataset(num_episodes=100, policy="random")

# Load dataset
agent.load_dataset(dataset)

# Train on offline data
agent.train(total_timesteps=50000)
```

**Parameters:**
- `cql_weight` (float): Conservative penalty weight (default: 1.0)

### IQLAgent

Implicit Q-Learning for offline RL.

```python
from rl_llm_toolkit.agents.iql import IQLAgent

agent = IQLAgent(
    env=env,
    learning_rate=3e-4,
    batch_size=256,
    expectile=0.7,
    temperature=3.0,
    seed=42
)
```

**Parameters:**
- `expectile` (float): Expectile for value learning (default: 0.7)
- `temperature` (float): Temperature for policy extraction (default: 3.0)

### MADDPGAgent

Multi-Agent Deep Deterministic Policy Gradient.

```python
from rl_llm_toolkit.multiagent import MADDPGAgent, CooperativeNavigationEnv

env = CooperativeNavigationEnv(num_agents=3, num_landmarks=3)
agent = MADDPGAgent(
    env=env,
    num_agents=3,
    learning_rate_actor=1e-4,
    learning_rate_critic=1e-3,
    gamma=0.95,
    tau=0.01,
    seed=42
)
```

---

## LLM Integration

### LLMRewardShaper

Shape rewards using LLM feedback.

```python
from rl_llm_toolkit import LLMRewardShaper
from rl_llm_toolkit.llm import OllamaBackend

llm = OllamaBackend(model="llama2")
shaper = LLMRewardShaper(
    llm_backend=llm,
    prompt_template="default",
    llm_weight=0.3,
    env_weight=0.7,
    use_cache=True,
    cache_size=10000
)

shaped_reward, metadata = shaper.shape_reward(
    env_reward=1.0,
    state=state,
    action=action,
    next_state=next_state,
    context={"env_name": "CartPole-v1"}
)
```

### OllamaBackend

Local LLM backend using Ollama.

```python
from rl_llm_toolkit.llm import OllamaBackend

llm = OllamaBackend(
    model="llama2",
    base_url="http://localhost:11434",
    temperature=0.7
)

response = llm.generate("Evaluate this action...")
```

### OpenAIBackend

OpenAI API backend.

```python
from rl_llm_toolkit.llm import OpenAIBackend

llm = OpenAIBackend(
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7
)
```

---

## Benchmarking

### BenchmarkSuite

Comprehensive benchmarking for RL algorithms.

```python
from rl_llm_toolkit.benchmarks import BenchmarkSuite

suite = BenchmarkSuite(output_dir="./benchmarks")

result = suite.run_benchmark(
    agent_factory=lambda env, seed: PPOAgent(env=env, seed=seed),
    env_name="CartPole-v1",
    num_seeds=5,
    total_timesteps=100000,
    eval_episodes=20
)

suite.export_results("results.json", format="json")
```

### PerformanceMetrics

Calculate performance metrics.

```python
from rl_llm_toolkit.benchmarks import PerformanceMetrics

stability = PerformanceMetrics.calculate_stability(rewards, window=10)
sample_efficiency = PerformanceMetrics.calculate_sample_efficiency(
    rewards, threshold=450, timesteps=timesteps
)
robustness = PerformanceMetrics.calculate_robustness(
    seed_results, metric_key="mean_reward"
)
```

---

## Collaboration

### CollaborationSession

Manage collaborative training sessions.

```python
from rl_llm_toolkit.collaboration import CollaborationSession

session = CollaborationSession(session_id="my_session")
session.join("agent_1", "PPO", metadata={"lr": 3e-4})

# Share experiences
session.share_experience("agent_1", experiences)

# Get shared experiences
shared = session.get_shared_experiences("agent_2", max_count=1000)

# Share model update
session.share_model_update(
    participant_id="agent_1",
    model_path="./model.pt",
    metrics={"mean_reward": 450.0}
)

# Get best model
best = session.get_best_model(metric="mean_reward")
```

### SharedReplayBuffer

Thread-safe shared replay buffer.

```python
from rl_llm_toolkit.collaboration import SharedReplayBuffer

buffer = SharedReplayBuffer(capacity=100000, min_size=1000)

# Add experiences
buffer.add(experiences, contributor_id="agent_1")

# Sample (excluding own contributions)
batch = buffer.sample(batch_size=64, exclude_contributor="agent_1")

# Check if ready
if buffer.is_ready():
    # Buffer has enough samples
    pass
```

---

## Integrations

### HuggingFaceHub

Upload and download models from Hugging Face.

```python
from rl_llm_toolkit.integrations import HuggingFaceHub

hub = HuggingFaceHub(token="your_hf_token")

# Upload model
url = hub.upload_model(
    model_path="./model.pt",
    repo_id="username/cartpole-ppo",
    metadata={"algorithm": "PPO", "mean_reward": 475.0}
)

# Download model
model_path = hub.download_model("username/cartpole-ppo")

# List models
models = hub.list_models(filter_tag="rl-llm-toolkit", limit=20)
```

### Leaderboard

Track and compare performance.

```python
from rl_llm_toolkit.integrations.leaderboard import Leaderboard

leaderboard = Leaderboard(db_path="./leaderboard.db")

# Submit results
submission_id = leaderboard.submit(
    env_name="CartPole-v1",
    algorithm="PPO",
    mean_reward=475.0,
    std_reward=25.0,
    total_timesteps=100000,
    hyperparameters={"lr": 3e-4},
    username="your_name"
)

# Get leaderboard
top_10 = leaderboard.get_leaderboard("CartPole-v1", limit=10)

# Export
leaderboard.export_leaderboard(
    env_name="CartPole-v1",
    output_path="./leaderboard.md",
    format="markdown"
)
```

---

## Vision

### VideoReasoningBackend

Analyze visual RL environments.

```python
from rl_llm_toolkit.vision import VideoReasoningBackend

video_backend = VideoReasoningBackend(
    model="gpt-4-vision",
    max_frames=4
)

# Analyze frame
analysis = video_backend.analyze_frame(
    frame=rgb_frame,
    context="CartPole balancing task"
)

# Analyze trajectory
trajectory_analysis = video_backend.analyze_trajectory(
    frames=[frame1, frame2, frame3],
    actions=[0, 1, 0],
    context="Navigate to goal"
)
```

### VisualRewardShaper

Shape rewards using visual analysis.

```python
from rl_llm_toolkit.vision import VisualRewardShaper, VideoReasoningBackend

video_backend = VideoReasoningBackend(model="gpt-4-vision")
visual_shaper = VisualRewardShaper(
    video_backend=video_backend,
    visual_weight=0.3,
    env_weight=0.7
)

shaped_reward, metadata = visual_shaper.shape_reward(
    env_reward=reward,
    before_frame=frame_before,
    after_frame=frame_after,
    action=action
)
```

---

## Environments

### CryptoTradingEnv

Cryptocurrency trading environment.

```python
from rl_llm_toolkit.environments.trading import CryptoTradingEnv

env = CryptoTradingEnv(
    initial_balance=10000.0,
    transaction_fee=0.001,
    max_position=1.0
)
```

### StockTradingEnv

Advanced stock trading with multiple stocks.

```python
from rl_llm_toolkit.environments.stock_trading import StockTradingEnv

env = StockTradingEnv(
    num_stocks=5,
    initial_balance=100000.0,
    transaction_cost=0.001,
    slippage=0.0005,
    max_position_per_stock=0.3
)
```

### CooperativeNavigationEnv

Multi-agent cooperative navigation.

```python
from rl_llm_toolkit.multiagent import CooperativeNavigationEnv

env = CooperativeNavigationEnv(
    num_agents=3,
    num_landmarks=3,
    world_size=2.0
)
```

---

## Utilities

### Training Statistics

All agents provide training statistics:

```python
stats = agent.get_training_stats()

print(stats["total_timesteps"])
print(stats["stats"]["episode_rewards"])
print(stats["stats"]["losses"])
```

### Evaluation

Standard evaluation interface:

```python
results = agent.evaluate(
    episodes=10,
    deterministic=True,
    render=False
)

print(f"Mean: {results['mean_reward']}")
print(f"Std: {results['std_reward']}")
print(f"Min: {results['min_reward']}")
print(f"Max: {results['max_reward']}")
```

---

## Version Information

```python
import rl_llm_toolkit
print(rl_llm_toolkit.__version__)  # 0.2.0
```

---

For more examples, see the `examples/` directory and Jupyter notebooks in `examples/notebooks/`.
