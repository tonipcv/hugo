# API Reference

Complete API documentation for RL-LLM Toolkit.

## Core Components

### RLEnvironment

Wrapper around Gymnasium environments with additional tracking.

```python
class RLEnvironment:
    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        **kwargs: Any
    )
```

**Parameters:**
- `env_id`: Gymnasium environment ID (e.g., "CartPole-v1")
- `render_mode`: Rendering mode ("human", "rgb_array", None)
- `**kwargs`: Additional arguments passed to gym.make()

**Methods:**
- `reset(seed, options)`: Reset environment
- `step(action)`: Take action in environment
- `close()`: Close environment
- `render()`: Render current state
- `episode_statistics`: Get episode statistics

**Properties:**
- `observation_space`: Environment observation space
- `action_space`: Environment action space

## Agents

### PPOAgent

Proximal Policy Optimization agent.

```python
class PPOAgent(BaseAgent):
    def __init__(
        self,
        env: RLEnvironment,
        reward_shaper: Optional[LLMRewardShaper] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        device: Optional[str] = None,
        seed: Optional[int] = None,
    )
```

**Key Methods:**
- `train(total_timesteps, log_interval, eval_interval)`: Train agent
- `predict(observation, deterministic)`: Get action for observation
- `evaluate(episodes, render, deterministic)`: Evaluate agent performance
- `save(path)`: Save model to disk
- `load(path)`: Load model from disk

### DQNAgent

Deep Q-Network agent.

```python
class DQNAgent(BaseAgent):
    def __init__(
        self,
        env: RLEnvironment,
        reward_shaper: Optional[LLMRewardShaper] = None,
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        batch_size: int = 32,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10000,
        target_update_freq: int = 1000,
        learning_starts: int = 1000,
        train_freq: int = 4,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        device: Optional[str] = None,
        seed: Optional[int] = None,
    )
```

## LLM Backends

### OllamaBackend

Local LLM backend using Ollama.

```python
class OllamaBackend(LLMBackend):
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: int = 30,
    )
```

**Methods:**
- `generate(prompt, system_prompt)`: Generate single response
- `generate_batch(prompts, system_prompt)`: Generate batch responses
- `get_usage_stats()`: Get token usage statistics

### OpenAIBackend

OpenAI API backend.

```python
class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: int = 30,
    )
```

## Reward Shaping

### LLMRewardShaper

LLM-based reward shaping.

```python
class LLMRewardShaper:
    def __init__(
        self,
        llm_backend: LLMBackend,
        prompt_template: str = "default",
        llm_weight: float = 0.3,
        env_weight: float = 0.7,
        cache_size: int = 10000,
        use_cache: bool = True,
    )
```

**Methods:**
- `shape_reward(env_reward, state, action, next_state, context)`: Shape reward
- `get_statistics()`: Get caching and usage statistics
- `clear_cache()`: Clear reward cache

**Properties:**
- `cache_hit_rate`: Percentage of cached responses

## Configuration

### Config

Main configuration class.

```python
class Config(BaseModel):
    env_name: str
    algorithm: Literal["ppo", "dqn", "a2c"] = "ppo"
    training: TrainingConfig
    llm: Optional[LLMConfig] = None
    reward_shaping: Optional[RewardShapingConfig] = None
    output_dir: str = "./outputs"
    experiment_name: Optional[str] = None
    use_wandb: bool = False
```

### TrainingConfig

Training hyperparameters.

```python
class TrainingConfig(BaseModel):
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_gpu: bool = True
    seed: Optional[int] = None
```

## Environments

### CryptoTradingEnv

Cryptocurrency trading environment.

```python
class CryptoTradingEnv(gym.Env):
    def __init__(
        self,
        price_data: Optional[np.ndarray] = None,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        max_position: float = 1.0,
        render_mode: Optional[str] = None,
    )
```

**Action Space:**
- 0: Sell (close position)
- 1: Hold
- 2: Buy (open position)

**Observation Space:**
- Current price (normalized)
- Balance (normalized)
- Position size (normalized)
- Portfolio value (normalized)
- Recent price returns (5 timesteps)

## Utilities

### Logger

Training logger with TensorBoard and W&B support.

```python
class Logger:
    def __init__(
        self,
        log_dir: Path,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
    )
```

**Methods:**
- `log_metrics(metrics, step)`: Log metrics
- `log_hyperparameters(params)`: Log hyperparameters
- `save_config(config)`: Save configuration
- `close()`: Close logger

### Visualization

```python
def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Training Curves",
) -> None

def save_video(
    frames: List[np.ndarray],
    save_path: Path,
    fps: int = 30,
) -> None
```

## CLI Commands

### train

Train an agent from configuration file.

```bash
rl-llm train CONFIG_FILE [OPTIONS]

Options:
  --output-dir, -o TEXT  Output directory
  --device, -d TEXT      Device (cpu/cuda)
```

### evaluate

Evaluate a trained model.

```bash
rl-llm evaluate MODEL_PATH ENV_NAME [OPTIONS]

Options:
  --episodes, -n INTEGER  Number of episodes
  --render/--no-render    Render environment
```

### quickstart

Generate quickstart configuration.

```bash
rl-llm quickstart [OPTIONS]

Options:
  --env, -e TEXT         Environment name
  --algorithm, -a TEXT   Algorithm (ppo/dqn)
  --llm/--no-llm        Use LLM reward shaping
```

## Type Hints

All public APIs include comprehensive type hints for better IDE support and type checking.

```python
from typing import Optional, Dict, Any, Tuple
import numpy as np

def predict(
    observation: np.ndarray, 
    deterministic: bool = True
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    ...
```
