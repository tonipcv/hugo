from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    model: str = "llama3"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: int = 30


class TrainingConfig(BaseModel):
    total_timesteps: int = Field(default=100000, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0)
    batch_size: int = Field(default=64, gt=0)
    n_steps: int = Field(default=2048, gt=0)
    n_epochs: int = Field(default=10, gt=0)
    gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip_range: float = Field(default=0.2, gt=0)
    ent_coef: float = Field(default=0.01, ge=0)
    vf_coef: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    use_gpu: bool = True
    seed: Optional[int] = None
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 5000
    eval_episodes: int = 10


class RewardShapingConfig(BaseModel):
    enabled: bool = True
    llm_weight: float = Field(default=0.3, ge=0, le=1)
    env_weight: float = Field(default=0.7, ge=0, le=1)
    prompt_template: str = "default"
    cache_responses: bool = True
    max_cache_size: int = 10000


class Config(BaseModel):
    env_name: str
    algorithm: Literal["ppo", "dqn", "a2c"] = "ppo"
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    llm: Optional[LLMConfig] = None
    reward_shaping: Optional[RewardShapingConfig] = None
    output_dir: str = "./outputs"
    experiment_name: Optional[str] = None
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    class Config:
        extra = "allow"
