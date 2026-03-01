import pytest
from rl_llm_toolkit.core.config import Config, TrainingConfig, LLMConfig, RewardShapingConfig


class TestConfig:
    def test_training_config_defaults(self):
        config = TrainingConfig()
        assert config.total_timesteps == 100000
        assert config.learning_rate == 3e-4
        assert config.batch_size == 64
        assert config.gamma == 0.99
    
    def test_training_config_custom(self):
        config = TrainingConfig(
            total_timesteps=50000,
            learning_rate=1e-4,
            batch_size=32,
        )
        assert config.total_timesteps == 50000
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
    
    def test_llm_config_defaults(self):
        config = LLMConfig()
        assert config.provider == "ollama"
        assert config.model == "llama3"
        assert config.temperature == 0.7
    
    def test_llm_config_custom(self):
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.5,
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.api_key == "test-key"
        assert config.temperature == 0.5
    
    def test_reward_shaping_config(self):
        config = RewardShapingConfig(
            enabled=True,
            llm_weight=0.4,
            env_weight=0.6,
        )
        assert config.enabled is True
        assert config.llm_weight == 0.4
        assert config.env_weight == 0.6
    
    def test_full_config(self):
        config = Config(
            env_name="CartPole-v1",
            algorithm="ppo",
            training=TrainingConfig(total_timesteps=50000),
            llm=LLMConfig(provider="ollama"),
            reward_shaping=RewardShapingConfig(enabled=True),
        )
        assert config.env_name == "CartPole-v1"
        assert config.algorithm == "ppo"
        assert config.training.total_timesteps == 50000
        assert config.llm.provider == "ollama"
        assert config.reward_shaping.enabled is True
    
    def test_config_validation(self):
        with pytest.raises(ValueError):
            TrainingConfig(total_timesteps=-1)
        
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0)
