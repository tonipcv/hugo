import pytest
import numpy as np
from rl_llm_toolkit.rewards.llm_shaper import LLMRewardShaper
from rl_llm_toolkit.rewards.prompts import get_template
from tests.test_llm import MockLLMBackend


class TestLLMRewardShaper:
    def test_reward_shaping(self):
        llm = MockLLMBackend(model="test-model")
        shaper = LLMRewardShaper(
            llm_backend=llm,
            prompt_template="default",
            llm_weight=0.3,
            env_weight=0.7,
        )
        
        state = np.array([1.0, 2.0, 3.0])
        action = 0
        next_state = np.array([1.1, 2.1, 3.1])
        env_reward = 1.0
        
        shaped_reward, metadata = shaper.shape_reward(
            env_reward, state, action, next_state
        )
        
        assert isinstance(shaped_reward, float)
        assert "env_reward" in metadata
        assert "llm_reward" in metadata
        assert "shaped_reward" in metadata
        assert metadata["env_reward"] == env_reward
    
    def test_cache_functionality(self):
        llm = MockLLMBackend(model="test-model")
        shaper = LLMRewardShaper(
            llm_backend=llm,
            use_cache=True,
            cache_size=100,
        )
        
        state = np.array([1.0, 2.0, 3.0])
        action = 0
        next_state = np.array([1.1, 2.1, 3.1])
        
        _, metadata1 = shaper.shape_reward(1.0, state, action, next_state)
        assert metadata1["cached"] is False
        
        _, metadata2 = shaper.shape_reward(1.0, state, action, next_state)
        assert metadata2["cached"] is True
        
        assert shaper.cache_hit_rate == 0.5
    
    def test_weight_combination(self):
        llm = MockLLMBackend(model="test-model")
        shaper = LLMRewardShaper(
            llm_backend=llm,
            llm_weight=0.4,
            env_weight=0.6,
        )
        
        state = np.array([1.0])
        action = 0
        next_state = np.array([1.1])
        env_reward = 1.0
        
        shaped_reward, metadata = shaper.shape_reward(
            env_reward, state, action, next_state
        )
        
        expected = 0.6 * env_reward + 0.4 * metadata["llm_reward"]
        assert abs(shaped_reward - expected) < 1e-6
    
    def test_statistics(self):
        llm = MockLLMBackend(model="test-model")
        shaper = LLMRewardShaper(llm_backend=llm, use_cache=True)
        
        state = np.array([1.0])
        
        for i in range(5):
            shaper.shape_reward(1.0, state, i, state)
        
        stats = shaper.get_statistics()
        assert "cache_size" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "llm_stats" in stats
    
    def test_clear_cache(self):
        llm = MockLLMBackend(model="test-model")
        shaper = LLMRewardShaper(llm_backend=llm, use_cache=True)
        
        state = np.array([1.0])
        shaper.shape_reward(1.0, state, 0, state)
        
        assert shaper._cache_misses > 0
        
        shaper.clear_cache()
        
        assert len(shaper._cache) == 0
        assert shaper._cache_hits == 0
        assert shaper._cache_misses == 0
