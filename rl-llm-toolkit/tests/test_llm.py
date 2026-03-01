import pytest
from unittest.mock import Mock, patch
from rl_llm_toolkit.llm.base import LLMBackend, LLMResponse


class MockLLMBackend(LLMBackend):
    def generate(self, prompt, system_prompt=None, **kwargs):
        self._total_requests += 1
        self._total_tokens += 100
        return LLMResponse(
            content="0.5",
            tokens_used=100,
            latency_ms=50.0,
        )
    
    def generate_batch(self, prompts, system_prompt=None, **kwargs):
        return [self.generate(p, system_prompt, **kwargs) for p in prompts]


class TestLLMBackend:
    def test_mock_backend(self):
        backend = MockLLMBackend(model="test-model")
        
        response = backend.generate("test prompt")
        
        assert isinstance(response, LLMResponse)
        assert response.content == "0.5"
        assert response.tokens_used == 100
        assert response.latency_ms == 50.0
    
    def test_usage_stats(self):
        backend = MockLLMBackend(model="test-model")
        
        backend.generate("prompt 1")
        backend.generate("prompt 2")
        backend.generate("prompt 3")
        
        stats = backend.get_usage_stats()
        assert stats["total_requests"] == 3
        assert stats["total_tokens"] == 300
        assert stats["avg_tokens_per_request"] == 100
    
    def test_reset_stats(self):
        backend = MockLLMBackend(model="test-model")
        
        backend.generate("prompt")
        backend.reset_stats()
        
        stats = backend.get_usage_stats()
        assert stats["total_requests"] == 0
        assert stats["total_tokens"] == 0
    
    def test_batch_generation(self):
        backend = MockLLMBackend(model="test-model")
        
        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        responses = backend.generate_batch(prompts)
        
        assert len(responses) == 3
        assert all(isinstance(r, LLMResponse) for r in responses)
