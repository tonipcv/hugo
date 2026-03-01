from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    tokens_used: int
    latency_ms: float
    metadata: Optional[Dict[str, Any]] = None


class LLMBackend(ABC):
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: int = 30,
        **kwargs: Any
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs
        self._total_tokens = 0
        self._total_requests = 0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> LLMResponse:
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> List[LLMResponse]:
        pass

    def get_usage_stats(self) -> Dict[str, int]:
        return {
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
            "avg_tokens_per_request": (
                self._total_tokens / self._total_requests if self._total_requests > 0 else 0
            ),
        }

    def reset_stats(self) -> None:
        self._total_tokens = 0
        self._total_requests = 0
