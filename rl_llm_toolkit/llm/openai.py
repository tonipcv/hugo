from typing import Optional, Dict, Any, List
import time
from rl_llm_toolkit.llm.base import LLMBackend, LLMResponse


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: int = 30,
        **kwargs: Any
    ):
        super().__init__(model, temperature, max_tokens, timeout, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout
                )
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> LLMResponse:
        client = self._get_client()
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            
            tokens_used = response.usage.total_tokens if response.usage else 0
            self._total_tokens += tokens_used
            self._total_requests += 1
            
            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                metadata={
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                }
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> List[LLMResponse]:
        return [self.generate(prompt, system_prompt, **kwargs) for prompt in prompts]
