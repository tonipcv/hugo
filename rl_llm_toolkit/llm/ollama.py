from typing import Optional, Dict, Any, List
import time
import json
from rl_llm_toolkit.llm.base import LLMBackend, LLMResponse


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: int = 30,
        **kwargs: Any
    ):
        super().__init__(model, temperature, max_tokens, timeout, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError(
                    "Ollama package not installed. Install with: pip install ollama"
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
            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                }
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response["message"]["content"]
            
            tokens_used = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
            self._total_tokens += tokens_used
            self._total_requests += 1
            
            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                metadata={
                    "model": response.get("model"),
                    "eval_count": response.get("eval_count"),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                }
            )
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}")

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> List[LLMResponse]:
        return [self.generate(prompt, system_prompt, **kwargs) for prompt in prompts]
