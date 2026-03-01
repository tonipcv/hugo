from typing import Optional, Dict, Any, Tuple
import re
import numpy as np
from collections import OrderedDict
from rl_llm_toolkit.llm.base import LLMBackend
from rl_llm_toolkit.rewards.prompts import PromptTemplate, get_template


class LLMRewardShaper:
    def __init__(
        self,
        llm_backend: LLMBackend,
        prompt_template: str = "default",
        llm_weight: float = 0.3,
        env_weight: float = 0.7,
        cache_size: int = 10000,
        use_cache: bool = True,
    ):
        self.llm = llm_backend
        self.template = get_template(prompt_template) if isinstance(prompt_template, str) else prompt_template
        self.llm_weight = llm_weight
        self.env_weight = env_weight
        self.use_cache = use_cache
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def shape_reward(
        self,
        env_reward: float,
        state: np.ndarray,
        action: Any,
        next_state: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        context = context or {}
        
        cache_key = self._get_cache_key(state, action, next_state)
        
        if self.use_cache and cache_key in self._cache:
            self._cache_hits += 1
            llm_reward = self._cache[cache_key]
            metadata = {"cached": True, "cache_hit_rate": self.cache_hit_rate}
        else:
            self._cache_misses += 1
            llm_reward = self._get_llm_reward(env_reward, state, action, next_state, context)
            
            if self.use_cache:
                self._add_to_cache(cache_key, llm_reward)
            
            metadata = {
                "cached": False,
                "cache_hit_rate": self.cache_hit_rate,
                "llm_tokens": self.llm.get_usage_stats()["total_tokens"],
            }
        
        shaped_reward = (self.env_weight * env_reward) + (self.llm_weight * llm_reward)
        
        metadata.update({
            "env_reward": env_reward,
            "llm_reward": llm_reward,
            "shaped_reward": shaped_reward,
            "llm_weight": self.llm_weight,
            "env_weight": self.env_weight,
        })
        
        return shaped_reward, metadata

    def _get_llm_reward(
        self,
        env_reward: float,
        state: np.ndarray,
        action: Any,
        next_state: np.ndarray,
        context: Dict[str, Any],
    ) -> float:
        prompt = self._format_prompt(env_reward, state, action, next_state, context)
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.template.system_prompt,
                temperature=0.3,
                max_tokens=10,
            )
            
            reward = self._parse_reward(response.content)
            return np.clip(reward, -1.0, 1.0)
        
        except Exception as e:
            print(f"Warning: LLM reward generation failed: {e}. Using env reward.")
            return np.clip(env_reward, -1.0, 1.0)

    def _format_prompt(
        self,
        env_reward: float,
        state: np.ndarray,
        action: Any,
        next_state: np.ndarray,
        context: Dict[str, Any],
    ) -> str:
        prompt_vars = {
            "env_reward": env_reward,
            "prev_state": self._format_state(state),
            "new_state": self._format_state(next_state),
            "action": action,
            **context,
        }
        
        return self.template.format(**prompt_vars)

    def _format_state(self, state: np.ndarray) -> str:
        if len(state) <= 10:
            return str(state.tolist())
        return f"[{state[0]:.3f}, ..., {state[-1]:.3f}] (dim={len(state)})"

    def _parse_reward(self, response: str) -> float:
        response = response.strip()
        
        number_match = re.search(r'-?\d+\.?\d*', response)
        if number_match:
            return float(number_match.group())
        
        return 0.0

    def _get_cache_key(self, state: np.ndarray, action: Any, next_state: np.ndarray) -> str:
        state_hash = hash(state.tobytes())
        action_hash = hash(str(action))
        next_state_hash = hash(next_state.tobytes())
        return f"{state_hash}_{action_hash}_{next_state_hash}"

    def _add_to_cache(self, key: str, value: float) -> None:
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = value

    @property
    def cache_hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "llm_stats": self.llm.get_usage_stats(),
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
