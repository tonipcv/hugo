from typing import Optional, Dict, Any, Tuple
import numpy as np

from rl_llm_toolkit.vision.video_reasoner import VideoReasoningBackend


class VisualRewardShaper:
    """
    Reward shaping using visual reasoning.
    
    Combines environment rewards with vision-based assessments
    for better training signals in visual RL tasks.
    """
    
    def __init__(
        self,
        video_backend: VideoReasoningBackend,
        visual_weight: float = 0.3,
        env_weight: float = 0.7,
        use_cache: bool = True,
        cache_size: int = 1000,
    ):
        self.video_backend = video_backend
        self.visual_weight = visual_weight
        self.env_weight = env_weight
        self.use_cache = use_cache
        
        self.cache = {} if use_cache else None
        self.cache_size = cache_size
        
        self.stats = {
            "total_shapings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
    
    def shape_reward(
        self,
        env_reward: float,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        action: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Shape reward using visual analysis.
        
        Args:
            env_reward: Original environment reward
            before_frame: Frame before action
            after_frame: Frame after action
            action: Action taken
            context: Optional context information
            
        Returns:
            Tuple of (shaped_reward, metadata)
        """
        cache_key = self._get_cache_key(before_frame, after_frame, action)
        
        if self.use_cache and cache_key in self.cache:
            visual_reward = self.cache[cache_key]
            self.stats["cache_hits"] += 1
            cached = True
        else:
            action_names = context.get("action_names") if context else None
            
            visual_reward = self.video_backend.assess_action_quality(
                before_frame=before_frame,
                after_frame=after_frame,
                action=action,
                action_names=action_names,
            )
            
            visual_reward = (visual_reward - 0.5) * 2.0
            
            if self.use_cache:
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[cache_key] = visual_reward
            
            self.stats["cache_misses"] += 1
            cached = False
        
        shaped_reward = (
            self.env_weight * env_reward +
            self.visual_weight * visual_reward
        )
        
        self.stats["total_shapings"] += 1
        
        metadata = {
            "env_reward": env_reward,
            "visual_reward": visual_reward,
            "shaped_reward": shaped_reward,
            "cached": cached,
            "visual_weight": self.visual_weight,
            "env_weight": self.env_weight,
        }
        
        return shaped_reward, metadata
    
    def _get_cache_key(
        self,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        action: int,
    ) -> str:
        """Generate cache key from frames and action."""
        before_hash = hash(before_frame.tobytes())
        after_hash = hash(after_frame.tobytes())
        return f"{before_hash}_{after_hash}_{action}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get shaping statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            self.stats["cache_hits"] / total_requests
            if total_requests > 0 else 0.0
        )
        
        return {
            "total_shapings": self.stats["total_shapings"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache) if self.cache else 0,
            "video_backend_stats": self.video_backend.get_usage_stats(),
        }
    
    def clear_cache(self) -> None:
        """Clear reward cache."""
        if self.cache:
            self.cache.clear()
