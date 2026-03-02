from typing import List, Dict, Any, Optional
import numpy as np
from collections import deque
import threading


class SharedReplayBuffer:
    """
    Thread-safe shared replay buffer for collaborative training.
    
    Allows multiple agents to share experiences efficiently.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        min_size: int = 1000,
    ):
        self.capacity = capacity
        self.min_size = min_size
        
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
        
        self.stats = {
            "total_added": 0,
            "total_sampled": 0,
            "contributors": set(),
        }
    
    def add(
        self,
        experiences: List[Dict[str, Any]],
        contributor_id: Optional[str] = None,
    ) -> None:
        """
        Add experiences to shared buffer.
        
        Args:
            experiences: List of experience dictionaries
            contributor_id: ID of contributing agent
        """
        with self.lock:
            for exp in experiences:
                if contributor_id:
                    exp["contributor_id"] = contributor_id
                self.buffer.append(exp)
            
            self.stats["total_added"] += len(experiences)
            
            if contributor_id:
                self.stats["contributors"].add(contributor_id)
    
    def sample(
        self,
        batch_size: int,
        exclude_contributor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sample batch from shared buffer.
        
        Args:
            batch_size: Number of experiences to sample
            exclude_contributor: Exclude experiences from this contributor
            
        Returns:
            List of sampled experiences
        """
        with self.lock:
            if len(self.buffer) < self.min_size:
                return []
            
            available = list(self.buffer)
            
            if exclude_contributor:
                available = [
                    exp for exp in available
                    if exp.get("contributor_id") != exclude_contributor
                ]
            
            if len(available) < batch_size:
                return []
            
            indices = np.random.choice(len(available), batch_size, replace=False)
            batch = [available[i] for i in indices]
            
            self.stats["total_sampled"] += batch_size
            
            return batch
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return self.size() >= self.min_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            return {
                "size": len(self.buffer),
                "capacity": self.capacity,
                "total_added": self.stats["total_added"],
                "total_sampled": self.stats["total_sampled"],
                "num_contributors": len(self.stats["contributors"]),
                "contributors": list(self.stats["contributors"]),
            }
    
    def clear(self) -> None:
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()
            self.stats = {
                "total_added": 0,
                "total_sampled": 0,
                "contributors": set(),
            }
