from typing import Dict, List, Any, Optional
import json
import time
from pathlib import Path
from datetime import datetime
import threading
import queue


class CollaborationSession:
    """
    Real-time collaboration session for distributed RL training.
    
    Features:
    - Share training experiences across agents
    - Synchronize model parameters
    - Track collaborative metrics
    - Enable remote training coordination
    """
    
    def __init__(
        self,
        session_id: str,
        storage_dir: Optional[Path] = None,
    ):
        self.session_id = session_id
        
        if storage_dir is None:
            storage_dir = Path.home() / ".rl_llm_toolkit" / "sessions"
        
        self.storage_dir = Path(storage_dir) / session_id
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.participants = {}
        self.message_queue = queue.Queue()
        self.running = False
        
        self._init_session()
    
    def _init_session(self) -> None:
        """Initialize session metadata."""
        self.metadata = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "participants": [],
            "total_experiences": 0,
            "total_updates": 0,
        }
        
        self._save_metadata()
    
    def _save_metadata(self) -> None:
        """Save session metadata to disk."""
        metadata_path = self.storage_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def join(
        self,
        participant_id: str,
        agent_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Join collaboration session.
        
        Args:
            participant_id: Unique identifier for participant
            agent_type: Type of agent (e.g., "PPO", "DQN")
            metadata: Additional participant metadata
            
        Returns:
            True if successfully joined
        """
        if participant_id in self.participants:
            print(f"⚠️  Participant {participant_id} already in session")
            return False
        
        participant_info = {
            "id": participant_id,
            "agent_type": agent_type,
            "joined_at": datetime.now().isoformat(),
            "experiences_shared": 0,
            "updates_received": 0,
            "metadata": metadata or {},
        }
        
        self.participants[participant_id] = participant_info
        self.metadata["participants"].append(participant_info)
        self._save_metadata()
        
        print(f"✅ {participant_id} joined session {self.session_id}")
        return True
    
    def share_experience(
        self,
        participant_id: str,
        experiences: List[Dict[str, Any]],
    ) -> None:
        """
        Share training experiences with other participants.
        
        Args:
            participant_id: ID of sharing participant
            experiences: List of experience dictionaries
        """
        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not in session")
        
        experience_file = self.storage_dir / f"experiences_{participant_id}_{int(time.time())}.json"
        
        with open(experience_file, "w") as f:
            json.dump(experiences, f)
        
        self.participants[participant_id]["experiences_shared"] += len(experiences)
        self.metadata["total_experiences"] += len(experiences)
        self._save_metadata()
        
        self.message_queue.put({
            "type": "experience_shared",
            "participant_id": participant_id,
            "count": len(experiences),
            "file": str(experience_file),
        })
    
    def get_shared_experiences(
        self,
        participant_id: str,
        max_count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get experiences shared by other participants.
        
        Args:
            participant_id: ID of requesting participant
            max_count: Maximum number of experiences to return
            
        Returns:
            List of shared experiences
        """
        all_experiences = []
        
        for exp_file in self.storage_dir.glob("experiences_*.json"):
            if participant_id not in exp_file.name:
                with open(exp_file) as f:
                    experiences = json.load(f)
                    all_experiences.extend(experiences)
        
        if max_count and len(all_experiences) > max_count:
            import random
            all_experiences = random.sample(all_experiences, max_count)
        
        return all_experiences
    
    def share_model_update(
        self,
        participant_id: str,
        model_path: Path,
        metrics: Dict[str, float],
    ) -> None:
        """
        Share model update with session.
        
        Args:
            participant_id: ID of sharing participant
            model_path: Path to model checkpoint
            metrics: Performance metrics
        """
        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not in session")
        
        update_info = {
            "participant_id": participant_id,
            "timestamp": datetime.now().isoformat(),
            "model_path": str(model_path),
            "metrics": metrics,
        }
        
        updates_file = self.storage_dir / "model_updates.jsonl"
        with open(updates_file, "a") as f:
            f.write(json.dumps(update_info) + "\n")
        
        self.metadata["total_updates"] += 1
        self._save_metadata()
        
        self.message_queue.put({
            "type": "model_update",
            "participant_id": participant_id,
            "metrics": metrics,
        })
    
    def get_best_model(self, metric: str = "mean_reward") -> Optional[Dict[str, Any]]:
        """
        Get information about the best performing model.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Best model information or None
        """
        updates_file = self.storage_dir / "model_updates.jsonl"
        
        if not updates_file.exists():
            return None
        
        best_update = None
        best_value = float('-inf')
        
        with open(updates_file) as f:
            for line in f:
                update = json.loads(line)
                value = update["metrics"].get(metric, float('-inf'))
                
                if value > best_value:
                    best_value = value
                    best_update = update
        
        return best_update
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "num_participants": len(self.participants),
            "total_experiences": self.metadata["total_experiences"],
            "total_updates": self.metadata["total_updates"],
            "participants": list(self.participants.values()),
        }
    
    def leave(self, participant_id: str) -> None:
        """Leave collaboration session."""
        if participant_id in self.participants:
            del self.participants[participant_id]
            print(f"👋 {participant_id} left session {self.session_id}")
