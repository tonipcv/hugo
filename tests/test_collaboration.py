import pytest
from pathlib import Path
import tempfile

from rl_llm_toolkit.collaboration import CollaborationSession, SharedReplayBuffer


class TestCollaborationSession:
    def test_session_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = CollaborationSession(
                session_id="test_session",
                storage_dir=Path(tmpdir),
            )
            
            assert session.session_id == "test_session"
            assert session.storage_dir.exists()
    
    def test_join_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = CollaborationSession(
                session_id="test_session",
                storage_dir=Path(tmpdir),
            )
            
            success = session.join("agent_1", "PPO", metadata={"lr": 3e-4})
            assert success
            assert "agent_1" in session.participants
    
    def test_share_experience(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = CollaborationSession(
                session_id="test_session",
                storage_dir=Path(tmpdir),
            )
            
            session.join("agent_1", "PPO")
            
            experiences = [
                {"obs": [1, 2, 3], "action": 0, "reward": 1.0},
                {"obs": [4, 5, 6], "action": 1, "reward": 0.5},
            ]
            
            session.share_experience("agent_1", experiences)
            
            assert session.participants["agent_1"]["experiences_shared"] == 2
    
    def test_get_session_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = CollaborationSession(
                session_id="test_session",
                storage_dir=Path(tmpdir),
            )
            
            session.join("agent_1", "PPO")
            session.join("agent_2", "DQN")
            
            stats = session.get_session_stats()
            
            assert stats["num_participants"] == 2
            assert stats["session_id"] == "test_session"


class TestSharedReplayBuffer:
    def test_buffer_creation(self):
        buffer = SharedReplayBuffer(capacity=1000, min_size=100)
        assert buffer.capacity == 1000
        assert buffer.min_size == 100
    
    def test_add_experiences(self):
        buffer = SharedReplayBuffer(capacity=1000)
        
        experiences = [
            {"obs": [1, 2], "action": 0, "reward": 1.0},
            {"obs": [3, 4], "action": 1, "reward": 0.5},
        ]
        
        buffer.add(experiences, contributor_id="agent_1")
        
        assert buffer.size() == 2
        assert "agent_1" in buffer.stats["contributors"]
    
    def test_sample_experiences(self):
        buffer = SharedReplayBuffer(capacity=1000, min_size=5)
        
        experiences = [
            {"obs": [i, i+1], "action": i % 2, "reward": float(i)}
            for i in range(10)
        ]
        
        buffer.add(experiences, contributor_id="agent_1")
        
        sample = buffer.sample(batch_size=3)
        assert len(sample) == 3
    
    def test_exclude_contributor(self):
        buffer = SharedReplayBuffer(capacity=1000, min_size=5)
        
        exp1 = [{"obs": [1, 2], "action": 0} for _ in range(5)]
        exp2 = [{"obs": [3, 4], "action": 1} for _ in range(5)]
        
        buffer.add(exp1, contributor_id="agent_1")
        buffer.add(exp2, contributor_id="agent_2")
        
        sample = buffer.sample(batch_size=3, exclude_contributor="agent_1")
        
        for exp in sample:
            assert exp.get("contributor_id") != "agent_1"
    
    def test_buffer_stats(self):
        buffer = SharedReplayBuffer(capacity=1000)
        
        buffer.add([{"obs": [1, 2]}], contributor_id="agent_1")
        buffer.add([{"obs": [3, 4]}], contributor_id="agent_2")
        
        stats = buffer.get_stats()
        
        assert stats["size"] == 2
        assert stats["num_contributors"] == 2
        assert "agent_1" in stats["contributors"]
        assert "agent_2" in stats["contributors"]
