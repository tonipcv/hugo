import pytest
from pathlib import Path
import tempfile
import json

from rl_llm_toolkit.integrations.leaderboard import Leaderboard
from rl_llm_toolkit.integrations.huggingface import HuggingFaceHub


class TestLeaderboard:
    def test_leaderboard_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_leaderboard.db"
            leaderboard = Leaderboard(db_path=db_path)
            
            assert db_path.exists()
    
    def test_submit_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_leaderboard.db"
            leaderboard = Leaderboard(db_path=db_path)
            
            submission_id = leaderboard.submit(
                env_name="CartPole-v1",
                algorithm="PPO",
                mean_reward=450.0,
                std_reward=50.0,
                total_timesteps=100000,
                hyperparameters={"lr": 0.0003},
                username="test_user",
            )
            
            assert submission_id > 0
    
    def test_get_leaderboard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_leaderboard.db"
            leaderboard = Leaderboard(db_path=db_path)
            
            leaderboard.submit(
                env_name="CartPole-v1",
                algorithm="PPO",
                mean_reward=450.0,
                std_reward=50.0,
                total_timesteps=100000,
                hyperparameters={"lr": 0.0003},
            )
            
            leaderboard.submit(
                env_name="CartPole-v1",
                algorithm="DQN",
                mean_reward=400.0,
                std_reward=60.0,
                total_timesteps=100000,
                hyperparameters={"lr": 0.0001},
            )
            
            results = leaderboard.get_leaderboard("CartPole-v1", limit=10)
            
            assert len(results) == 2
            assert results[0]["mean_reward"] >= results[1]["mean_reward"]
    
    def test_compare_algorithms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_leaderboard.db"
            leaderboard = Leaderboard(db_path=db_path)
            
            leaderboard.submit(
                env_name="CartPole-v1",
                algorithm="PPO",
                mean_reward=450.0,
                std_reward=50.0,
                total_timesteps=100000,
                hyperparameters={"lr": 0.0003},
            )
            
            leaderboard.submit(
                env_name="CartPole-v1",
                algorithm="PPO",
                mean_reward=460.0,
                std_reward=45.0,
                total_timesteps=100000,
                hyperparameters={"lr": 0.0005},
            )
            
            comparison = leaderboard.compare_algorithms("CartPole-v1")
            
            assert "PPO" in comparison
            assert comparison["PPO"]["best_reward"] == 460.0
            assert comparison["PPO"]["num_submissions"] == 2
    
    def test_export_leaderboard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_leaderboard.db"
            leaderboard = Leaderboard(db_path=db_path)
            
            leaderboard.submit(
                env_name="CartPole-v1",
                algorithm="PPO",
                mean_reward=450.0,
                std_reward=50.0,
                total_timesteps=100000,
                hyperparameters={"lr": 0.0003},
            )
            
            export_path = Path(tmpdir) / "leaderboard.json"
            leaderboard.export_leaderboard(
                env_name="CartPole-v1",
                output_path=export_path,
                format="json",
            )
            
            assert export_path.exists()
            
            with open(export_path) as f:
                data = json.load(f)
                assert len(data) == 1


class TestHuggingFaceHub:
    def test_hub_creation(self):
        hub = HuggingFaceHub()
        assert hub is not None
    
    def test_generate_model_card(self):
        hub = HuggingFaceHub()
        
        metadata = {
            "algorithm": "PPO",
            "env_name": "CartPole-v1",
            "mean_reward": 450.0,
            "total_timesteps": 100000,
        }
        
        card = hub._generate_model_card(metadata)
        
        assert "PPO" in card
        assert "CartPole-v1" in card
        assert "450.0" in card
    
    def test_create_leaderboard_entry(self):
        hub = HuggingFaceHub()
        
        entry = hub.create_leaderboard_entry(
            env_name="CartPole-v1",
            algorithm="PPO",
            mean_reward=450.0,
            std_reward=50.0,
            model_id="username/model",
        )
        
        assert entry["env_name"] == "CartPole-v1"
        assert entry["algorithm"] == "PPO"
        assert entry["mean_reward"] == 450.0
        assert entry["framework"] == "rl-llm-toolkit"
