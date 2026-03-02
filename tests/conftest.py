import os
import tempfile
import pytest
import yaml


@pytest.fixture()
def temp_config_file():
    cfg = {
        "env_name": "CartPole-v1",
        "algorithm": "ppo",
        "training": {
            "total_timesteps": 1000,
            "learning_rate": 3e-4,
            "batch_size": 32,
            "n_steps": 256,
            "n_epochs": 1,
            "seed": 42,
        },
        "experiment_name": "cli_test",
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
        yield path
