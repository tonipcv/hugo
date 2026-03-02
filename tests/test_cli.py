import os
from pathlib import Path
from click.testing import CliRunner

from rl_llm_toolkit import __version__
from rl_llm_toolkit.cli.main import cli


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])   # e.g., 'cli, version 0.2.0'
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_list_commands():
    runner = CliRunner()

    # list-envs
    r1 = runner.invoke(cli, ["list-envs"])
    assert r1.exit_code == 0
    assert "CryptoTradingEnv" in r1.output
    assert "GridWorldEnv" in r1.output

    # list-agents
    r2 = runner.invoke(cli, ["list-agents"])
    assert r2.exit_code == 0
    assert "PPOAgent" in r2.output
    assert "DQNAgent" in r2.output

    # list-backends
    r3 = runner.invoke(cli, ["list-backends"])
    assert r3.exit_code == 0
    assert "OllamaBackend" in r3.output
    assert "OpenAIBackend" in r3.output


def test_quickstart_creates_config(tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["quickstart", "-e", "CartPole-v1", "-a", "ppo", "--no-llm"])
        assert result.exit_code == 0
        # File should be created in CWD
        cfg = Path("config_quickstart.yaml")
        assert cfg.exists(), f"Expected quickstart file at {cfg}"
        # Output hint should mention 'hugo train'
        assert "hugo train" in result.output


def test_train_dry_run(temp_config_file: str, tmp_path: Path):
    runner = CliRunner()
    # Use isolated filesystem to avoid writing logs/model
    with runner.isolated_filesystem():
        # Copy provided temp config into CWD so relative paths work
        cfg_path = Path("config.yaml")
        cfg_path.write_text(Path(temp_config_file).read_text())
        result = runner.invoke(cli, ["train", str(cfg_path), "--dry-run"]) 
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "configuration validated" in result.output
