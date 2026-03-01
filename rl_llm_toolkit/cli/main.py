import click
from pathlib import Path
import yaml
from rich.console import Console
from rich.table import Table

from rl_llm_toolkit.core.config import Config
from rl_llm_toolkit.core.environment import RLEnvironment
from rl_llm_toolkit.agents.ppo import PPOAgent
from rl_llm_toolkit.agents.dqn import DQNAgent
from rl_llm_toolkit.llm.ollama import OllamaBackend
from rl_llm_toolkit.llm.openai import OpenAIBackend
from rl_llm_toolkit.rewards.llm_shaper import LLMRewardShaper
from rl_llm_toolkit.utils.logger import Logger

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="./outputs", help="Output directory")
@click.option("--device", "-d", default=None, help="Device (cpu/cuda)")
def train(config_file: str, output_dir: str, device: str) -> None:
    console.print("[bold green]Starting training...[/bold green]")
    
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(**config_dict)
    
    env = RLEnvironment(config.env_name)
    
    reward_shaper = None
    if config.llm and config.reward_shaping and config.reward_shaping.enabled:
        if config.llm.provider == "ollama":
            llm = OllamaBackend(
                model=config.llm.model,
                base_url=config.llm.base_url or "http://localhost:11434",
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            )
        elif config.llm.provider == "openai":
            llm = OpenAIBackend(
                model=config.llm.model,
                api_key=config.llm.api_key,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm.provider}")
        
        reward_shaper = LLMRewardShaper(
            llm_backend=llm,
            prompt_template=config.reward_shaping.prompt_template,
            llm_weight=config.reward_shaping.llm_weight,
            env_weight=config.reward_shaping.env_weight,
            use_cache=config.reward_shaping.cache_responses,
            cache_size=config.reward_shaping.max_cache_size,
        )
    
    if config.algorithm == "ppo":
        agent = PPOAgent(
            env=env,
            reward_shaper=reward_shaper,
            learning_rate=config.training.learning_rate,
            n_steps=config.training.n_steps,
            batch_size=config.training.batch_size,
            n_epochs=config.training.n_epochs,
            gamma=config.training.gamma,
            gae_lambda=config.training.gae_lambda,
            clip_range=config.training.clip_range,
            ent_coef=config.training.ent_coef,
            vf_coef=config.training.vf_coef,
            max_grad_norm=config.training.max_grad_norm,
            device=device,
            seed=config.training.seed,
        )
    elif config.algorithm == "dqn":
        agent = DQNAgent(
            env=env,
            reward_shaper=reward_shaper,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            gamma=config.training.gamma,
            device=device,
            seed=config.training.seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    
    logger = Logger(
        log_dir=Path(output_dir),
        experiment_name=config.experiment_name,
        use_tensorboard=True,
        use_wandb=config.use_wandb,
        wandb_config={
            "project": config.wandb_project,
            "entity": config.wandb_entity,
            "config": config_dict,
        } if config.use_wandb else None,
    )
    
    logger.save_config(config_dict)
    
    results = agent.train(
        total_timesteps=config.training.total_timesteps,
        log_interval=config.training.log_interval,
        eval_interval=config.training.eval_interval,
        eval_episodes=config.training.eval_episodes,
    )
    
    model_path = Path(output_dir) / config.experiment_name / "model.pt"
    agent.save(model_path)
    
    console.print(f"[bold green]Training complete! Model saved to {model_path}[/bold green]")
    
    logger.close()
    env.close()


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("env_name")
@click.option("--episodes", "-n", default=10, help="Number of episodes")
@click.option("--render/--no-render", default=False, help="Render environment")
def evaluate(model_path: str, env_name: str, episodes: int, render: bool) -> None:
    console.print("[bold blue]Evaluating agent...[/bold blue]")
    
    env = RLEnvironment(env_name, render_mode="human" if render else None)
    
    agent = PPOAgent(env=env)
    agent.load(Path(model_path))
    
    results = agent.evaluate(episodes=episodes, render=render, deterministic=True)
    
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in results.items():
        table.add_row(key, f"{value:.2f}")
    
    console.print(table)
    
    env.close()


@cli.command()
@click.option("--env", "-e", default="CartPole-v1", help="Environment name")
@click.option("--algorithm", "-a", default="ppo", type=click.Choice(["ppo", "dqn"]))
@click.option("--llm/--no-llm", default=False, help="Use LLM reward shaping")
def quickstart(env: str, algorithm: str, llm: bool) -> None:
    console.print("[bold green]Creating quickstart configuration...[/bold green]")
    
    config = {
        "env_name": env,
        "algorithm": algorithm,
        "training": {
            "total_timesteps": 100000,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "seed": 42,
        },
        "output_dir": "./outputs",
        "experiment_name": f"{env}_{algorithm}_quickstart",
    }
    
    if llm:
        config["llm"] = {
            "provider": "ollama",
            "model": "llama3",
            "temperature": 0.7,
        }
        config["reward_shaping"] = {
            "enabled": True,
            "llm_weight": 0.3,
            "env_weight": 0.7,
        }
    
    config_path = Path("config_quickstart.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    console.print(f"[bold green]Configuration saved to {config_path}[/bold green]")
    console.print("\nTo start training, run:")
    console.print(f"[bold cyan]rl-llm train {config_path}[/bold cyan]")


if __name__ == "__main__":
    cli()
