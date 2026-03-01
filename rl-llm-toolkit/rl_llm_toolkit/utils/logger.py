from typing import Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime
import logging


class Logger:
    def __init__(
        self,
        log_dir: Path,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.experiment_dir / "metrics.jsonl"
        
        self.tensorboard_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_writer = SummaryWriter(
                    log_dir=str(self.experiment_dir / "tensorboard")
                )
            except ImportError:
                logging.warning("TensorBoard not available. Install with: pip install tensorboard")
        
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_config.get("project", "rl-llm-toolkit"),
                    entity=wandb_config.get("entity"),
                    name=experiment_name,
                    config=wandb_config.get("config", {}),
                    dir=str(self.experiment_dir),
                )
            except ImportError:
                logging.warning("Weights & Biases not available. Install with: pip install wandb")

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        metrics["step"] = step
        metrics["timestamp"] = datetime.now().isoformat()
        
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        
        if self.tensorboard_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(key, value, step)
        
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        params_file = self.experiment_dir / "hyperparameters.json"
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)

    def save_config(self, config: Dict[str, Any]) -> None:
        config_file = self.experiment_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def close(self) -> None:
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            self.wandb_run.finish()


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
