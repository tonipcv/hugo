from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import os
import tempfile
import shutil


class HuggingFaceHub:
    """
    Integration with Hugging Face Hub for model sharing and discovery.
    
    Features:
    - Upload trained models to HF Hub
    - Download pre-trained models
    - Browse community models
    - Track model metadata and performance
    """
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("HF_TOKEN")
        self._client = None
    
    def _get_client(self):
        """Lazy load Hugging Face Hub client."""
        if self._client is None:
            try:
                from huggingface_hub import HfApi
                self._client = HfApi(token=self.token)
            except ImportError:
                raise ImportError(
                    "huggingface_hub not installed. Install with: pip install huggingface_hub"
                )
        return self._client
    
    def upload_model(
        self,
        model_path: Path,
        repo_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        commit_message: str = "Upload RL model",
        private: bool = False,
    ) -> str:
        """
        Upload a trained model to Hugging Face Hub.
        
        Args:
            model_path: Path to saved model file
            repo_id: Repository ID (username/model-name)
            metadata: Model metadata (algorithm, env, performance, etc.)
            commit_message: Commit message
            private: Whether to make repo private
            
        Returns:
            URL to the uploaded model
        """
        client = self._get_client()
        
        try:
            client.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
            )
        except Exception as e:
            print(f"Repository may already exist: {e}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            shutil.copy(model_path, tmp_path / "model.pt")
            
            if metadata:
                metadata_enhanced = {
                    **metadata,
                    "framework": "rl-llm-toolkit",
                    "model_file": "model.pt",
                }
                
                with open(tmp_path / "metadata.json", "w") as f:
                    json.dump(metadata_enhanced, f, indent=2)
                
                readme_content = self._generate_model_card(metadata_enhanced)
                with open(tmp_path / "README.md", "w") as f:
                    f.write(readme_content)
            
            client.upload_folder(
                folder_path=str(tmp_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
        
        url = f"https://huggingface.co/{repo_id}"
        print(f"✅ Model uploaded to: {url}")
        return url
    
    def download_model(
        self,
        repo_id: str,
        local_dir: Optional[Path] = None,
    ) -> Path:
        """
        Download a model from Hugging Face Hub.
        
        Args:
            repo_id: Repository ID (username/model-name)
            local_dir: Local directory to save model
            
        Returns:
            Path to downloaded model file
        """
        client = self._get_client()
        
        if local_dir is None:
            local_dir = Path("./models") / repo_id.replace("/", "_")
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            client.snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(local_dir),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
        
        model_path = local_dir / "model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found in {local_dir}")
        
        metadata_path = local_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                print(f"Model metadata: {json.dumps(metadata, indent=2)}")
        
        print(f"✅ Model downloaded to: {model_path}")
        return model_path
    
    def list_models(
        self,
        filter_tag: Optional[str] = "rl-llm-toolkit",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        List models on Hugging Face Hub.
        
        Args:
            filter_tag: Filter by tag
            limit: Maximum number of models to return
            
        Returns:
            List of model information dictionaries
        """
        client = self._get_client()
        
        try:
            models = client.list_models(
                filter=filter_tag,
                limit=limit,
                sort="downloads",
                direction=-1,
            )
            
            model_list = []
            for model in models:
                model_list.append({
                    "id": model.modelId,
                    "author": model.author,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                })
            
            return model_list
        
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def _generate_model_card(self, metadata: Dict[str, Any]) -> str:
        """Generate a model card (README) for the uploaded model."""
        
        algorithm = metadata.get("algorithm", "Unknown")
        env_name = metadata.get("env_name", "Unknown")
        mean_reward = metadata.get("mean_reward", "N/A")
        total_timesteps = metadata.get("total_timesteps", "N/A")
        
        card = f"""---
tags:
- reinforcement-learning
- rl-llm-toolkit
- {algorithm.lower()}
- {env_name.lower().replace('-', '_')}
library_name: rl-llm-toolkit
---

# {algorithm} Agent for {env_name}

This model was trained using [RL-LLM Toolkit](https://github.com/yourusername/rl-llm-toolkit).

## Model Description

- **Algorithm**: {algorithm}
- **Environment**: {env_name}
- **Training Timesteps**: {total_timesteps}
- **Mean Reward**: {mean_reward}

## Performance

"""
        
        if "eval_results" in metadata:
            eval_results = metadata["eval_results"]
            card += f"""
| Metric | Value |
|--------|-------|
| Mean Reward | {eval_results.get('mean_reward', 'N/A'):.2f} |
| Std Reward | {eval_results.get('std_reward', 'N/A'):.2f} |
| Min Reward | {eval_results.get('min_reward', 'N/A'):.2f} |
| Max Reward | {eval_results.get('max_reward', 'N/A'):.2f} |
"""
        
        card += f"""

## Usage

```python
from rl_llm_toolkit import RLEnvironment, {algorithm}Agent
from rl_llm_toolkit.integrations import HuggingFaceHub

# Download model
hub = HuggingFaceHub()
model_path = hub.download_model("{metadata.get('repo_id', 'username/model')}")

# Load and use
env = RLEnvironment("{env_name}")
agent = {algorithm}Agent(env=env)
agent.load(model_path)

# Evaluate
results = agent.evaluate(episodes=10)
print(f"Mean reward: {{results['mean_reward']:.2f}}")
```

## Training Details

"""
        
        if "hyperparameters" in metadata:
            card += "### Hyperparameters\n\n```json\n"
            card += json.dumps(metadata["hyperparameters"], indent=2)
            card += "\n```\n\n"
        
        if "llm_config" in metadata:
            card += "### LLM Reward Shaping\n\n"
            card += f"- **Provider**: {metadata['llm_config'].get('provider', 'N/A')}\n"
            card += f"- **Model**: {metadata['llm_config'].get('model', 'N/A')}\n"
            card += f"- **LLM Weight**: {metadata['llm_config'].get('llm_weight', 'N/A')}\n\n"
        
        card += """
## Citation

```bibtex
@software{rl_llm_toolkit,
  title = {RL-LLM Toolkit},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/rl-llm-toolkit}
}
```
"""
        
        return card
    
    def create_leaderboard_entry(
        self,
        env_name: str,
        algorithm: str,
        mean_reward: float,
        std_reward: float,
        model_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a leaderboard entry for a trained model.
        
        Args:
            env_name: Environment name
            algorithm: Algorithm used
            mean_reward: Mean evaluation reward
            std_reward: Standard deviation of reward
            model_id: Hugging Face model ID
            metadata: Additional metadata
            
        Returns:
            Leaderboard entry dictionary
        """
        entry = {
            "env_name": env_name,
            "algorithm": algorithm,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "model_id": model_id,
            "framework": "rl-llm-toolkit",
        }
        
        if metadata:
            entry.update(metadata)
        
        return entry
