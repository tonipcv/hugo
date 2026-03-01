from typing import List, Optional, Dict, Any
import numpy as np
from pathlib import Path


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Training Curves",
) -> None:
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            axes[idx].plot(values)
            axes[idx].set_title(metric_name)
            axes[idx].set_xlabel("Step")
            axes[idx].set_ylabel("Value")
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()
    
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")


def save_video(
    frames: List[np.ndarray],
    save_path: Path,
    fps: int = 30,
) -> None:
    try:
        import imageio
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        imageio.mimsave(str(save_path), frames, fps=fps)
        print(f"Video saved to {save_path}")
    
    except ImportError:
        print("imageio not available. Install with: pip install imageio")


def smooth_curve(values: List[float], window_size: int = 10) -> np.ndarray:
    if len(values) < window_size:
        return np.array(values)
    
    smoothed = np.convolve(values, np.ones(window_size) / window_size, mode="valid")
    return smoothed
