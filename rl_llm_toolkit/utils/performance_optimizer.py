"""Performance optimization utilities for RL-LLM Toolkit."""

import torch
import numpy as np
from typing import Dict, Any, Optional
import time
from functools import wraps


class PerformanceOptimizer:
    """Utilities for optimizing training performance."""
    
    @staticmethod
    def enable_cudnn_benchmark():
        """Enable cuDNN benchmark mode for faster training."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print("✓ cuDNN benchmark mode enabled")
    
    @staticmethod
    def set_num_threads(num_threads: int):
        """Set number of threads for PyTorch operations."""
        torch.set_num_threads(num_threads)
        print(f"✓ PyTorch threads set to {num_threads}")
    
    @staticmethod
    def enable_tf32():
        """Enable TF32 for faster training on Ampere GPUs."""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✓ TF32 enabled for faster training")
    
    @staticmethod
    def optimize_memory():
        """Optimize memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ CUDA cache cleared")
    
    @staticmethod
    def get_device(prefer_cuda: bool = True) -> torch.device:
        """Get optimal device for training."""
        if prefer_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ Using MPS (Apple Silicon) device")
        else:
            device = torch.device("cpu")
            print("✓ Using CPU device")
        
        return device
    
    @staticmethod
    def profile_function(func):
        """Decorator to profile function execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            print(f"⏱️  {func.__name__} took {end_time - start_time:.4f}s")
            return result
        
        return wrapper
    
    @staticmethod
    def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """Move batch to device efficiently."""
        return {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    
    @staticmethod
    def compile_model(model: torch.nn.Module, mode: str = "default") -> torch.nn.Module:
        """Compile model for faster execution (PyTorch 2.0+)."""
        if hasattr(torch, 'compile'):
            try:
                compiled_model = torch.compile(model, mode=mode)
                print(f"✓ Model compiled with mode: {mode}")
                return compiled_model
            except Exception as e:
                print(f"⚠️  Model compilation failed: {e}")
                return model
        else:
            print("⚠️  torch.compile not available (requires PyTorch 2.0+)")
            return model
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            stats['cuda_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats['cuda_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
            stats['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        return stats
    
    @staticmethod
    def optimize_dataloader_workers(num_workers: Optional[int] = None) -> int:
        """Get optimal number of dataloader workers."""
        import os
        
        if num_workers is not None:
            return num_workers
        
        # Use number of CPU cores, but cap at 8
        cpu_count = os.cpu_count() or 1
        optimal_workers = min(cpu_count, 8)
        
        return optimal_workers
    
    @staticmethod
    def enable_mixed_precision() -> torch.cuda.amp.GradScaler:
        """Enable automatic mixed precision training."""
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            print("✓ Mixed precision training enabled")
            return scaler
        else:
            print("⚠️  Mixed precision requires CUDA")
            return None
    
    @staticmethod
    def gradient_checkpointing(model: torch.nn.Module):
        """Enable gradient checkpointing to save memory."""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        else:
            print("⚠️  Model doesn't support gradient checkpointing")


class TrainingProfiler:
    """Profile training performance."""
    
    def __init__(self):
        self.timings = {}
        self.current_section = None
        self.start_time = None
    
    def start(self, section_name: str):
        """Start timing a section."""
        self.current_section = section_name
        self.start_time = time.time()
    
    def end(self):
        """End timing current section."""
        if self.current_section and self.start_time:
            elapsed = time.time() - self.start_time
            
            if self.current_section not in self.timings:
                self.timings[self.current_section] = []
            
            self.timings[self.current_section].append(elapsed)
            self.current_section = None
            self.start_time = None
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary."""
        summary = {}
        
        for section, times in self.timings.items():
            summary[section] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
                'count': len(times)
            }
        
        return summary
    
    def print_summary(self):
        """Print timing summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("Training Performance Profile")
        print("=" * 60)
        
        for section, stats in summary.items():
            print(f"\n{section}:")
            print(f"  Mean:  {stats['mean']:.4f}s")
            print(f"  Std:   {stats['std']:.4f}s")
            print(f"  Min:   {stats['min']:.4f}s")
            print(f"  Max:   {stats['max']:.4f}s")
            print(f"  Total: {stats['total']:.2f}s ({stats['count']} calls)")
        
        print("=" * 60)
    
    def reset(self):
        """Reset all timings."""
        self.timings = {}
        self.current_section = None
        self.start_time = None


def optimize_training_setup(
    use_cuda: bool = True,
    num_threads: Optional[int] = None,
    enable_amp: bool = False,
    enable_compile: bool = False
) -> Dict[str, Any]:
    """
    Optimize training setup with recommended settings.
    
    Args:
        use_cuda: Whether to use CUDA if available
        num_threads: Number of CPU threads (None for auto)
        enable_amp: Enable automatic mixed precision
        enable_compile: Enable model compilation (PyTorch 2.0+)
    
    Returns:
        Dictionary with optimization settings
    """
    optimizer = PerformanceOptimizer()
    
    print("=" * 60)
    print("Optimizing Training Setup")
    print("=" * 60)
    
    # Get device
    device = optimizer.get_device(prefer_cuda=use_cuda)
    
    # Set threads
    if num_threads:
        optimizer.set_num_threads(num_threads)
    
    # Enable optimizations
    if use_cuda and torch.cuda.is_available():
        optimizer.enable_cudnn_benchmark()
        optimizer.enable_tf32()
        
        if enable_amp:
            scaler = optimizer.enable_mixed_precision()
        else:
            scaler = None
    else:
        scaler = None
    
    # Get optimal dataloader workers
    num_workers = optimizer.optimize_dataloader_workers(num_threads)
    
    print(f"✓ Dataloader workers: {num_workers}")
    print("=" * 60)
    
    return {
        'device': device,
        'scaler': scaler,
        'num_workers': num_workers,
        'compile_enabled': enable_compile
    }
