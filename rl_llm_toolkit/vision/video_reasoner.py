from typing import Optional, Dict, Any, List
import numpy as np
from pathlib import Path
import base64
from io import BytesIO


class VideoReasoningBackend:
    """
    Video reasoning backend for visual RL environments.
    
    Integrates with vision-language models to provide:
    - Frame-by-frame analysis
    - Action quality assessment
    - Visual reward shaping
    - Trajectory understanding
    """
    
    def __init__(
        self,
        model: str = "gpt-4-vision",
        api_key: Optional[str] = None,
        max_frames: int = 4,
    ):
        self.model = model
        self.api_key = api_key
        self.max_frames = max_frames
        
        self.usage_stats = {
            "total_requests": 0,
            "total_frames_processed": 0,
        }
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a single frame.
        
        Args:
            frame: RGB frame as numpy array (H, W, 3)
            context: Optional context about the task
            
        Returns:
            Analysis results
        """
        frame_b64 = self._encode_frame(frame)
        
        prompt = "Analyze this game/environment frame. "
        if context:
            prompt += f"Context: {context}. "
        prompt += "Describe what you see and rate the current state (0-1):"
        
        response = self._call_vision_model(frame_b64, prompt)
        
        self.usage_stats["total_requests"] += 1
        self.usage_stats["total_frames_processed"] += 1
        
        return {
            "description": response.get("description", ""),
            "state_quality": response.get("quality", 0.5),
            "suggestions": response.get("suggestions", []),
        }
    
    def analyze_trajectory(
        self,
        frames: List[np.ndarray],
        actions: List[int],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a sequence of frames and actions.
        
        Args:
            frames: List of RGB frames
            actions: List of actions taken
            context: Optional task context
            
        Returns:
            Trajectory analysis
        """
        if len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames = [frames[i] for i in indices]
            actions = [actions[i] for i in indices if i < len(actions)]
        
        frames_b64 = [self._encode_frame(f) for f in frames]
        
        prompt = f"Analyze this sequence of {len(frames)} frames. "
        if context:
            prompt += f"Task: {context}. "
        prompt += f"Actions taken: {actions}. "
        prompt += "Evaluate the trajectory quality (0-1) and provide feedback:"
        
        response = self._call_vision_model_batch(frames_b64, prompt)
        
        self.usage_stats["total_requests"] += 1
        self.usage_stats["total_frames_processed"] += len(frames)
        
        return {
            "trajectory_quality": response.get("quality", 0.5),
            "feedback": response.get("feedback", ""),
            "key_moments": response.get("key_moments", []),
        }
    
    def assess_action_quality(
        self,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        action: int,
        action_names: Optional[List[str]] = None,
    ) -> float:
        """
        Assess quality of an action based on visual change.
        
        Args:
            before_frame: Frame before action
            after_frame: Frame after action
            action: Action taken
            action_names: Optional names for actions
            
        Returns:
            Quality score (0-1)
        """
        before_b64 = self._encode_frame(before_frame)
        after_b64 = self._encode_frame(after_frame)
        
        action_name = action_names[action] if action_names and action < len(action_names) else str(action)
        
        prompt = (
            f"Compare these two frames. Action taken: {action_name}. "
            "Rate how good this action was (0-1) based on the visual change. "
            "Respond with only a number:"
        )
        
        response = self._call_vision_model_batch([before_b64, after_b64], prompt)
        
        self.usage_stats["total_requests"] += 1
        self.usage_stats["total_frames_processed"] += 2
        
        try:
            quality = float(response.get("quality", 0.5))
            return np.clip(quality, 0.0, 1.0)
        except (ValueError, TypeError):
            return 0.5
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame to base64 string."""
        try:
            from PIL import Image
            
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            img = Image.fromarray(frame)
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
        except ImportError:
            return ""
    
    def _call_vision_model(
        self,
        frame_b64: str,
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Call vision model API (mock implementation).
        
        In production, this would call GPT-4 Vision, Claude with vision, etc.
        """
        return {
            "description": "Mock frame analysis",
            "quality": 0.7,
            "suggestions": ["Continue current strategy"],
        }
    
    def _call_vision_model_batch(
        self,
        frames_b64: List[str],
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Call vision model with multiple frames (mock implementation).
        """
        return {
            "quality": 0.75,
            "feedback": "Mock trajectory analysis",
            "key_moments": [0, len(frames_b64) - 1],
        }
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return self.usage_stats.copy()
