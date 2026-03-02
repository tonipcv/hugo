#!/usr/bin/env python3
"""Test all package imports to verify module structure."""

import sys
import traceback

def test_import(module_path, items=None):
    """Test importing a module or specific items from it."""
    try:
        if items:
            exec(f"from {module_path} import {', '.join(items)}")
            print(f"✅ {module_path}: {', '.join(items)}")
        else:
            exec(f"import {module_path}")
            print(f"✅ {module_path}")
        return True
    except Exception as e:
        print(f"❌ {module_path}: {e}")
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("TESTING PACKAGE IMPORTS")
    print("="*80)
    
    tests = [
        # Core modules
        ("rl_llm_toolkit", None),
        ("rl_llm_toolkit.core.environment", ["RLEnvironment"]),
        
        # Agents
        ("rl_llm_toolkit.agents.base", ["BaseAgent"]),
        ("rl_llm_toolkit.agents.ppo", ["PPOAgent"]),
        ("rl_llm_toolkit.agents.dqn", ["DQNAgent"]),
        ("rl_llm_toolkit.agents.cql", ["CQLAgent"]),
        ("rl_llm_toolkit.agents.iql", ["IQLAgent"]),
        
        # LLM backends
        ("rl_llm_toolkit.llm.base", ["LLMBackend"]),
        ("rl_llm_toolkit.llm.ollama", ["OllamaBackend"]),
        ("rl_llm_toolkit.llm.openai", ["OpenAIBackend"]),
        
        # Rewards
        ("rl_llm_toolkit.rewards.llm_shaper", ["LLMRewardShaper"]),
        
        # Environments
        ("rl_llm_toolkit.environments.trading", ["CryptoTradingEnv"]),
        ("rl_llm_toolkit.environments.stock_trading", ["StockTradingEnv"]),
        
        # Multi-agent
        ("rl_llm_toolkit.multiagent", ["MADDPGAgent", "MultiAgentEnv"]),
        ("rl_llm_toolkit.multiagent.maddpg", ["MADDPGAgent"]),
        ("rl_llm_toolkit.multiagent.environment", ["MultiAgentEnv", "CooperativeNavigationEnv"]),
        
        # Integrations
        ("rl_llm_toolkit.integrations", ["HuggingFaceHub"]),
        ("rl_llm_toolkit.integrations.huggingface", ["HuggingFaceHub"]),
        ("rl_llm_toolkit.integrations.leaderboard", ["Leaderboard"]),
        
        # Benchmarks
        ("rl_llm_toolkit.benchmarks", ["BenchmarkSuite", "PerformanceMetrics"]),
        ("rl_llm_toolkit.benchmarks.suite", ["BenchmarkSuite"]),
        ("rl_llm_toolkit.benchmarks.metrics", ["PerformanceMetrics"]),
        
        # Collaboration
        ("rl_llm_toolkit.collaboration", ["CollaborationSession", "SharedReplayBuffer"]),
        ("rl_llm_toolkit.collaboration.session", ["CollaborationSession"]),
        ("rl_llm_toolkit.collaboration.shared_buffer", ["SharedReplayBuffer"]),
        
        # Vision
        ("rl_llm_toolkit.vision", ["VideoReasoningBackend", "VisualRewardShaper"]),
        ("rl_llm_toolkit.vision.video_reasoner", ["VideoReasoningBackend"]),
        ("rl_llm_toolkit.vision.visual_reward_shaper", ["VisualRewardShaper"]),
    ]
    
    passed = 0
    failed = 0
    
    for module_path, items in tests:
        if test_import(module_path, items):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*80)
    print(f"IMPORT TEST RESULTS: {passed} passed, {failed} failed")
    print("="*80)
    
    if failed > 0:
        sys.exit(1)
    
    print("\n✅ All imports successful!")
    
    # Test top-level imports
    print("\n" + "="*80)
    print("TESTING TOP-LEVEL PACKAGE IMPORTS")
    print("="*80)
    
    try:
        from rl_llm_toolkit import (
            RLEnvironment, PPOAgent, DQNAgent, CQLAgent, IQLAgent,
            MADDPGAgent, MultiAgentEnv, LLMRewardShaper,
            LLMBackend, OllamaBackend, OpenAIBackend,
            HuggingFaceHub, Leaderboard,
            BenchmarkSuite, PerformanceMetrics,
            CollaborationSession, SharedReplayBuffer,
            VideoReasoningBackend, VisualRewardShaper
        )
        print("✅ All top-level imports successful!")
        print(f"   Package version: {__import__('rl_llm_toolkit').__version__}")
    except Exception as e:
        print(f"❌ Top-level import failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
