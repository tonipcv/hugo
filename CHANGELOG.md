# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-agent support
- Offline RL algorithms
- Hugging Face model hub integration
- Vectorized environments
- Advanced visualization tools

## [0.2.0] - 2026-03-02

### Added - Offline RL
- Conservative Q-Learning (CQL) algorithm for offline RL
- Implicit Q-Learning (IQL) algorithm for offline RL
- Dataset collection utilities
- Offline RL examples and tutorials

### Added - Multi-Agent Systems
- MADDPG (Multi-Agent DDPG) implementation
- Cooperative Navigation environment
- Multi-agent base classes and utilities
- Multi-agent training examples

### Added - Integrations
- Hugging Face Hub integration for model sharing
- Community leaderboard system with SQLite backend
- Collaborative training session management
- Shared replay buffer for distributed training

### Added - Advanced Environments
- Stock trading environment with technical indicators
- Multiple stock portfolio management
- Transaction costs and slippage simulation
- Sharpe ratio and drawdown tracking

### Added - Vision & Reasoning
- Video reasoning backend for visual RL
- Visual reward shaping with vision-language models
- Frame-by-frame analysis capabilities
- Trajectory quality assessment

### Added - Benchmarking
- Comprehensive benchmarking suite
- Performance metrics calculation
- Algorithm comparison tools
- Profiling utilities (FPS, memory usage)

### Added - Documentation
- Advanced features guide
- 4 comprehensive Jupyter notebooks
- Offline RL tutorial
- Multi-agent tutorial
- API documentation updates
- FAQ expansions

### Added - Examples
- Offline RL example (CQL/IQL)
- Multi-agent example (MADDPG)
- Stock trading example
- Hugging Face integration example
- Collaboration example
- Benchmark example

### Added - Tests
- Offline RL agent tests
- Multi-agent system tests
- Integration tests (leaderboard, HF Hub)
- Environment tests

### Changed
- Updated package version to 0.2.0
- Enhanced README with new features
- Improved module exports
- Extended API reference

## [0.1.0] - 2026-03-01

### Added
- Initial release of RL-LLM Toolkit
- Core RL framework with Gymnasium integration
- PPO (Proximal Policy Optimization) algorithm
- DQN (Deep Q-Network) algorithm
- LLM integration for reward shaping
  - Ollama backend (local)
  - OpenAI backend (API)
- Reward shaping with caching
- Custom prompt templates
- Crypto trading environment
- CLI tools for training and evaluation
- Comprehensive test suite
- Documentation and tutorials
- Example scripts and configurations
- TensorBoard and Weights & Biases logging
- Docker support
- CI/CD with GitHub Actions

### Features
- Modular architecture for easy extension
- GPU/CPU automatic detection
- Type-safe configuration with Pydantic
- Efficient LLM response caching (80%+ hit rate)
- Episode statistics tracking
- Model checkpointing and loading
- Visualization utilities

### Documentation
- Quick start guide
- API reference
- Architecture overview
- FAQ
- Contributing guidelines
- Example notebooks

### Examples
- CartPole with PPO
- CartPole with LLM reward shaping
- LunarLander with DQN
- Crypto trading bot
- Algorithm comparison benchmark
- Advanced training with logging

## [0.0.1] - 2026-02-15

### Added
- Project structure and initial setup
- Basic environment wrapper
- Proof of concept for LLM integration

---

## Version History

### Version 0.1.0 (Current)
**Release Date**: March 1, 2026

**Highlights**:
- First stable release
- Production-ready PPO and DQN implementations
- Full LLM integration with Ollama and OpenAI
- Comprehensive documentation and examples

**Breaking Changes**: None (initial release)

**Known Issues**:
- Multi-GPU training not yet supported
- Limited to discrete and continuous action spaces
- Anthropic Claude backend not yet implemented

**Contributors**: 
- Core development team
- Community contributors (see CONTRIBUTORS.md)

---

## Migration Guides

### From Pre-release to 0.1.0
No migration needed - this is the first stable release.

---

## Deprecation Notices

None yet.

---

## Security Updates

None yet. We follow responsible disclosure practices. Report security issues to security@rl-llm-toolkit.dev.
