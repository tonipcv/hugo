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
