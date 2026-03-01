# Contributing to RL-LLM Toolkit

Thank you for your interest in contributing to RL-LLM Toolkit! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- **Bug Reports**: Found a bug? Open an issue with detailed reproduction steps
- **Feature Requests**: Have an idea? Share it in the issues section
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Help improve our docs, tutorials, and examples
- **Community Support**: Answer questions and help other users

## 🚀 Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/rl-llm-toolkit.git
cd rl-llm-toolkit
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,examples,llm]"

# Install pre-commit hooks
pre-commit install
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rl_llm_toolkit --cov-report=html

# Run specific test file
pytest tests/test_agents.py
```

### 4. Code Quality

```bash
# Format code with black
black rl_llm_toolkit tests examples

# Lint with ruff
ruff check rl_llm_toolkit tests examples

# Type checking with mypy
mypy rl_llm_toolkit
```

## 📝 Pull Request Process

1. **Create a Branch**: Use descriptive names like `feature/llm-caching` or `fix/ppo-gradient-bug`

2. **Write Tests**: All new features must include tests

3. **Update Documentation**: Add docstrings and update relevant docs

4. **Follow Code Style**: 
   - Use Black for formatting (line length: 100)
   - Follow PEP 8 guidelines
   - Add type hints to all functions
   - Write clear, descriptive variable names

5. **Commit Messages**: Use clear, descriptive commit messages
   ```
   feat: Add support for Anthropic Claude LLM backend
   fix: Resolve PPO gradient clipping issue
   docs: Update installation instructions
   ```

6. **Submit PR**: 
   - Fill out the PR template completely
   - Link related issues
   - Request review from maintainers

## 🧪 Testing Guidelines

- Write unit tests for all new functionality
- Aim for >80% code coverage
- Test edge cases and error handling
- Use fixtures for common test setups
- Mock external dependencies (LLM APIs, etc.)

Example test structure:
```python
class TestNewFeature:
    def test_basic_functionality(self):
        # Arrange
        env = RLEnvironment("CartPole-v1")
        
        # Act
        result = some_function(env)
        
        # Assert
        assert result is not None
        env.close()
```

## 📚 Documentation Standards

- All public functions/classes must have docstrings
- Use Google-style docstrings
- Include examples in docstrings when helpful
- Update README.md for major features

Example docstring:
```python
def train_agent(env: RLEnvironment, timesteps: int) -> Dict[str, Any]:
    """Train an RL agent in the given environment.
    
    Args:
        env: The environment to train in
        timesteps: Total number of training timesteps
        
    Returns:
        Dictionary containing training statistics
        
    Raises:
        ValueError: If timesteps is negative
        
    Example:
        >>> env = RLEnvironment("CartPole-v1")
        >>> results = train_agent(env, 10000)
        >>> print(results["mean_reward"])
    """
```

## 🎯 Feature Development Guidelines

### Adding New RL Algorithms

1. Create new file in `rl_llm_toolkit/agents/`
2. Inherit from `BaseAgent`
3. Implement required methods: `train()`, `predict()`, `save()`, `load()`
4. Add comprehensive tests
5. Create example script in `examples/`
6. Update documentation

### Adding New LLM Backends

1. Create new file in `rl_llm_toolkit/llm/`
2. Inherit from `LLMBackend`
3. Implement `generate()` and `generate_batch()`
4. Handle errors gracefully
5. Add tests with mocked API calls
6. Update configuration options

### Adding New Environments

1. Create in `rl_llm_toolkit/environments/`
2. Follow Gymnasium API standards
3. Include docstrings explaining the environment
4. Add visualization/rendering if applicable
5. Create example training script
6. Add tests for all actions and edge cases

## 🐛 Bug Report Template

When reporting bugs, include:

- **Description**: Clear description of the bug
- **Reproduction Steps**: Minimal code to reproduce
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: Python version, OS, package versions
- **Logs/Errors**: Full error messages and stack traces

## 💡 Feature Request Template

When requesting features, include:

- **Problem**: What problem does this solve?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Use Case**: Real-world scenario where this helps
- **Implementation Ideas**: Technical approach (optional)

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions
- No harassment or discrimination

## 📞 Getting Help

- **Discord**: Join our community server
- **GitHub Discussions**: Ask questions and share ideas
- **Twitter/X**: Follow @your_handle for updates
- **Email**: contact@rl-llm-toolkit.dev

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Eligible for contributor badges
- Invited to maintainer team (for consistent contributors)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for making RL-LLM Toolkit better!** 🚀
