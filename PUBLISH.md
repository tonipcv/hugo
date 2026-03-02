# Publishing Guide for RL-LLM Toolkit

This guide explains how to publish the RL-LLM Toolkit to PyPI.

## Prerequisites

1. PyPI account: https://pypi.org/account/register/
2. TestPyPI account (for testing): https://test.pypi.org/account/register/
3. Install build tools:
   ```bash
   pip install build twine
   ```

## Pre-Publication Checklist

- [x] All tests passing (87/87 ✅)
- [x] Version updated in `pyproject.toml` (0.2.0)
- [x] CHANGELOG.md updated
- [x] README.md complete
- [x] Documentation complete
- [x] Examples working
- [x] Requirements specified
- [ ] License file present (MIT)
- [ ] Code quality checks passed

## Build Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build distribution
python -m build

# This creates:
# - dist/rl_llm_toolkit-0.2.0-py3-none-any.whl
# - dist/rl-llm-toolkit-0.2.0.tar.gz
```

## Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ rl-llm-toolkit

# Verify
python -c "import rl_llm_toolkit; print(rl_llm_toolkit.__version__)"
```

## Publish to PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# Enter credentials when prompted
# Username: __token__
# Password: your-pypi-token
```

## Post-Publication

1. Create GitHub release:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

2. Update documentation site

3. Announce release:
   - GitHub Discussions
   - Twitter/X
   - Reddit (r/MachineLearning, r/reinforcementlearning)
   - Hugging Face Hub

## Installation Verification

Users can install with:

```bash
pip install rl-llm-toolkit

# With optional dependencies
pip install rl-llm-toolkit[llm]  # LLM backends
pip install rl-llm-toolkit[dev]  # Development tools
pip install rl-llm-toolkit[examples]  # Jupyter notebooks
pip install rl-llm-toolkit[all]  # Everything
```

## Troubleshooting

### Build Errors

```bash
# Check pyproject.toml syntax
python -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))"

# Validate package
python -m twine check dist/*
```

### Upload Errors

- Ensure version number is unique (not already on PyPI)
- Check API token permissions
- Verify package name availability

### Import Errors After Installation

- Check dependencies in pyproject.toml
- Verify package structure
- Test in clean virtual environment

## Version Management

Follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Current: 0.2.0
- 0.1.0: Initial release
- 0.2.0: Offline RL, multi-agent, integrations

Next planned: 0.3.0
- Advanced visualization
- More environments
- Performance optimizations

## Continuous Integration

Set up GitHub Actions for automatic publishing:

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

## Support

For publishing issues:
- PyPI Help: https://pypi.org/help/
- Packaging Guide: https://packaging.python.org/
- GitHub Issues: https://github.com/tonipcv/hugo/issues

## License

Ensure LICENSE file is included in distribution (MIT License recommended for open source).
