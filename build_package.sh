#!/bin/bash
# Build script for RL-LLM Toolkit package

set -e

echo "=========================================="
echo "Building RL-LLM Toolkit Package"
echo "=========================================="

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Run tests
echo "Running tests..."
python -m pytest tests/ -v --cov=rl_llm_toolkit --cov-report=term

# Check code quality
echo "Checking code quality..."
if command -v black &> /dev/null; then
    echo "Running Black..."
    black --check rl_llm_toolkit/ tests/ || echo "Black formatting issues found"
fi

if command -v ruff &> /dev/null; then
    echo "Running Ruff..."
    ruff check rl_llm_toolkit/ tests/ || echo "Ruff linting issues found"
fi

# Build package
echo "Building package..."
python -m build

# Validate package
echo "Validating package..."
python -m twine check dist/*

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Distribution files created in dist/"
ls -lh dist/

echo ""
echo "Next steps:"
echo "1. Test installation: pip install dist/rl_llm_toolkit-*.whl"
echo "2. Upload to TestPyPI: twine upload --repository testpypi dist/*"
echo "3. Upload to PyPI: twine upload dist/*"
