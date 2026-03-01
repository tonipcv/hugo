.PHONY: install test lint format clean docs

install:
	pip install -e ".[dev,examples,llm]"

install-dev:
	pip install -e ".[dev,examples,llm]"
	pre-commit install

test:
	pytest --cov=rl_llm_toolkit --cov-report=html --cov-report=term

test-fast:
	pytest -x --ff

lint:
	ruff check rl_llm_toolkit tests examples
	mypy rl_llm_toolkit --ignore-missing-imports

format:
	black rl_llm_toolkit tests examples
	ruff check --fix rl_llm_toolkit tests examples

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python -m build

publish-test:
	python -m build
	twine upload --repository testpypi dist/*

publish:
	python -m build
	twine upload dist/*

docs:
	cd docs && mkdocs build

serve-docs:
	cd docs && mkdocs serve

example-cartpole:
	python examples/cartpole_basic.py

example-llm:
	python examples/cartpole_with_llm.py

example-trading:
	python examples/trading_example.py

benchmark:
	python examples/compare_algorithms.py
