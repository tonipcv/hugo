FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py MANIFEST.in ./
COPY rl_llm_toolkit/ ./rl_llm_toolkit/
COPY examples/ ./examples/
COPY README.md LICENSE ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[examples]"

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "rl_llm_toolkit.cli.main", "--help"]
