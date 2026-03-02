# Deployment Guide

This guide covers deploying RL-LLM Toolkit models in production environments.

## Table of Contents

1. [Model Serving](#model-serving)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Performance Optimization](#performance-optimization)
5. [Monitoring](#monitoring)
6. [Security](#security)

---

## Model Serving

### Basic Model Server

Create a simple Flask API for serving trained models:

```python
from flask import Flask, request, jsonify
from rl_llm_toolkit import RLEnvironment, PPOAgent
from pathlib import Path
import numpy as np

app = Flask(__name__)

# Load model at startup
env = RLEnvironment("CartPole-v1")
agent = PPOAgent(env=env)
agent.load(Path("./models/cartpole_ppo.pt"))

@app.route('/predict', methods=['POST'])
def predict():
    """Predict action given observation."""
    data = request.json
    observation = np.array(data['observation'])
    
    action, info = agent.predict(observation, deterministic=True)
    
    return jsonify({
        'action': int(action),
        'q_values': info.get('q_values', []).tolist() if 'q_values' in info else None
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI Server (Recommended)

For better performance and automatic API documentation:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rl_llm_toolkit import RLEnvironment, PPOAgent
import numpy as np

app = FastAPI(title="RL Model API")

# Load model
env = RLEnvironment("CartPole-v1")
agent = PPOAgent(env=env)
agent.load("./models/cartpole_ppo.pt")

class PredictionRequest(BaseModel):
    observation: list[float]
    deterministic: bool = True

class PredictionResponse(BaseModel):
    action: int
    confidence: float = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict action from observation."""
    try:
        obs = np.array(request.observation)
        action, info = agent.predict(obs, deterministic=request.deterministic)
        
        return PredictionResponse(
            action=int(action),
            confidence=float(info.get('max_q_value', 0.0))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

Run with:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Docker Deployment

### Dockerfile for Production

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install -e .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  rl-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    environment:
      - MODEL_PATH=/app/models/cartpole_ppo.pt
      - LOG_LEVEL=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - rl-api
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t rl-llm-toolkit:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  rl-llm-toolkit:latest

# Or use docker-compose
docker-compose up -d
```

---

## Cloud Deployment

### AWS Deployment

#### Using AWS Lambda

```python
# lambda_handler.py
import json
import boto3
import numpy as np
from rl_llm_toolkit import PPOAgent, RLEnvironment

# Load model from S3
s3 = boto3.client('s3')
s3.download_file('my-bucket', 'models/agent.pt', '/tmp/agent.pt')

env = RLEnvironment("CartPole-v1")
agent = PPOAgent(env=env)
agent.load('/tmp/agent.pt')

def lambda_handler(event, context):
    """AWS Lambda handler."""
    body = json.loads(event['body'])
    observation = np.array(body['observation'])
    
    action, _ = agent.predict(observation, deterministic=True)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'action': int(action)})
    }
```

#### Using ECS/Fargate

```yaml
# task-definition.json
{
  "family": "rl-model-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "rl-api",
      "image": "your-ecr-repo/rl-llm-toolkit:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/models/agent.pt"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rl-model-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/rl-api

# Deploy to Cloud Run
gcloud run deploy rl-api \
  --image gcr.io/PROJECT_ID/rl-api \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10
```

### Azure

#### Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name rl-api \
  --image myregistry.azurecr.io/rl-llm-toolkit:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables MODEL_PATH=/models/agent.pt
```

---

## Performance Optimization

### Model Optimization

```python
import torch
from rl_llm_toolkit import PPOAgent, RLEnvironment

env = RLEnvironment("CartPole-v1")
agent = PPOAgent(env=env)
agent.load("./models/agent.pt")

# Convert to TorchScript for faster inference
scripted_model = torch.jit.script(agent.policy_network)
scripted_model.save("./models/agent_scripted.pt")

# Quantization for smaller models
quantized_model = torch.quantization.quantize_dynamic(
    agent.policy_network,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### Batch Prediction

```python
@app.post("/predict_batch")
async def predict_batch(observations: list[list[float]]):
    """Batch prediction for better throughput."""
    obs_array = np.array(observations)
    
    # Batch inference
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs_array).float()
        actions = agent.policy_network(obs_tensor).argmax(dim=1)
    
    return {"actions": actions.tolist()}
```

### Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_predict(obs_hash: str):
    """Cache predictions for identical observations."""
    obs = np.frombuffer(bytes.fromhex(obs_hash))
    action, info = agent.predict(obs, deterministic=True)
    return int(action)

def predict_with_cache(observation):
    obs_hash = hashlib.sha256(observation.tobytes()).hexdigest()
    return cached_predict(obs_hash)
```

---

## Monitoring

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_confidence = Gauge('model_confidence', 'Model confidence')

@app.post("/predict")
@prediction_latency.time()
async def predict(request: PredictionRequest):
    prediction_counter.inc()
    
    action, info = agent.predict(observation, deterministic=True)
    
    confidence = info.get('max_q_value', 0.0)
    model_confidence.set(confidence)
    
    return {"action": int(action)}

# Start metrics server
start_http_server(9090)
```

### Logging

```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

@app.post("/predict")
async def predict(request: PredictionRequest):
    logger.info("prediction_request", 
                observation_shape=len(request.observation))
    
    try:
        action, info = agent.predict(observation, deterministic=True)
        logger.info("prediction_success", action=int(action))
        return {"action": int(action)}
    except Exception as e:
        logger.error("prediction_failed", error=str(e))
        raise
```

---

## Security

### API Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    token = credentials.credentials
    if token != os.getenv("API_TOKEN"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return token

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    # Protected endpoint
    pass
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request, data: PredictionRequest):
    # Rate limited endpoint
    pass
```

### Input Validation

```python
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    observation: list[float]
    
    @validator('observation')
    def validate_observation(cls, v):
        if len(v) != 4:  # CartPole observation size
            raise ValueError('Observation must have 4 elements')
        if any(abs(x) > 10 for x in v):
            raise ValueError('Observation values out of range')
        return v
```

---

## Best Practices

1. **Model Versioning**: Use semantic versioning for models
2. **Health Checks**: Implement comprehensive health checks
3. **Graceful Shutdown**: Handle SIGTERM for clean shutdowns
4. **Resource Limits**: Set memory and CPU limits
5. **Monitoring**: Track latency, throughput, and errors
6. **Logging**: Use structured logging for better debugging
7. **Security**: Implement authentication and rate limiting
8. **Testing**: Load test before production deployment
9. **Rollback Plan**: Have a rollback strategy ready
10. **Documentation**: Document API endpoints and usage

## Troubleshooting

### High Latency
- Use batch prediction
- Enable model caching
- Optimize model (quantization, pruning)
- Scale horizontally

### Memory Issues
- Reduce batch size
- Use model quantization
- Implement request queuing
- Monitor memory usage

### Deployment Failures
- Check logs
- Verify dependencies
- Test locally first
- Use health checks

## Next Steps

- Set up CI/CD pipeline
- Implement A/B testing
- Add model monitoring
- Create deployment automation
