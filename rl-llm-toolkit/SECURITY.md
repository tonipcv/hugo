# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of RL-LLM Toolkit seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please Do Not

- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before it has been addressed

### Please Do

**Report security vulnerabilities to**: security@rl-llm-toolkit.dev

Include the following information:
- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
2. **Assessment**: We will assess the vulnerability and determine its impact and severity
3. **Fix Development**: We will develop a fix and prepare a security advisory
4. **Disclosure**: We will coordinate disclosure with you
5. **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

### Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Release**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next regular release

## Security Best Practices

### For Users

#### API Key Management

**Never hardcode API keys**:
```python
# ❌ BAD
llm = OpenAIBackend(api_key="sk-...")

# ✅ GOOD
import os
llm = OpenAIBackend(api_key=os.getenv("OPENAI_API_KEY"))
```

**Use environment variables**:
```bash
export OPENAI_API_KEY="your-key-here"
```

**Use .env files** (never commit them):
```bash
# .env
OPENAI_API_KEY=your-key-here
```

```python
from dotenv import load_dotenv
load_dotenv()
```

#### LLM Safety

**Validate LLM outputs**:
```python
reward_shaper = LLMRewardShaper(
    llm_backend=llm,
    # Outputs are automatically clipped to [-1, 1]
)
```

**Set timeouts**:
```python
llm = OllamaBackend(
    model="llama3",
    timeout=30,  # Prevent hanging
)
```

**Rate limiting** (for API backends):
```python
# Implement rate limiting in production
import time
from functools import wraps

def rate_limit(max_per_minute):
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator
```

#### File System Safety

**Validate paths**:
```python
from pathlib import Path

def safe_save(agent, path_str):
    path = Path(path_str).resolve()
    
    # Ensure path is within allowed directory
    allowed_dir = Path("./models").resolve()
    if not path.is_relative_to(allowed_dir):
        raise ValueError("Path outside allowed directory")
    
    agent.save(path)
```

#### Model Loading

**Verify checksums**:
```python
import hashlib

def verify_model(path, expected_hash):
    with open(path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    if file_hash != expected_hash:
        raise ValueError("Model checksum mismatch")
```

### For Contributors

#### Code Security

- Never commit secrets or API keys
- Use `.gitignore` for sensitive files
- Review dependencies for vulnerabilities
- Sanitize user inputs
- Validate configuration files
- Use type hints and validation (Pydantic)

#### Dependency Security

We use:
- **Dependabot**: Automatic dependency updates
- **pip-audit**: Vulnerability scanning
- **Safety**: Security checks in CI

Run locally:
```bash
pip install pip-audit
pip-audit
```

#### Testing Security

Include security tests:
```python
def test_api_key_not_logged():
    """Ensure API keys are not logged"""
    llm = OpenAIBackend(api_key="secret-key")
    # Check logs don't contain the key
```

## Known Security Considerations

### LLM Prompt Injection

**Risk**: Malicious state observations could inject prompts

**Mitigation**: 
- State values are formatted as numbers/arrays
- Prompts use structured templates
- LLM outputs are validated and clipped

### Model Poisoning

**Risk**: Loading untrusted model checkpoints

**Mitigation**:
- Only load models from trusted sources
- Verify checksums when available
- Use sandboxed environments for testing

### Resource Exhaustion

**Risk**: Unbounded LLM calls or memory usage

**Mitigation**:
- Caching reduces LLM calls by 80%+
- Timeouts prevent hanging
- Memory limits on buffers
- Configurable batch sizes

### Dependency Vulnerabilities

**Risk**: Vulnerabilities in third-party packages

**Mitigation**:
- Regular dependency updates
- Automated vulnerability scanning
- Minimal dependency footprint
- Pin versions in production

## Security Updates

Security updates will be announced via:
- GitHub Security Advisories
- Release notes
- Email to registered users (if applicable)
- Twitter/X announcements

## Compliance

This project aims to follow:
- OWASP Top 10 guidelines
- CWE/SANS Top 25 Most Dangerous Software Errors
- Python security best practices

## Bug Bounty

We currently do not offer a bug bounty program, but we greatly appreciate responsible disclosure and will credit researchers in our security advisories.

## Contact

- **Security Issues**: security@rl-llm-toolkit.dev
- **General Questions**: GitHub Discussions
- **PGP Key**: Available upon request

---

**Last Updated**: March 1, 2026
