# Development

Guide for setting up a local development environment.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support

## Local Setup

### Clone Repository

```bash
git clone https://github.com/johannhartmann/linguagap.git
cd linguagap
```

### Install Dependencies

```bash
# Install dev dependencies only (no ML libraries)
uv sync --only-dev

# Install all dependencies (requires CUDA)
uv sync
```

### Pre-commit Hooks

```bash
# Install pre-commit
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

## Code Quality

### Linting

```bash
# Check code style
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/
```

### Formatting

```bash
# Check formatting
uv run ruff format --check src/

# Auto-format
uv run ruff format src/
```

### Type Checking

```bash
uv run ty check src/
```

### Security Scanning

```bash
uv run bandit -r src/
```

## Testing

### Run All Tests

```bash
PYTHONPATH=src uv run pytest tests/ -v
```

### Run Specific Tests

```bash
# Single file
PYTHONPATH=src uv run pytest tests/test_streaming_policy.py -v

# Single test
PYTHONPATH=src uv run pytest tests/test_mt.py::test_translate -v
```

### Coverage Report

```bash
PYTHONPATH=src uv run pytest tests/ -v \
  --cov=src/app \
  --cov-report=term-missing \
  --cov-report=html
```

### Smoke Tests (Requires GPU)

```bash
# Via Docker
docker compose exec backend python -m app.scripts.asr_smoke
docker compose exec backend python -m app.scripts.mt_smoke
docker compose exec backend python -m app.scripts.pipeline_smoke
```

## Project Structure

```
linguagap/
├── src/app/                    # Main application
│   ├── main.py                 # FastAPI entry point
│   ├── asr.py                  # ASR module
│   ├── mt.py                   # MT module
│   ├── streaming.py            # WebSocket handler
│   ├── streaming_policy.py     # Segment tracking
│   └── scripts/                # Utility scripts
│       ├── asr_smoke.py
│       ├── mt_smoke.py
│       └── pipeline_smoke.py
├── tests/                      # Test suite
│   ├── conftest.py             # Fixtures
│   ├── test_main.py
│   ├── test_asr.py
│   ├── test_mt.py
│   ├── test_streaming.py
│   └── test_streaming_policy.py
├── static/                     # Web UI
│   └── index.html
├── chart/                      # Helm chart
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
├── docs/                       # Documentation
├── .github/workflows/          # CI/CD
│   └── ci.yml
├── pyproject.toml              # Python project config
├── Dockerfile                  # Container build
├── compose.yaml                # Local development
└── mkdocs.yml                  # Documentation config
```

## Code Style Guidelines

### Python

- Line length: 100 characters
- Quote style: Double quotes
- Import sorting: isort compatible (via ruff)
- Type hints: Required for public functions

### Example

```python
from dataclasses import dataclass


@dataclass
class Segment:
    """A transcribed segment with timing information."""

    id: int
    abs_start: float
    abs_end: float
    src: str
    src_lang: str
    final: bool


def process_segment(segment: Segment) -> dict[str, str]:
    """Process a segment and return results.

    Args:
        segment: The segment to process.

    Returns:
        Dictionary with processing results.
    """
    return {
        "text": segment.src,
        "language": segment.src_lang,
    }
```

## Adding New Features

### 1. Create Feature Branch

```bash
git checkout -b feature/my-feature
```

### 2. Write Tests First

```python
# tests/test_my_feature.py
def test_my_feature():
    result = my_feature()
    assert result == expected
```

### 3. Implement Feature

```python
# src/app/my_feature.py
def my_feature():
    return expected
```

### 4. Run Quality Checks

```bash
uv run pre-commit run --all-files
PYTHONPATH=src uv run pytest tests/ -v
```

### 5. Commit and Push

```bash
git add .
git commit -m "Add my feature"
git push origin feature/my-feature
```

### 6. Create Pull Request

Open PR on GitHub. CI will run automatically.

## Debugging

### Local Debugging

```python
# Add to code
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

### Docker Debugging

```bash
# Run with debug port
docker compose run --service-ports backend \
  python -m debugpy --listen 0.0.0.0:5678 \
  -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

## WebSocket Testing

### Using wscat

```bash
# Install
npm install -g wscat

# Connect
wscat -c ws://localhost:8000/ws

# Send config
{"type": "config", "src_lang": "en", "tgt_lang": "de"}
```

### Using Python

```python
import asyncio
import websockets

async def test_ws():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send('{"type": "config", "src_lang": "en", "tgt_lang": "de"}')
        response = await ws.recv()
        print(response)

asyncio.run(test_ws())
```

## Common Development Tasks

### Update Dependencies

```bash
# Update lock file
uv lock

# Sync environment
uv sync
```

### Build Docker Image

```bash
docker build -t linguagap:dev .
```

### Run Single Service

```bash
docker compose up backend
```

### Access Container Shell

```bash
docker compose exec backend bash
```

### View Model Cache

```bash
docker compose exec backend ls -la /data/hf
```
