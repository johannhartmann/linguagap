# Contributing to LinguaGap

Thank you for your interest in contributing to LinguaGap! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose (for full testing)
- NVIDIA GPU with CUDA support (optional, for GPU testing)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/johannhartmann/linguagap.git
   cd linguagap
   ```

2. **Install uv**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dev dependencies**
   ```bash
   uv sync --only-dev
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

### Making Changes

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run quality checks:
   ```bash
   pre-commit run --all-files
   ```

4. Run tests:
   ```bash
   uv run pytest tests/ -v
   ```

5. Commit your changes with a descriptive message

6. Push and create a Pull Request

### Code Quality Standards

All code must pass the following checks (enforced by pre-commit and CI):

- **Ruff Linting** - Code style and potential errors
- **Ruff Formatting** - Consistent code formatting
- **ty Type Checking** - Type annotations and correctness
- **Bandit Security Scan** - Security vulnerability detection
- **pytest** - All tests must pass

### Running Individual Checks

```bash
# Linting
uv run ruff check src/

# Formatting (check only)
uv run ruff format --check src/

# Formatting (auto-fix)
uv run ruff format src/

# Type checking
uv run ty check src/

# Security scan
uv run bandit -c pyproject.toml -r src/

# Tests with coverage
uv run pytest tests/ -v --cov=src/app --cov-report=term-missing
```

## Pull Request Guidelines

### Before Submitting

- [ ] All pre-commit hooks pass
- [ ] All tests pass
- [ ] New features include tests
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive

### PR Description

Please include:
- **What** - Brief description of changes
- **Why** - Motivation for the change
- **How** - Technical approach (if non-obvious)
- **Testing** - How you tested the changes

### Review Process

1. PRs require at least one approval
2. CI must pass (lint, typecheck, security, tests, Docker build)
3. Merge conflicts must be resolved

## Project Structure

```
linguagap/
├── src/
│   └── app/
│       ├── main.py           # FastAPI application
│       ├── asr.py            # ASR (Whisper) module
│       ├── mt.py             # MT (Translation) module
│       ├── streaming.py      # WebSocket streaming
│       ├── streaming_policy.py # Segment tracking
│       └── scripts/          # Utility scripts
├── tests/                    # Test files
├── static/                   # Web interface
├── .github/workflows/        # CI/CD
├── Dockerfile               # Container build
└── compose.yaml             # Docker Compose config
```

## Testing

### Unit Tests

Located in `tests/`. Run with:
```bash
uv run pytest tests/ -v
```

### Integration Tests

Run smoke tests in Docker:
```bash
docker compose up --build backend
docker compose exec backend python -m app.scripts.asr_smoke
docker compose exec backend python -m app.scripts.mt_smoke
```

### Manual Testing

1. Start the service: `docker compose up --build backend`
2. Open http://localhost:8000
3. Test microphone capture and real-time transcription

## Reporting Issues

### Bug Reports

Include:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version, GPU if applicable)
- Relevant logs

### Feature Requests

Include:
- Use case description
- Proposed solution (if any)
- Alternatives considered

## Questions?

Open a [GitHub Issue](https://github.com/johannhartmann/linguagap/issues) for questions or discussions.
