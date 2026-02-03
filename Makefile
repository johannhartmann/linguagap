# LinguaGap Development Makefile
# ================================

.PHONY: help dev backend test test-one lint format typecheck security check all clean logs docker-build docker-push docker

# Default target
help:
	@echo "LinguaGap Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start backend server (alias for backend)"
	@echo "  make backend      - Start backend server with uvicorn"
	@echo "  make logs         - Tail backend logs"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all E2E evaluation scenarios"
	@echo "  make test-one S=  - Run single scenario (e.g., make test-one S=en_customer_service)"
	@echo "  make test-unit    - Run unit tests with pytest"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Format code with ruff"
	@echo "  make typecheck    - Run ty type checker"
	@echo "  make security     - Run bandit security scan"
	@echo "  make check        - Run all checks (lint, format, typecheck, security)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker       - Build and push Docker image"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-push  - Push Docker image to ghcr.io"
	@echo "  make docker-run   - Run Docker container locally"
	@echo ""
	@echo "Other:"
	@echo "  make clean        - Clean up generated files"
	@echo "  make smoke-asr    - Run ASR smoke test"
	@echo "  make smoke-mt     - Run MT smoke test"
	@echo "  make smoke        - Run all smoke tests"

# ============================================================================
# Development
# ============================================================================

# Conda environment setup for cuDNN
CONDA_ENV := $(HOME)/anaconda3/envs/linguagap
export LD_LIBRARY_PATH := $(CONDA_ENV)/lib:$(LD_LIBRARY_PATH)
export CUDA_HOME := $(CONDA_ENV)

# Backend server
dev: backend

backend:
	@echo "Starting LinguaGap backend..."
	HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

backend-reload:
	@echo "Starting LinguaGap backend with auto-reload..."
	HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

logs:
	@tail -f /tmp/backend.log 2>/dev/null || echo "No backend log found at /tmp/backend.log"

# ============================================================================
# Testing
# ============================================================================

# Run all E2E scenarios
test:
	@echo "Running all E2E evaluation scenarios..."
	PYTHONPATH=src uv run python tests/e2e/scripts/evaluate_scenarios.py

# Run single scenario (usage: make test-one S=en_customer_service)
test-one:
ifndef S
	@echo "Usage: make test-one S=<scenario_name>"
	@echo "Example: make test-one S=en_customer_service"
	@exit 1
endif
	@echo "Running scenario: $(S)"
	PYTHONPATH=src uv run python tests/e2e/scripts/evaluate_scenarios.py $(S)

# Run scenarios matching pattern (usage: make test-pattern P="*_customer_service")
test-pattern:
ifndef P
	@echo "Usage: make test-pattern P=<pattern>"
	@echo "Example: make test-pattern P='*_customer_service'"
	@exit 1
endif
	@echo "Running scenarios matching: $(P)"
	PYTHONPATH=src uv run python tests/e2e/scripts/evaluate_scenarios.py --pattern "$(P)"

# Run unit tests
test-unit:
	@echo "Running unit tests..."
	uv run pytest tests/ -v --ignore=tests/e2e

# ============================================================================
# Code Quality
# ============================================================================

lint:
	@echo "Running ruff linter..."
	uv run ruff check src/

format:
	@echo "Formatting code with ruff..."
	uv run ruff format src/

typecheck:
	@echo "Running type checker..."
	uv run ty check src/

security:
	@echo "Running security scan..."
	uv run bandit -r src/

# Run all checks
check: lint format typecheck security
	@echo "All checks passed!"

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files

# ============================================================================
# Smoke Tests
# ============================================================================

smoke-asr:
	@echo "Running ASR smoke test..."
	PYTHONPATH=src uv run python -m app.scripts.asr_smoke

smoke-mt:
	@echo "Running MT smoke test..."
	PYTHONPATH=src uv run python -m app.scripts.mt_smoke

smoke-pipeline:
	@echo "Running pipeline smoke test..."
	PYTHONPATH=src uv run python -m app.scripts.pipeline_smoke

smoke: smoke-asr smoke-mt smoke-pipeline

# ============================================================================
# Docker
# ============================================================================

DOCKER_REGISTRY := ghcr.io
DOCKER_IMAGE := $(DOCKER_REGISTRY)/johannhartmann/linguagap
DOCKER_TAG ?= latest

docker: docker-build docker-push

docker-build:
	@echo "Building Docker image: $(DOCKER_IMAGE):$(DOCKER_TAG)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-push:
	@echo "Pushing Docker image: $(DOCKER_IMAGE):$(DOCKER_TAG)"
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run:
	@echo "Running Docker container..."
	docker run --rm -it --gpus all \
		-p 8000:8000 \
		-e HF_TOKEN \
		-v $(HOME)/.cache/huggingface:/data/hf \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

# Build with custom tag (usage: make docker-tag T=v1.0.0)
docker-tag:
ifndef T
	@echo "Usage: make docker-tag T=<tag>"
	@echo "Example: make docker-tag T=v1.0.0"
	@exit 1
endif
	@echo "Building and pushing: $(DOCKER_IMAGE):$(T)"
	docker build -t $(DOCKER_IMAGE):$(T) .
	docker push $(DOCKER_IMAGE):$(T)

# ============================================================================
# Utilities
# ============================================================================

# Verify imports work
verify:
	@echo "Verifying imports..."
	PYTHONPATH=src uv run python -c "from app.main import app; print('OK: imports work')"

# Install dev dependencies
install:
	uv sync --only-dev

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf tests/reports/*.json tests/reports/*.md
	rm -rf __pycache__ src/**/__pycache__ tests/**/__pycache__
	rm -rf .pytest_cache .ruff_cache
	@echo "Done."

# Clean test audio cache (regenerates on next run)
clean-audio:
	@echo "Cleaning TTS audio cache..."
	rm -rf tests/fixtures/e2e_audio/*.wav
	@echo "Done."
