FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip3 install --no-cache-dir uv --break-system-packages

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Create venv and install dependencies
RUN uv sync --frozen

# Copy application code
COPY src/ ./src/

# Set PYTHONPATH to include src directory
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Run uvicorn
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
