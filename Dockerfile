# CUDA 12.9 with cuDNN 9.8+ required for PyTorch 2.8+ and Blackwell GPUs (sm_120)
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 and build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    ca-certificates \
    curl \
    cmake \
    ninja-build \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip3 install --no-cache-dir uv --break-system-packages

# Enable CUDA for llama-cpp-python build with Blackwell (sm_120) support
# FORCE_CUBLAS avoids custom kernel crashes, NO_PINNED for GDDR7 compatibility
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS=1 -DGGML_CUDA_NO_PINNED=1 -DCMAKE_CUDA_ARCHITECTURES=120"
ENV FORCE_CMAKE=1

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Create symlink for CUDA stub library and install dependencies
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 && \
    ln -s /usr/local/cuda/lib64/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so && \
    ldconfig && \
    uv sync --frozen

# Runtime stage - must match builder CUDA version
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 runtime and required libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/

# Copy static files
COPY static/ ./static/

# Set PYTHONPATH to include src directory
ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Run uvicorn directly from venv
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
