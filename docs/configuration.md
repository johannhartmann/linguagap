# Configuration

LinguaGap is configured via environment variables and Helm values.

## Environment Variables

### ASR Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Whisper model identifier |
| `ASR_DEVICE` | `cuda` | Device for ASR (`cuda` or `cpu`) |
| `ASR_COMPUTE_TYPE` | `int8_float16` | Compute precision (`float16`, `int8_float16`, `int8`) |

### MT Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MT_MODEL_REPO` | `Qwen/Qwen3-14B-GGUF` | HuggingFace model repository |
| `MT_MODEL_FILE` | `Qwen3-14B-Q4_K_M.gguf` | GGUF model filename |
| `MT_N_GPU_LAYERS` | `-1` | GPU layers (-1 = all) |
| `MT_N_CTX` | `2048` | Context window size |

### Streaming Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WINDOW_SEC` | `8.0` | ASR transcription window (seconds) |
| `TICK_SEC` | `0.5` | ASR update interval (seconds) |
| `STABILITY_SEC` | `1.25` | Segment finalization threshold (seconds) |
| `MAX_BUFFER_SEC` | `30.0` | Maximum audio buffer size (seconds) |
| `LIVE_SEGMENT_GRACE_TICKS` | `3` | Ticks before dropping missing segments |

### HuggingFace Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `/data/hf` | HuggingFace cache directory |
| `HF_HUB_DISABLE_TELEMETRY` | `1` | Disable HF telemetry |

## Helm Chart Values

The Kubernetes deployment is configured via `chart/values.yaml`.

### Image Configuration

```yaml
image:
  repository: ghcr.io/johannhartmann/linguagap
  tag: latest
  pullPolicy: Always
```

### Resource Requirements

```yaml
resources:
  requests:
    memory: "12Gi"
    cpu: "2"
    nvidia.com/gpu: "1"
  limits:
    memory: "24Gi"
    cpu: "8"
    nvidia.com/gpu: "1"
```

!!! warning "GPU Required"
    LinguaGap requires an NVIDIA GPU. The Qwen3-14B model needs approximately 10-12GB VRAM.

### Ingress Configuration

```yaml
ingress:
  enabled: true
  className: traefik
  host: linguagap.data.mayflower.tech
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt
  tls:
    enabled: true
    secretName: linguagap-tls
```

### Persistence

```yaml
persistence:
  enabled: true
  size: 20Gi
  storageClassName: longhorn
  accessMode: ReadWriteOnce
```

The persistent volume stores downloaded models from HuggingFace.

### Health Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 120
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 120
  periodSeconds: 10
```

!!! note "Initial Delay"
    The 120-second initial delay allows time for model loading on cold start.

### Full Environment Configuration

```yaml
env:
  ASR_MODEL: deepdml/faster-whisper-large-v3-turbo-ct2
  ASR_DEVICE: cuda
  ASR_COMPUTE_TYPE: float16
  MT_MODEL_REPO: Qwen/Qwen3-14B-GGUF
  MT_MODEL_FILE: Qwen3-14B-Q4_K_M.gguf
  MT_N_GPU_LAYERS: "-1"
  MT_N_CTX: "2048"
  WINDOW_SEC: "8.0"
  TICK_SEC: "0.5"
  STABILITY_SEC: "1.25"
  MAX_BUFFER_SEC: "30.0"
  HF_HOME: /data/hf
  HF_HUB_DISABLE_TELEMETRY: "1"
  PYTHONUNBUFFERED: "1"
```

## Docker Compose Configuration

For local development, use `compose.yaml`:

```yaml
services:
  backend:
    build: .
    ports:
      - "0.0.0.0:8000:8000"
    volumes:
      - hf-cache:/data/hf
    environment:
      - PYTHONUNBUFFERED=1
      - HF_HOME=/data/hf
      - HF_HUB_DISABLE_TELEMETRY=1
      - MT_MODEL_REPO=Qwen/Qwen3-14B-GGUF
      - MT_MODEL_FILE=Qwen3-14B-Q4_K_M.gguf
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: 1

volumes:
  hf-cache:
```

## Model Selection Guide

### ASR Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `faster-whisper-large-v3-turbo-ct2` | ~1.5GB | Fast | High |
| `faster-whisper-large-v3` | ~3GB | Medium | Highest |
| `faster-whisper-medium` | ~1.5GB | Faster | Good |

### MT Models

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| Qwen3-4B-Q4_K_M | ~3GB | Fast | Good |
| Qwen3-8B-Q4_K_M | ~5GB | Medium | Better |
| Qwen3-14B-Q4_K_M | ~10GB | Slower | Best |

## Tuning Recommendations

### Low Latency

For minimal delay:

```bash
TICK_SEC=0.25        # Faster updates
STABILITY_SEC=0.75   # Quicker finalization
WINDOW_SEC=4.0       # Smaller window
```

### Higher Accuracy

For better transcription:

```bash
TICK_SEC=1.0         # Less frequent updates
STABILITY_SEC=2.0    # More stable segments
WINDOW_SEC=15.0      # Larger context
```

### Memory Constrained

For limited GPU memory:

```bash
ASR_COMPUTE_TYPE=int8
MT_MODEL_REPO=Qwen/Qwen3-4B-GGUF
MT_MODEL_FILE=Qwen3-4B-Q4_K_M.gguf
MT_N_CTX=1024
```
