# LinguaGap

Real-time speech transcription and translation system using GPU-accelerated ASR and MT models.

## Features

- Real-time speech-to-text using faster-whisper (CTranslate2)
- Real-time translation to German using M2M100
- WebSocket streaming with segment-based updates
- Web interface with microphone capture
- GPU acceleration with CUDA 12.8

## Prerequisites

- Docker and Docker Compose v2
- NVIDIA GPU with compatible driver (>= 525.60.13 recommended)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

### Verify NVIDIA Container Toolkit

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi
```

## Quick Start

### 1. GPU Smoke Test

```bash
docker compose up gpu-smoke
```

### 2. Start Backend

```bash
docker compose up --build backend
```

### 3. Open Web Interface

Navigate to http://localhost:8000 in your browser.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| ASR_MODEL | deepdml/faster-whisper-large-v3-turbo-ct2 | Whisper model for ASR |
| ASR_DEVICE | cuda | Device for ASR (cuda/cpu) |
| ASR_COMPUTE_TYPE | int8_float16 | Compute type for ASR |
| MT_MODEL | michaelfeil/ct2fast-m2m100_418M | M2M100 model for translation |
| MT_DEVICE | cuda | Device for MT (cuda/cpu) |
| MT_COMPUTE_TYPE | int8_float16 | Compute type for MT |
| WINDOW_SEC | 8.0 | Transcription window in seconds |
| TICK_SEC | 0.5 | Update interval in seconds |
| STABILITY_SEC | 1.25 | Segment finalization threshold |
| MAX_BUFFER_SEC | 30.0 | Maximum audio buffer size |
| HF_HOME | /data/hf | HuggingFace cache directory |

### Recommended GPU Compute Types

- **int8_float16**: Best balance of speed and quality (default)
- **float16**: Higher quality, slightly slower
- **int8**: Fastest, lower quality

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Web interface |
| /health | GET | Health check |
| /metrics | GET | Performance metrics |
| /asr_smoke | GET | ASR smoke test |
| /mt_smoke | GET | MT smoke test |
| /transcribe_translate | POST | File upload transcription |
| /ws | WebSocket | Real-time streaming |

## Supported Languages

Albanian (sq), Arabic (ar), Bulgarian (bg), Croatian (hr), English (en), French (fr), German (de), Hungarian (hu), Italian (it), Persian (fa), Polish (pl), Romanian (ro), Russian (ru), Serbian (sr), Spanish (es), Turkish (tr), Ukrainian (uk)

## Model Cache

Models are cached in a Docker volume (`hf-cache`) mounted at `/data/hf`. This persists across container restarts.

To clear the cache:
```bash
docker volume rm linguagap_hf-cache
```

## Troubleshooting

### "could not select device driver" error

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### GPU not visible

1. Check host driver: `nvidia-smi`
2. Check Docker daemon config includes nvidia runtime
3. Verify toolkit installation: `nvidia-ctk --version`

### Slow first request

The first request loads models into GPU memory. Subsequent requests are fast.
Models are warmed up on server startup to minimize cold start latency.

### Out of GPU memory

- Use a smaller ASR model: `ASR_MODEL=tiny`
- Use CPU for MT: `MT_DEVICE=cpu`

### WebSocket connection drops

- Check firewall settings
- Ensure stable network connection
- Check server logs: `docker compose logs backend`

## Development

### Run smoke tests

```bash
# ASR test
docker compose exec backend uv run python -m app.scripts.asr_smoke

# MT test
docker compose exec backend uv run python -m app.scripts.mt_smoke

# Pipeline test
docker compose exec backend uv run python -m app.scripts.pipeline_smoke

# Streaming test
docker compose exec backend uv run python -m app.scripts.stream_client
```

### Check metrics

```bash
curl http://localhost:8000/metrics
```
