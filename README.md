# LinguaGap

Real-time speech transcription and translation system using GPU-accelerated ASR and MT models.

## Prerequisites

- Docker and Docker Compose v2
- NVIDIA GPU with compatible driver (>= 525.60.13 recommended)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

### Verify NVIDIA Container Toolkit

```bash
# Check if nvidia-smi works in a container
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi
```

## Quick Start

### GPU Smoke Test

Verify GPU access through Docker Compose:

```bash
docker compose up gpu-smoke
```

**Success looks like:** NVIDIA-SMI table printed showing your GPU(s), container exits with code 0.

Example output:
```
gpu-smoke-1  | +-----------------------------------------------------------------------------------------+
gpu-smoke-1  | | NVIDIA-SMI 560.xx.xx    Driver Version: 560.xx.xx    CUDA Version: 12.8               |
gpu-smoke-1  | |-----------------------------------------+------------------------+----------------------+
gpu-smoke-1  | | GPU  Name                 ...           |                        |                      |
gpu-smoke-1  | +=========================================+========================+======================+
gpu-smoke-1  | |   0  NVIDIA GeForce RTX ...             |                        |                      |
gpu-smoke-1  | +-----------------------------------------+------------------------+----------------------+
gpu-smoke-1 exited with code 0
```

## Troubleshooting

### "could not select device driver" error

Ensure NVIDIA Container Toolkit is installed:
```bash
# Ubuntu/Debian
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### GPU not visible

1. Check host driver: `nvidia-smi`
2. Check Docker daemon config includes nvidia runtime
3. Verify toolkit installation: `nvidia-ctk --version`
