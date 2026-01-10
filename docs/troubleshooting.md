# Troubleshooting

Common issues and solutions.

## Model Loading Issues

### Models Not Downloading

**Symptom:** Startup hangs or fails during model download.

**Solutions:**

1. Check network connectivity:
   ```bash
   curl -I https://huggingface.co
   ```

2. Verify HuggingFace cache permissions:
   ```bash
   docker compose exec backend ls -la /data/hf
   ```

3. Check disk space:
   ```bash
   df -h
   ```

4. Try manual download:
   ```bash
   docker compose exec backend python -c "
   from huggingface_hub import hf_hub_download
   hf_hub_download('Qwen/Qwen3-14B-GGUF', 'Qwen3-14B-Q4_K_M.gguf')
   "
   ```

### Model Load Timeout

**Symptom:** Pod restarts due to failed health checks during model warmup.

**Solution:** Increase probe initial delay:

```yaml
livenessProbe:
  initialDelaySeconds: 300  # 5 minutes
readinessProbe:
  initialDelaySeconds: 300
```

### Out of Memory During Load

**Symptom:** OOM kill during model loading.

**Solutions:**

1. Use smaller model:
   ```bash
   MT_MODEL_REPO=Qwen/Qwen3-4B-GGUF
   MT_MODEL_FILE=Qwen3-4B-Q4_K_M.gguf
   ```

2. Increase memory limits:
   ```yaml
   resources:
     limits:
       memory: "32Gi"
   ```

## GPU Issues

### CUDA Not Available

**Symptom:** Error "CUDA not available" or model runs on CPU.

**Diagnosis:**

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base nvidia-smi

# Check container GPU
docker compose exec backend python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions:**

1. Install NVIDIA Container Toolkit:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. Check compose.yaml GPU config:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             capabilities: [gpu]
             count: 1
   ```

### CUDA Out of Memory

**Symptom:** "CUDA out of memory" error during inference.

**Solutions:**

1. Reduce model size (see above)

2. Reduce context window:
   ```bash
   MT_N_CTX=1024
   ```

3. Use smaller batch size (if applicable)

4. Check for GPU memory leaks:
   ```bash
   watch -n 1 nvidia-smi
   ```

### Blackwell GPU Compatibility

**Symptom:** Errors on RTX 50-series or Blackwell GPUs.

**Solution:** Use float16 compute type:

```bash
ASR_COMPUTE_TYPE=float16
```

## WebSocket Issues

### Connection Refused

**Symptom:** Cannot connect to WebSocket endpoint.

**Diagnosis:**

```bash
# Check service is running
curl http://localhost:8000/health

# Check WebSocket endpoint
wscat -c ws://localhost:8000/ws
```

**Solutions:**

1. Verify port binding:
   ```bash
   docker compose ps
   netstat -tlnp | grep 8000
   ```

2. Check firewall rules

3. Verify ingress configuration (Kubernetes)

### Connection Drops

**Symptom:** WebSocket disconnects during streaming.

**Possible Causes:**

1. **Timeout:** Increase timeouts in proxy/ingress
2. **Memory:** Container OOM - increase limits
3. **Network:** Unstable connection - check logs

**Ingress Timeout Fix (Traefik):**

```yaml
annotations:
  traefik.ingress.kubernetes.io/router.middlewares: default-timeout@kubernetescrd
```

### No Audio Processing

**Symptom:** Audio sent but no transcription returned.

**Diagnosis:**

```bash
# Check logs for errors
docker compose logs -f backend | grep -i error

# Test ASR manually
docker compose exec backend python -m app.scripts.asr_smoke
```

**Solutions:**

1. Verify audio format (must be PCM16 16kHz mono)
2. Check config message was sent first
3. Verify source language is supported

## Translation Issues

### Translations Not Appearing

**Symptom:** Transcription works but no translations.

**Diagnosis:**

```bash
# Test MT manually
docker compose exec backend python -m app.scripts.mt_smoke

# Check MT queue in logs
docker compose logs backend | grep -i translation
```

**Solutions:**

1. Verify target language is supported
2. Check MT model loaded successfully
3. Ensure segments are being finalized

### Poor Translation Quality

**Symptom:** Translations are incorrect or nonsensical.

**Solutions:**

1. Use larger model:
   ```bash
   MT_MODEL_REPO=Qwen/Qwen3-14B-GGUF
   MT_MODEL_FILE=Qwen3-14B-Q4_K_M.gguf
   ```

2. Increase context window:
   ```bash
   MT_N_CTX=4096
   ```

3. Check source language detection

## Kubernetes Issues

### Pod Not Starting

**Symptom:** Pod stuck in Pending or CrashLoopBackOff.

**Diagnosis:**

```bash
# Get pod status
kubectl describe pod -n linguagap -l app.kubernetes.io/name=linguagap

# Check events
kubectl get events -n linguagap --sort-by='.lastTimestamp'
```

**Common Issues:**

1. **GPU not available:**
   ```bash
   kubectl describe node | grep -A5 nvidia
   ```

2. **PVC not bound:**
   ```bash
   kubectl get pvc -n linguagap
   ```

3. **Image pull error:**
   ```bash
   kubectl get events -n linguagap | grep -i pull
   ```

### ArgoCD Sync Failed

**Symptom:** ArgoCD shows sync failed or out of sync.

**Diagnosis:**

```bash
argocd app get linguagap
argocd app diff linguagap
```

**Solutions:**

1. Manual sync:
   ```bash
   argocd app sync linguagap --force
   ```

2. Check Helm values syntax

3. Verify chart templates render:
   ```bash
   helm template chart/
   ```

## Performance Issues

### High Latency

**Symptom:** Significant delay between speech and transcription.

**Tuning Options:**

```bash
# Faster updates
TICK_SEC=0.25

# Quicker finalization
STABILITY_SEC=0.75

# Smaller window
WINDOW_SEC=4.0
```

### Segments Disappearing

**Symptom:** Text appears briefly then vanishes.

**Solution:** Increase grace period:

```bash
LIVE_SEGMENT_GRACE_TICKS=5
```

### Memory Growing

**Symptom:** Container memory usage increases over time.

**Diagnosis:**

```bash
# Monitor memory
docker stats

# Check for leaks
docker compose exec backend python -c "
import tracemalloc
tracemalloc.start()
# ... run test ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')[:10]
for stat in top_stats:
    print(stat)
"
```

## Getting Help

### Collect Debug Information

```bash
# System info
uname -a
nvidia-smi
docker version

# Container logs
docker compose logs backend > backend.log 2>&1

# Pod logs (Kubernetes)
kubectl logs -n linguagap deployment/linguagap > pod.log 2>&1
kubectl describe pod -n linguagap -l app.kubernetes.io/name=linguagap > pod-describe.log
```

### Report Issues

Open an issue at [GitHub Issues](https://github.com/johannhartmann/linguagap/issues) with:

1. Description of the problem
2. Steps to reproduce
3. Expected vs actual behavior
4. Debug information (logs, system info)
5. Configuration (environment variables, values.yaml)
