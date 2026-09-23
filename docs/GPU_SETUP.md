# GPU Setup for the Ollama / Qwen2.5-VL Condition Pipeline

The `condition/` package calls a local Ollama server (the `ollama` service in
`docker-compose.yml`) running `qwen2.5vl:7b` to extract vehicle condition from
listing photos + text. This document covers getting that service running on
an NVIDIA GPU, and what to expect if it isn't.

Target hardware for this project is a single **NVIDIA GTX 1080 Ti (11GB
VRAM)**. The instructions below are written generically for any NVIDIA GPU +
Docker host, not hardcoded to the 1080 Ti specifically — the dev machine this
was built on has a different, smaller-VRAM card, and the guidance here should
hold for both.

## 1. Prerequisites on the host

- An NVIDIA GPU visible to the host OS (`nvidia-smi` works outside Docker).
- A recent NVIDIA driver. There isn't a single official minimum pinned to
  Ollama's container image, but as a rule of thumb: **driver version r535 or
  newer** is a safe target — that's roughly the generation that lines up with
  CUDA 12.x, which is what recent Ollama container builds are built against.
  If `nvidia-smi` on the host reports a CUDA version of 12.x or newer, you're
  almost certainly fine. Treat this as an approximation, not a hard
  requirement pulled from Ollama's release notes — if in doubt, update to the
  latest driver for your card before troubleshooting further.
- Docker Engine with Compose v2 (`docker compose version`, not the old
  standalone `docker-compose` v1 binary).
- The **NVIDIA Container Toolkit** (`nvidia-ctk`) installed and configured —
  this is what lets a container request `--gpus all` / the Compose
  `deploy.resources.reservations.devices` block and actually see the GPU.

### Installing the NVIDIA Container Toolkit

On Linux hosts (the typical case for a dedicated GPU box; adjust package
manager for your distro — see NVIDIA's own docs for the current exact
commands, package names/repos do shift over time):

```bash
# Add NVIDIA's container toolkit repo, then:
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

On Windows with Docker Desktop + WSL2: install a current NVIDIA driver with
WSL support on the Windows host (not inside WSL), enable the WSL2 backend in
Docker Desktop, and enable "GPU support" under Docker Desktop's resource
settings. Docker Desktop bundles the necessary toolkit wiring for WSL2 GPU
passthrough — you generally don't run `nvidia-ctk` manually in that setup.

## 2. Verify GPU passthrough works

Two checks, from least to most specific:

```bash
# 1. Does Docker itself see the GPU via the NVIDIA runtime at all?
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 2. Does it work through THIS project's compose file / ollama service?
docker compose run --rm ollama nvidia-smi
```

Both should print a normal `nvidia-smi` table (driver version, CUDA version,
GPU name, memory usage). If (1) fails, the problem is the toolkit/driver
setup, not this project's compose file. If (1) succeeds but (2) fails, check
that you're on Compose v2 with a version that understands the
`deploy.resources.reservations.devices` syntax outside swarm mode (recent
Compose v2 releases do), or fall back to the CPU-only override for now (see
below) while you sort that out.

## 3. Bring up the stack and pull the model

```bash
cp .env.example .env
docker compose up -d db ollama
docker compose exec ollama ollama pull qwen2.5vl:7b
```

The pull is a one-time, several-GB download (the model tag itself is ~5-6GB
on disk at Ollama's default Q4_K_M quantization); it's cached in the
`ollama_data` named volume, so you won't re-download it on every
`docker compose up`.

## 4. VRAM footprint and why an 11GB card is enough

`qwen2.5vl:7b` at Ollama's default quantization (Q4_K_M) is approximately:

- **~6GB on disk**
- **~7-8GB resident in VRAM including KV cache** during inference, for the
  image + prompt sizes this pipeline uses (`IMAGE_MAX_DIMENSION_PX=1024`,
  `OLLAMA_NUM_CTX=8192` by default — see `.env.example`)

These are approximations — actual resident VRAM varies with the number/size
of images per request, context length in use, and Ollama version — but they
leave meaningful headroom on an 11GB GTX 1080 Ti (roughly 3-4GB free for the
OS/other processes/driver overhead). An 8GB card can also work but with much
less margin — reduce `MAX_IMAGES_PER_LISTING` and/or `OLLAMA_NUM_CTX` if you
see OOM errors on a smaller card. The 32B/72B Qwen2.5-VL variants are *not* a
supported target for this deployment — they need 20-40GB+ even quantized and
will not fit this hardware.

## 5. Graceful degradation — no GPU, or GPU passthrough not working

You have three ways to run without (working) GPU passthrough, and they
compose:

1. **`OLLAMA_USE_GPU=false`** in your `.env` — this is a config flag consumed
   by the Python side (`config.py`'s `OLLAMA_USE_GPU`); it doesn't change
   Docker's GPU reservation itself, but signals the app-side code not to
   assume/require GPU-accelerated Ollama.
2. **`docker-compose.cpu-only.yml` override** — removes the `deploy:` GPU
   reservation block from the `ollama` service entirely:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.cpu-only.yml up
   ```
   Use this when the GPU reservation block itself causes `docker compose` to
   error (rather than just quietly running without a GPU).
3. **Do nothing** — Ollama detects at runtime whether a GPU is visible to the
   container and **automatically falls back to CPU inference** if not, with
   no configuration required on your part.

Be plain with yourself about the consequence of CPU fallback: it will work
correctly, but a 7B-class vision-language model doing image+text inference on
CPU is **slow** — expect it to take substantially longer per listing than on
GPU (easily an order of magnitude or more, depending on your CPU). This is
fine for a handful of test listings; it is not a realistic way to run the
full condition backfill over a large historical listing set. Get GPU
passthrough working before running `backfill.backfill_conditions` at scale.
