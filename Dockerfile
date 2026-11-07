# Training image for the minimal PFN code in `src/` (GPU via the NVIDIA Container Toolkit).
#
# No CUDA base image is needed: on Linux, the locked torch wheels (uv.lock) ship their own CUDA 13
# runtime libraries as pip packages; the container toolkit only injects the host's NVIDIA driver,
# which must support CUDA 13 (driver >= 580).
#
# Build (on an Apple Silicon Mac, cross-build for the Linux/x86_64 GPU machine):
#   docker buildx build --platform linux/amd64 -t pfn-train .
# Run on the GPU machine (the trained model is written to the mounted ./outputs directory):
#   docker run --rm --gpus all -v "$PWD/outputs:/app/outputs" pfn-train \
#       --epochs 50 --steps_per_epoch 200 --save_path outputs/model.pt
# A YAML config can be mounted and passed with `--config`, e.g.
#   -v "$PWD/configs:/app/configs" pfn-train --config configs/run.yaml --save_path outputs/model.pt

FROM python:3.12-slim-bookworm

# Pin uv to a known version (same as used locally to create uv.lock).
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached layer; only rebuilt when pyproject.toml / uv.lock change).
# --frozen: install exactly what uv.lock specifies, fail instead of re-resolving.
# --no-dev: skip the `dev` dependency group (jupyter, seaborn); optional extras are skipped by default.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --frozen --no-install-project --no-dev

ENV PATH="/app/.venv/bin:$PATH"

COPY src ./src

ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--help"]
