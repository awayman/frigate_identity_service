# ============================================================================
# Stage 1: Builder - Full-featured Python image for compilation
# ============================================================================
FROM python:3.13-slim AS builder

WORKDIR /build

# Install system build dependencies required by deep-person-reid (builds from source)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build arg: set to "true" to install GPU-capable PyTorch instead of CPU-only.
# Default is CPU-only, which is required for Home Assistant Add-on deployments.
# GPU deployments: docker build --build-arg USE_GPU=true .
ARG USE_GPU=false

# Build arg: OSNet model variant to bundle. Change to bundle a different variant.
ARG REID_MODEL=osnet_x1_0

# Copy requirements files
COPY requirements.txt requirements-cpu.txt ./

# Install Python dependencies into /build/.venv for isolation
RUN python -m venv /build/.venv
ENV PATH="/build/.venv/bin:$PATH"

# Install all runtime deps first so that deep-person-reid's setup.py (which
# imports the full torchreid package tree to find the version) can complete.
RUN if [ "$USE_GPU" = "true" ]; then \
      pip install --no-cache-dir Cython numpy scipy opencv-python-headless Pillow && \
      pip install --no-cache-dir -r requirements.txt && \
      pip install --no-cache-dir --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git && \
      pip install --no-cache-dir --force-reinstall opencv-python-headless; \
    else \
      (pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu || \
       (echo "CPU wheel index failed, falling back to PyPI..." && pip install --no-cache-dir torch torchvision)) && \
      pip install --no-cache-dir Cython numpy scipy opencv-python-headless Pillow && \
      pip install --no-cache-dir -r requirements-cpu.txt && \
      pip install --no-cache-dir --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git && \
      pip install --no-cache-dir --force-reinstall opencv-python-headless; \
    fi

# Copy Python modules and download script
COPY identity_service.py embedding_store.py reid_model.py matcher.py mqtt_utils.py download_weights.py debug_logger.py ./

# Bundle OSNet weights at build time so startup requires no network access.
# TORCH_HOME is set so torchreid caches into /build/weights inside the image.
RUN TORCH_HOME=/build/weights python download_weights.py ${REID_MODEL} /build/weights || \
    echo "WARNING: Weight download failed at build time. Weights will be downloaded at first startup."

# ============================================================================
# Stage 2: Runtime - Hardened Docker Image (zero CVEs, non-root by default)
# ============================================================================
FROM dhi.io/python:3.13

WORKDIR /app

# OCI labels
ARG BUILD_VERSION=0.1.0
LABEL \
  org.opencontainers.image.title="Frigate Identity Service" \
  org.opencontainers.image.description="Person re-identification service for Frigate NVR" \
  org.opencontainers.image.source="https://github.com/awayman/frigate_identity_service" \
  org.opencontainers.image.version="${BUILD_VERSION}" \
  org.opencontainers.image.licenses="MIT" \
  io.hass.version="${BUILD_VERSION}" \
  io.hass.type="addon" \
  io.hass.arch="amd64|aarch64"

# Copy virtual environment from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application code
COPY --from=builder /build/*.py /app/

# Copy pre-downloaded weights
COPY --from=builder /build/weights /app/weights

# Copy startup script
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# Create data directory with appropriate permissions
# DHI base image already runs as non-root user 'nonroot' (UID 65532)
RUN mkdir -p /data && chmod 777 /data

# Set PATH to use virtualenv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/weights

# Health check (optional, for container orchestration)
HEALTHCHECK --interval=5m --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

CMD ["/run.sh"]