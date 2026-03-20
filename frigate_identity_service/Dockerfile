# ============================================================================
# Stage 1: Builder - Python 3.12 with build dependencies
# ============================================================================
FROM python:3.12-slim AS builder

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

# Install Python dependencies directly (no venv needed for multi-stage build)

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
COPY identity_service.py embedding_store.py reid_model.py matcher.py mqtt_utils.py download_weights.py debug_logger.py snapshot_crop.py ./

# Bundle OSNet weights at build time so startup requires no network access.
# TORCH_HOME is set so torchreid caches into /build/weights inside the image.
RUN TORCH_HOME=/build/weights python download_weights.py ${REID_MODEL} /build/weights || \
    echo "WARNING: Weight download failed at build time. Weights will be downloaded at first startup."

# ============================================================================
# Stage 2: Runtime - Python 3.12 slim with manual hardening
# ============================================================================
FROM python:3.12-slim

WORKDIR /app

# **IMPORTANT: Running as root for Home Assistant Add-on compatibility**
#
# Home Assistant Supervisor writes /data/options.json with restrictive
# permissions (mode 0600, owned by root UID 0). If this container runs as
# a non-root user, it will get "Permission denied" when trying to read
# /data/options.json and cannot load user configuration.
#
# This is a known gotcha in HA add-on development:
# - DO NOT add USER <username> at the end of this Dockerfile
# - DO NOT create a non-root user for HA add-on deployments
# - If deploying outside HA (e.g., Docker, K8s), see the commented non-root
#   section below and uncomment it for security hardening.

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

# Copy Python packages from builder (system site-packages)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /build/*.py /app/

# Copy pre-downloaded weights
COPY --from=builder /build/weights /app/weights

# Copy startup script and make executable
COPY --chmod=755 run.sh /run.sh

# Create data directory for persistence
RUN mkdir -p /data && chmod 777 /data

# ============================================================================
# USER CONFIGURATION - CHOOSE ONE BASED ON DEPLOYMENT MODE
# ============================================================================
#
# FOR HOME ASSISTANT ADD-ON (default, enabled below):
#   - Run as root (UID 0)
#   - Can read /data/options.json (HA Supervisor config)  
#   - HA containers are isolated, security risk is minimal
#   - This is the standard pattern for all official HA add-ons
#
# FOR STANDALONE DOCKER/K8S DEPLOYMENT (optional):
#   - Uncomment the non-root section below
#   - Creates unprivileged user 'frigate_identity' (UID 1000)
#   - Better security posture for internet-facing deployments
#   - ONLY use this if NOT deploying as HA add-on
#
# DO NOT use non-root for HA add-on mode - it will break config loading!
# ============================================================================

# NOTE: Currently running as root for HA add-on compatibility (see above)
# RUN groupadd -r frigate_identity && \
#     useradd -r -g frigate_identity -u 1000 frigate_identity && \
#     chown -R frigate_identity:frigate_identity /app /data && \
#     chmod 755 /app && \
#     chmod 777 /data
#
# USER frigate_identity

# Python environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/weights

# Health check (optional, for container orchestration)
HEALTHCHECK --interval=5m --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

CMD ["/run.sh"]