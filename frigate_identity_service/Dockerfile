ARG BUILD_FROM=python:3.11-slim-bookworm
FROM ${BUILD_FROM}

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

# Build arg: set to "true" to install GPU-capable PyTorch instead of CPU-only.
# Default is CPU-only, which is required for Home Assistant Add-on deployments.
# GPU deployments: docker build --build-arg USE_GPU=true .
ARG USE_GPU=false

# Copy requirements files
COPY requirements.txt requirements-cpu.txt ./

# Install Python dependencies
RUN if [ "$USE_GPU" = "true" ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu || \
      (echo "CPU wheel index failed, falling back to PyPI..." && pip install --no-cache-dir torch torchvision); \
      pip install --no-cache-dir -r requirements-cpu.txt; \
    fi

# Copy all Python modules
COPY identity_service.py embedding_store.py reid_model.py matcher.py mqtt_utils.py ./

# Pre-download model weights at build time so startup is instant.
# When torchreid is available the default OSNet model is used; otherwise
# fall back to ResNet50 from torchvision.
RUN python - <<'EOF'
try:
    from torchreid.utils import FeatureExtractor
    print("Pre-downloading OSNet (osnet_x1_0) weights via torchreid...")
    FeatureExtractor(model_name="osnet_x1_0", model_path="", device="cpu")
    print("OSNet weights cached successfully.")
except Exception as e:
    print(f"torchreid not available ({e}), pre-downloading ResNet50 weights...")
    import torch.nn as nn
    from torchvision.models import resnet50, ResNet50_Weights
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(resnet.children())[:-1])
    print("ResNet50 weights cached successfully.")
EOF

# Startup script (also used as the Home Assistant Add-on entry point)
COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD ["/run.sh"]