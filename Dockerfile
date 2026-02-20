FROM python:3.11-slim-bookworm

WORKDIR /app

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
      pip install --no-cache-dir -r requirements-cpu.txt; \
    fi

# Copy all Python modules
COPY identity_service.py embedding_store.py reid_model.py matcher.py mqtt_utils.py ./

# Startup script (also used as the Home Assistant Add-on entry point)
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# Set default environment variables (can be overridden at runtime)
ENV MQTT_BROKER=localhost
ENV MQTT_PORT=1883
ENV MQTT_TOPIC=identity/person/recognized

CMD ["/run.sh"]