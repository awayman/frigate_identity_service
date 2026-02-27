FROM python:3.12-slim-bookworm

WORKDIR /app

# Install required packages
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    paho-mqtt==2.1.0 \
    Pillow>=10.0.0 \
    numpy>=1.24.0

# Copy mock Frigate script
COPY mock_frigate.py .

# Run the mock Frigate service
CMD ["python", "mock_frigate.py"]
