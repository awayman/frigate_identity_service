FROM python:3.11-slim-bookworm

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir paho-mqtt

# Copy the Python script
COPY identity_service.py .

# Set environment variables (can be overridden in Home Assistant)
ENV MQTT_BROKER=localhost
ENV MQTT_PORT=1883
ENV MQTT_TOPIC=identity/person/recognized

# Command to run the script
CMD ["python", "identity_service.py"]