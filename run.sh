#!/usr/bin/env sh
# Startup script for the Frigate Identity Service.
# When running as a Home Assistant Add-on, /data/options.json is written by
# the Supervisor and is read automatically by identity_service.py.
set -e

exec python3 /app/identity_service.py
