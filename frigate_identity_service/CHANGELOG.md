## 0.6.1

### Fixed
- Add snapshot_crop.py to Dockerfile for complete module inclusion

### Changed
- docs: Update flowchart and sequence diagram in README for improved clarity on identity service processes

## 0.6.0

### Added
- release: Generate release notes from CHANGELOG and create GitHub release
- Add icon and logo images for branding
- Add script to check for broken local Markdown links
- Revise README structure with improved headings and table of contents
- Enhance README with detailed architecture and snapshot flow descriptions

### Fixed
- Update links in README and Implementation Summary for correct file paths

### Changed
- Improve code formatting for readability in identity_service.py, snapshot_crop.py, and test files
- Add ReID model embedding extraction from PIL images and enhance snapshot handling
- Refactor test cases and utility functions for improved readability and consistency
- chore: Remove trailing whitespace in CHANGELOG

﻿## 0.5.9

### Added
- Add flowcharts to README for snapshot flow and sequence diagram
- Add service uptime logging to heartbeat status
- Update TODO list with completed tasks and move Home Assistant related items to separate tracking
- Suppress verbose job execution logs from APScheduler
- Refactor Home Assistant options loading for improved logging and diagnostics

## 0.5.8

### Added
- Update Dockerfile for Home Assistant add-on compatibility and clarify user configuration

## 0.5.7

### Added
- Enhance logging for Home Assistant options loading with detailed diagnostics

## 0.5.6

### Added
- Add mqtt_host alias for MQTT_BROKER and update tests

## 0.5.5

### Added
- Prevent overriding existing environment variables when loading Home Assistant options

## 0.5.4

### Added
- Update requirements to include requests package

## 0.5.3

### Added
- Refactor loading of Home Assistant options to improve environment variable handling and logging

## 0.5.2

### Added
- Improve logging for Home Assistant options loading and MQTT configuration

## 0.5.1

### Added
- Enable confidence weighting for embeddings in configuration
- Enhance event filtering and reporting in integration tests

### Changed
- Raise default `REID_SIMILARITY_THRESHOLD` from 0.6 to 0.75 to reduce borderline ReID misidentifications
- Update docs, environment templates, and configuration examples to reflect the 0.75 default

## 0.5.0

### Added
- Implement embedding retention policies and recency weighting
### Changed
- test: Fix unit tests to use EmbeddingMatcher instance methods

### Added
- Configurable recency weighting system with three decay modes: `linear`, `exponential`, and `none`
- Optional confidence weighting to prioritize high-confidence embeddings
- Comprehensive test suite for recency and confidence weighting (`test_recency_weighting.py`)
- New configuration options: `RECENCY_DECAY_MODE`, `RECENCY_WEIGHT_FLOOR`, `USE_CONFIDENCE_WEIGHTING`

### Changed
- Replace hardcoded midnight embedding clear with configurable retention policy (`age_prune`, `full_clear_daily`, `manual`), defaulting to age-based pruning for better identity continuity
- Recency weighting now dynamically scales with `EMBEDDING_MAX_AGE_HOURS` instead of hardcoded 24-hour decay
- EmbeddingMatcher refactored from static methods to instance-based with configuration
- Weight floor reduced from 0.5 to 0.3 (default) for better differentiation across retention window
- Linear decay now covers full retention period (e.g., 48h) instead of plateauing at 24h

### Improved
- Better weight differentiation for embeddings at different ages within retention window
- More granular control over how embedding age affects matching confidence
- Exponential decay option provides faster initial decay for time-sensitive scenarios

## 0.4.9

### Changed
- Add local testing files to .gitignore
- Downgrade Python version to 3.12 in CI, auto-release, and Dockerfile for compatibility
- Update Python version to 3.13 in CI and release workflows; enhance Dockerfile for multi-stage builds; add configuration validation in identity_service.py; improve config.yaml with type constraints; create .dockerignore for build context management.

## 0.4.8

### Changed
- Refactor configuration handling and improve error logging for MQTT connectivity issues

## 0.4.7

### Fixed
- Fixed environment variable configuration issue where incorrect environment: section syntax caused literal string values to overwrite correctly loaded options from /data/options.json, breaking MQTT connectivity. Removed problematic environment mappings - load_ha_options() function correctly handles option-to-environment-variable conversion. Improved error logging to diagnose configuration loading failures.

## 0.4.5

### Changed
- Remove USER directive for non-root user in Dockerfile

## 0.4.4

### Changed
- Enhance logging for Home Assistant options file handling

## 0.4.3

### Changed
- Add Docker Compose setup and mock Frigate service for testing

## 0.4.2

### Changed
- Merge pull request #39 from awayman/copilot/fix-container-start-error

## 0.4.1

### Fixed
- update Python base image to 3.12 and add debug logger to Dockerfile

## 0.4.0

### Fixed
- update RECENCY_DECAY_HOURS to 12 for improved embedding weight calculation
### Changed
- Add debug logging features for misidentification analysis
- format code for better readability and consistency

## 0.3.2

### Added
- add .venv-1 to .gitignore and update requirements for apscheduler

## 0.3.1

### Added
- remove persons.yaml from .gitignore
- enhance embeddings configuration for container support and persistence
- add functions to retrieve and categorize commits since last tag for changelog updates

## 0.3.0

## 0.2.21

## 0.2.20

## 0.2.19

## 0.2.18

## 0.2.17

## 0.2.16

## 0.2.15

## 0.2.14

## 0.2.13

## 0.2.12

## 0.2.11

## 0.2.10

## 0.2.9

## 0.2.8

## 0.2.7

## 0.2.6

## 0.2.5

## 0.2.4

## 0.2.3

## 0.2.2

## 0.2.1

## 0.2.0

### Changed
- Pre-built Docker images on GHCR for fast installs (no more local PyTorch builds)
- Added CI/CD with GitHub Actions
- Added OCI labels to Dockerfile
- Removed stale root config.json

## 0.1.0

### Added
- Initial release
- Person re-identification using OSNet ReID model
- MQTT integration with Frigate NVR
- SQLAlchemy-based embedding store
- CPU and GPU support
- Home Assistant Add-on support
