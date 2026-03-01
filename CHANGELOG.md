# Changelog

All notable changes to Frigate Identity Service will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.4.9] - 2026-03-01

### Changed
- Add local testing files to .gitignore
- Downgrade Python version to 3.12 in CI, auto-release, and Dockerfile for compatibility
- Update Python version to 3.13 in CI and release workflows; enhance Dockerfile for multi-stage builds; add configuration validation in identity_service.py; improve config.yaml with type constraints; create .dockerignore for build context management.

## [0.4.8] - 2026-03-01

### Changed
- Refactor configuration handling and improve error logging for MQTT connectivity issues

## [0.4.7] - 2026-03-01

### Fixed
- Fixed environment variable configuration issue where incorrect environment: section syntax caused literal string values to overwrite correctly loaded options from /data/options.json, breaking MQTT connectivity. Removed problematic environment mappings - load_ha_options() function correctly handles option-to-environment-variable conversion. Improved error logging to diagnose configuration loading failures.

## [0.4.5] - 2026-02-27

### Changed
- Remove USER directive for non-root user in Dockerfile

## [0.4.4] - 2026-02-27

### Changed
- Enhance logging for Home Assistant options file handling

## [0.4.3] - 2026-02-27

### Changed
- Add Docker Compose setup and mock Frigate service for testing

## [0.4.2] - 2026-02-26

### Changed
- Merge pull request #39 from awayman/copilot/fix-container-start-error

## [0.4.1] - 2026-02-26

### Fixed
- update Python base image to 3.12 and add debug logger to Dockerfile

## [0.4.0] - 2026-02-26

### Fixed
- update RECENCY_DECAY_HOURS to 12 for improved embedding weight calculation
### Changed
- Add debug logging features for misidentification analysis
- format code for better readability and consistency

## [0.3.2] - 2026-02-23

### Added
- add .venv-1 to .gitignore and update requirements for apscheduler

## [0.3.1] - 2026-02-23

### Added
- remove persons.yaml from .gitignore
- enhance embeddings configuration for container support and persistence
- add functions to retrieve and categorize commits since last tag for changelog updates

## [0.3.0] - 2026-02-22

## [0.2.21] - 2026-02-21

## [0.2.20] - 2026-02-21

## [0.2.19] - 2026-02-21

## [0.2.18] - 2026-02-21

## [0.2.17] - 2026-02-21

## [0.2.16] - 2026-02-21

## [0.2.15] - 2026-02-21

## [0.2.14] - 2026-02-21

## [0.2.13] - 2026-02-21

## [0.2.12] - 2026-02-21

## [0.2.11] - 2026-02-21

## [0.2.10] - 2026-02-21

## [0.2.9] - 2026-02-21

## [0.2.8] - 2026-02-21

## [0.2.7] - 2026-02-21

## [0.2.6] - 2026-02-21

## [0.2.5] - 2026-02-21

## [0.2.4] - 2026-02-21

## [0.2.3] - 2026-02-21

## [0.2.2] - 2026-02-21

## [0.2.1] - 2026-02-21

## [0.2.0] - 2026-02-21

### Changed
- Pre-built Docker images on GHCR for fast installs (no more local PyTorch builds)
- Added CI/CD with GitHub Actions
- Added OCI labels to Dockerfile
- Removed stale root config.json

## [0.1.0]

### Added
- Initial release
- Person re-identification using OSNet ReID model
- MQTT integration with Frigate NVR
- SQLAlchemy-based embedding store
- CPU and GPU support
- Home Assistant Add-on support
