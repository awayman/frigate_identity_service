# Changelog

All notable changes to Frigate Identity Service will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

### Removed


## [0.6.1] - 2026-03-20

### Fixed
- Add snapshot_crop.py to Dockerfile for complete module inclusion
### Changed
- docs: Update flowchart and sequence diagram in README for improved clarity on identity service processes
## [0.6.0] - 2026-03-20

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
## [0.5.9] - 2026-03-07

### Added
- Add flowcharts to README for snapshot flow and sequence diagram
- Add service uptime logging to heartbeat status
- Update TODO list with completed tasks and move Home Assistant related items to separate tracking
- Suppress verbose job execution logs from APScheduler
- Refactor Home Assistant options loading for improved logging and diagnostics

## [0.5.8] - 2026-03-04

### Added
- Update Dockerfile for Home Assistant add-on compatibility and clarify user configuration

## [0.5.7] - 2026-03-04

### Added
- Enhance logging for Home Assistant options loading with detailed diagnostics

## [0.5.6] - 2026-03-04

### Added
- Add mqtt_host alias for MQTT_BROKER and update tests

## [0.5.5] - 2026-03-02

### Added
- Prevent overriding existing environment variables when loading Home Assistant options

## [0.5.4] - 2026-03-02

### Added
- Update requirements to include requests package

## [0.5.3] - 2026-03-02

### Added
- Refactor loading of Home Assistant options to improve environment variable handling and logging

## [0.5.2] - 2026-03-02

### Added
- Improve logging for Home Assistant options loading and MQTT configuration

## [0.5.1] - 2026-03-02

### Added
- Enable confidence weighting for embeddings in configuration
- Enhance event filtering and reporting in integration tests

### Changed
- Raise default `REID_SIMILARITY_THRESHOLD` from 0.6 to 0.75 to reduce borderline ReID misidentifications
- Update docs, environment templates, and configuration examples to reflect the 0.75 default

## [0.5.0] - 2026-03-02

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
