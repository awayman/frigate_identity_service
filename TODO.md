# Frigate Identity Service - TODO List

Comprehensive list of discussed features and their implementation status.

---

## ✅ COMPLETED - Core Architecture

### Identity Service
- [x] Two-tier snapshot architecture (MQTT for speed + API for accuracy)
- [x] Subscribe to Frigate native MQTT topics (`frigate/+/+/update`, `frigate/+/person/snapshot`)
- [x] API snapshot fetching from Frigate with caching
- [x] Camera-based correlation queue for matching snapshots to persons
- [x] Republish person-specific snapshots to `identity/snapshots/{person_id}`
- [x] Pass-through of Frigate zones, event IDs, bounding boxes
- [x] Vehicle detection (car/truck snapshot subscriptions)
- [x] Environment configuration via `.env` file
- [x] Configurable snapshot correlation window
- [x] Configurable cache TTL for API snapshots

### Home Assistant Integration
- [x] Updated `sensor.py` with two sensors:
  - `sensor.frigate_identity_last_person` (most recent detection)
  - `sensor.frigate_identity_all_persons` (tracks all detected persons)
- [x] Enhanced attributes: zones, event_id, snapshot_url, source, timestamps
- [x] MQTT camera entity support for live snapshots
- [x] Proper handling of Frigate's sub_label (face recognition)

### Blueprints (4 total)
- [x] `child_danger_zone_alert.yaml` - Alert when child in dangerous zone
- [x] `vehicle_children_outside_alert.yaml` - Vehicle + children safety
- [x] `supervision_detection.yaml` - Binary sensor for supervision tracking
- [x] `notification_action_handlers.yaml` - Handle notification action buttons

### Configuration Files
- [x] `.env` / `.env.example` - Environment configuration
- [x] `persons.yaml` - Person roles, ages, supervision requirements
- [x] `requirements.txt` - Updated with `requests` library

### Documentation
- [x] `QUICK_START.md` - Step-by-step setup guide
- [x] `CONFIGURATION_EXAMPLES.md` - Advanced configuration examples
- [x] `blueprints/.../README.md` - Blueprint usage guide
- [x] `examples/dashboard.yaml` - Example Lovelace dashboard
- [x] Updated README files for both repositories
- [x] `test_system.py` - System validation script

---

## 🔴 NOT IMPLEMENTED - Advanced Features

### Multi-Factor Confidence Scoring
- [ ] Composite confidence calculation formula:
  - [ ] Base ReID similarity (50%)
  - [ ] Temporal factor (20%) - based on time since last seen
  - [ ] Facial recognition bonus (20%)
  - [ ] Continuity factor (10%) - consecutive frames/same camera
- [ ] Implement weighted scoring in `identity_service.py`
- [ ] Publish composite scores in identity events
- [ ] Different thresholds for facial vs ReID vs continuity

### Matcher Refactor — Vectorized Matching & Tests
- [x] Implement recency weighting in matcher (linear/exponential/none)
- [x] Add optional confidence weighting in matcher
- [ ] Ensure embeddings are L2-normalized at extraction (`reid_model.extract_embedding`)
- [ ] Build per-person embedding matrices and recency weight vectors
- [ ] Replace per-pair scipy loops with NumPy vectorized dot-products for matching
- [ ] Preserve per-embedding recency weighting (max over weighted similarities)
- [ ] Add unit tests validating equivalence between loop and vectorized implementations
- [ ] Add performance/benchmark tests for large galleries (e.g., 1k+ embeddings)
- [x] Expose recency controls via runtime config (`EMBEDDING_MAX_AGE_HOURS`, `RECENCY_DECAY_MODE`, `RECENCY_WEIGHT_FLOOR`)
- [ ] Keep scipy fallback when NumPy unavailable
- [ ] Document matching assumptions in `IMPLEMENTATION_SUMMARY.md`

### Time-Based Confidence Decay
- [ ] Track timestamp of last detection per person
- [ ] Reduce confidence 10% per minute after 5 minutes unseen
- [ ] Reset to 0 after 15 minutes of no detection
- [ ] Publish decay status in MQTT messages
- [ ] (MOVED_TO_HA) Home Assistant template sensor for effective confidence
- [ ] Configurable decay rates per person role (child vs adult)

### Door Camera Logic (Indoor/Outdoor Detection)
- [ ] Define door cameras in configuration
- [ ] Detect entry/exit direction based on movement vector
- [ ] Publish indoor/outdoor state transitions
- [ ] (MOVED_TO_HA) `binary_sensor.{person}_outdoors` in Home Assistant
- [ ] Reset confidence when person goes inside for 10+ minutes
- [ ] Handle multiple door cameras (front, back, garage)
- [ ] Account for false positives (person lingers near door)

### Person Roles & Safety Rules System
- [ ] Load `persons.yaml` at startup (currently just example file)
- [ ] Validate person roles against detected persons
- [ ] Per-person danger zone configuration
- [ ] Age-based zone safety rules (5-year-old vs 10-year-old)
- [ ] Publish role information with identity events
- [ ] Dynamic role updates without restart
- [ ] (MOVED_TO_HA) UI configuration in Home Assistant

### Embedding Quality Validation
- [ ] Check embedding vector for anomalies
- [ ] Detect blurry or partial images
- [ ] Image quality scoring (brightness, contrast, sharpness)
- [ ] Reject low-quality embeddings from storage
- [ ] Log quality metrics per person
- [ ] Minimum quality threshold configuration
- [ ] Alert when person needs better facial training data

### Persistent Tracking History
- [ ] SQLite database for detection history
- [ ] Store: person_id, timestamp, camera, zone, confidence, source
- [ ] Configurable retention period (7-30 days)
- [ ] Query API for historical data
- [ ] Export to CSV/JSON
- [ ] Clear old data automatically
- [ ] Database migration system for schema updates

### HTTP Configuration Endpoint
- [ ] Flask or FastAPI web server
- [ ] GET `/api/config` - Current configuration
- [ ] POST `/api/config` - Update thresholds
- [ ] GET `/api/persons` - List all persons
- [ ] POST `/api/persons/{person_id}/embedding` - Refresh embedding
- [ ] GET `/api/stats` - System statistics
- [ ] GET `/api/health` - Health check endpoint
- [ ] Authentication/authorization for API access

### MQTT Reconnection & Error Recovery
- [x] Startup connect retry logic (`MQTT_CONNECT_RETRIES`, `MQTT_CONNECT_RETRY_DELAY`)
- [ ] Automatic reconnection on disconnect
- [ ] Exponential backoff retry logic
- [ ] Queue messages during disconnection
- [ ] Republish queued messages on reconnect
- [ ] Frigate API retry logic with backoff
- [ ] Health monitoring and alerting
- [ ] Graceful degradation when Frigate unavailable

### Advanced Supervision Logic
- [ ] Multi-camera supervision (adult on adjacent camera)
- [ ] Zone-based supervision requirements
- [ ] Time-of-day supervision rules (stricter at dusk)
- [ ] Supervision confidence scoring
- [ ] Multiple supervision strategies (proximity, line-of-sight, same zone)
- [ ] "Responsible adult" identification beyond trusted list
- [ ] Supervision timeout grace periods

---

## 🟡 PARTIALLY IMPLEMENTED - Needs Enhancement

### Vehicle Safety System
- [x] Basic vehicle detection (subscribe to car/truck topics)
- [x] Publish vehicle events
- [x] Blueprint for vehicle + children alerts
- [ ] Gate state integration logic (currently manual input_boolean)
- [ ] Physical gate sensor support (Zigbee/Z-Wave)
- [ ] Vehicle tracking (car entering vs exiting)
- [ ] Driveway occupancy detection
- [ ] Alert escalation (gate open = critical)
- [ ] Vehicle identification (recognize family cars vs strangers)

### Snapshot Correlation
- [x] Basic temporal correlation (2-second window)
- [x] Camera-based person queue
- [x] Multi-person conflict detection
- [ ] Spatial correlation (bounding box matching)
- [ ] Improved multi-person handling (assign snapshots to correct person)
- [ ] Confidence scoring for correlation quality
- [ ] Fallback strategies when correlation fails
- [ ] Zone-based snapshot filtering

### Person Tracking Continuity
- [x] Track last camera per person
- [x] Store event_id for API lookups
- [ ] Track person transitions (camera A → camera B)
- [ ] Maintain identity across cameras without face visible
- [ ] Path prediction (movement patterns)
- [ ] Stuck detection (person stationary >N minutes)
- [ ] Activity classification (playing, walking, running)

### Configuration Management
- [x] Environment variables via `.env`
- [x] Example `persons.yaml` created
- [ ] Load and parse `persons.yaml` in identity service
- [x] Validate configuration on startup
- [ ] Watch for config file changes (hot reload)
- [ ] Configuration schema validation
- [ ] Migration tools for config updates
- [ ] Import/export configuration

---

## 🟢 FUTURE ENHANCEMENTS - Not Discussed in Detail

### Analytics & Reporting
- [ ] Daily outdoor time per child
- [ ] Safety incident log (danger zone entries)
- [ ] Zone dwell time statistics
- [ ] Supervision coverage percentage
- [ ] Movement heatmaps
- [ ] Weekly/monthly reports
- [ ] Export to PDF or email

### Predictive Safety Alerts
- [ ] Machine learning on movement patterns
- [ ] Predict trajectory toward dangerous zones
- [ ] Proactive alerts before entering danger zone
- [ ] Anomaly detection (unusual behavior)
- [ ] Risk scoring based on multiple factors
- [ ] Historical pattern analysis

### Advanced Notifications
- [ ] Notification priority levels (info, warning, critical)
- [ ] Escalation to phone call if not acknowledged
- [ ] SMS fallback for critical alerts
- [ ] Multi-recipient notifications (Mom + Dad)
- [ ] Notification groups (family, neighbors)
- [ ] Rich media in notifications (video clips)
- [ ] TTS announcements on smart speakers

### Dashboard & UI
- [x] (MOVED_TO_HA) Example dashboard YAML created
- [ ] (MOVED_TO_HA) Custom Lovelace card for person tracking
- [ ] (MOVED_TO_HA) Visual zone map overlay on camera feeds
- [ ] (MOVED_TO_HA) Timeline view of detections
- [ ] (MOVED_TO_HA) Real-time confidence graphs
- [ ] (MOVED_TO_HA) Interactive zone editing
- [ ] (MOVED_TO_HA) Mobile-optimized dashboard

### Multi-Home & Cloud Sync
- [ ] Deploy to multiple locations (grandparents' house)
- [ ] Separate person databases per location
- [ ] Centralized alerting dashboard
- [ ] Person roaming between locations
- [ ] Cloud backup of embeddings
- [ ] Remote monitoring portal

### Pet Tracking
- [ ] Extend to family pets (dogs, cats)
- [ ] Pet-specific zones and rules
- [ ] Pet + gate escape detection
- [ ] Pet activity monitoring
- [ ] Multiple pet tracking
- [ ] Pet vs person classification

### Smart Speaker Integration
- [ ] "Alexa, where are the kids?" queries
- [ ] Voice announcements for alerts
- [ ] Manual supervision via voice command
- [ ] Status queries ("Is Alice outside?")
- [ ] Custom Alexa skill or Google Assistant action

### Testing & Quality Assurance
- [x] Basic system test script (`test_system.py`)
- [x] Unit tests for identity service components
- [x] Integration tests with mock Frigate/MQTT flows
- [x] End-to-end test scenarios (real Frigate test suite)
- [ ] Performance benchmarking
- [ ] Load testing (multiple cameras/persons)
- [x] Mock Frigate event generator (`frigate_mock/mock_frigate.py`)
- [x] Continuous integration setup (`.github/workflows/ci.yml`)

### DevOps & Deployment
- [x] Docker Compose setup (all services)
- [ ] Kubernetes manifests
- [x] Home Assistant Supervisor add-on
- [ ] Auto-update mechanism
- [ ] Backup and restore scripts
- [ ] Monitoring dashboards (Grafana)
- [ ] Log aggregation (ELK stack)

---

## 📊 Implementation Status Summary

Status counts in this document have drifted over time and should be treated as directional, not exact.

- Core architecture is complete and stable.
- Test and release automation are now substantially more complete than shown in older counts.
- Main gaps remain advanced safety logic, richer tracking semantics, and runtime resiliency features.

---

## 🧹 Obsolete or Moved (Current Design)

The following items are no longer primary work for this repository's current scope and design:

- Home Assistant UI/entity concerns should be tracked in `frigate_identity_ha` (for example: `binary_sensor.{person}_outdoors`, HA template sensors, Lovelace-specific UX items).
- HTTP configuration endpoint (`/api/config`, `/api/health`, etc.) is currently de-prioritized in favor of env vars + Home Assistant add-on options (`/data/options.json`).
- `RECENCY_DECAY_HOURS` naming is obsolete; recency behavior is now configured via `EMBEDDING_MAX_AGE_HOURS`, `RECENCY_DECAY_MODE`, and `RECENCY_WEIGHT_FLOOR`.
- Checklist rows explicitly prefixed with `(MOVED_TO_HA)` should be planned and delivered in the Home Assistant integration repository, not this service repository.

---

## 🎯 Recommended Next Steps (Priority Order)

### Phase 1: Core Stability (Next Sprint)
1. ~~Fix MQTT client compatibility issues (CallbackAPIVersion)~~ ✅ **COMPLETED**
2. Decide and document `persons.yaml` direction for service (implement here vs keep HA-only)
3. Implement basic confidence decay (time-based)
4. Add full MQTT disconnect recovery (backoff + reconnect behavior)
5. Add benchmark coverage for matcher performance

### Phase 2: Safety Enhancements (Sprint 2)
1. Implement multi-factor confidence scoring
2. Door camera indoor/outdoor detection
3. Gate sensor integration
4. Improved supervision logic (multi-camera)
5. Notification escalation system

### Phase 3: User Experience (Sprint 3)
1. Better error messages and debugging ergonomics
2. Analytics/reporting foundations for troubleshooting
3. Optional lightweight health/stats endpoint (only if deployment needs it)
4. Document HA-repo ownership for UI/dashboard work
5. Improve operator docs for retention/recovery tuning

### Phase 4: Advanced Features (Sprint 4+)
1. Persistent tracking history (SQLite)
2. Predictive safety alerts (ML)
3. Multi-home support
4. Pet tracking
5. Smart speaker integration

---

## 🚀 What's Ready to Use Right Now

The following are **fully functional** and ready for production use:

✅ **Core person identification** (Frigate face recognition + ReID)
✅ **Two-tier snapshot system** (fast display + accurate embeddings)
✅ **Vehicle detection** (car/truck alerts)
✅ **4 Home Assistant Blueprints** (easy automation setup)
✅ **MQTT camera entities** (live person snapshots)
✅ **Basic safety alerts** (dangerous zone detection)
✅ **Manual supervision override** (input_boolean)
✅ **Comprehensive documentation** (3 guides + examples)
✅ **System validation** (test_system.py)

**You can start using these features immediately** after configuring your environment!

---

## 📝 Notes

- Many "not implemented" features were discussed architecturally but intentionally left for future development
- The current implementation focuses on **core functionality** and **ease of use** (blueprints)
- Advanced features like ML predictions and analytics require more planning and testing
- Some features depend on hardware (gate sensors, door cameras) that may not be available yet
- The modular architecture makes it easy to add these features incrementally

---

**Last Updated:** March 3, 2026
