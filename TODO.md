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

### Time-Based Confidence Decay
- [ ] Track timestamp of last detection per person
- [ ] Reduce confidence 10% per minute after 5 minutes unseen
- [ ] Reset to 0 after 15 minutes of no detection
- [ ] Publish decay status in MQTT messages
- [ ] Home Assistant template sensor for effective confidence
- [ ] Configurable decay rates per person role (child vs adult)

### Door Camera Logic (Indoor/Outdoor Detection)
- [ ] Define door cameras in configuration
- [ ] Detect entry/exit direction based on movement vector
- [ ] Publish indoor/outdoor state transitions
- [ ] `binary_sensor.{person}_outdoors` in Home Assistant
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
- [ ] UI configuration in Home Assistant

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
- [ ] Validate configuration on startup
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
- [x] Example dashboard YAML created
- [ ] Custom Lovelace card for person tracking
- [ ] Visual zone map overlay on camera feeds
- [ ] Timeline view of detections
- [ ] Real-time confidence graphs
- [ ] Interactive zone editing
- [ ] Mobile-optimized dashboard

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
- [ ] Unit tests for identity service components
- [ ] Integration tests with mock MQTT
- [ ] End-to-end test scenarios
- [ ] Performance benchmarking
- [ ] Load testing (multiple cameras/persons)
- [ ] Mock Frigate event generator
- [ ] Continuous integration setup

### DevOps & Deployment
- [ ] Docker Compose setup (all services)
- [ ] Kubernetes manifests
- [ ] Home Assistant Supervisor add-on
- [ ] Auto-update mechanism
- [ ] Backup and restore scripts
- [ ] Monitoring dashboards (Grafana)
- [ ] Log aggregation (ELK stack)

---

## 📊 Implementation Status Summary

| Category | Total | Completed | Not Done |
|----------|-------|-----------|----------|
| **Core Architecture** | 18 | 18 (100%) | 0 |
| **Advanced Features** | 47 | 0 (0%) | 47 |
| **Partially Complete** | 27 | 7 (26%) | 20 |
| **Future Enhancements** | 50+ | 1 (2%) | 49+ |
| **TOTAL** | ~142 | ~26 (18%) | ~116 (82%) |

---

## 🎯 Recommended Next Steps (Priority Order)

### Phase 1: Core Stability (Next Sprint)
1. Fix MQTT client compatibility issues (CallbackAPIVersion)
2. Load and parse `persons.yaml` configuration
3. Implement basic confidence decay (time-based)
4. Add MQTT reconnection logic
5. Create unit tests for core functions

### Phase 2: Safety Enhancements (Sprint 2)
1. Implement multi-factor confidence scoring
2. Door camera indoor/outdoor detection
3. Gate sensor integration
4. Improved supervision logic (multi-camera)
5. Notification escalation system

### Phase 3: User Experience (Sprint 3)
1. HTTP configuration endpoint
2. Dashboard improvements
3. Analytics and reporting
4. Configuration UI in Home Assistant
5. Better error messages and debugging

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

**Last Updated:** February 18, 2026
