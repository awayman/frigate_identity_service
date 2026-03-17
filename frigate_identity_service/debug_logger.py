import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import base64

logger = logging.getLogger(__name__)


class DebugLogger:
    """
    Logs facial recognition and ReID matching data for debugging misidentifications.

    Captures:
    - Frigate facial recognition snapshots and metadata
    - ReID matching snapshots with similarity scores and match details
    - Multi-person correlation issues
    - Optional: Full embeddings for analysis

    Storage structure:
        /data/debug/
        ├── snapshots/
        │   └── {date}/
        │       ├── {timestamp}_{camera}_{person_id}.jpg        (API snapshots)
        │       └── ...
        └── logs/
            ├── {date}_facial_recognition.jsonl
            ├── {date}_reid_matches.jsonl
            └── {date}_correlation_issues.jsonl

    Access control: Enable/disable via MQTT topic or configuration.
    """

    def __init__(
        self,
        debug_path: str = "debug",
        enabled: bool = False,
        save_embeddings: bool = False,
        retention_days: int = 7,
    ):
        """
        Initialize debug logger.

        Args:
            debug_path: Base directory for debug logs and snapshots
            enabled: Whether debug logging is currently enabled
            save_embeddings: Whether to include embeddings in JSON logs
            retention_days: How many days to retain debug logs before cleanup
        """
        self.debug_path = Path(debug_path)
        self.enabled = enabled
        self.save_embeddings = save_embeddings
        self.retention_days = retention_days

        self.snapshots_dir = self.debug_path / "snapshots"
        self.logs_dir = self.debug_path / "logs"

        # Persistent state file to survive restarts
        self.state_file = self.debug_path / "enabled"

        # Load state from file if it exists
        if self.state_file.exists():
            try:
                self.enabled = self.state_file.read_text().strip().lower() == "true"
                logger.info(f"[DEBUG] Loaded debug state from {self.state_file}")
            except Exception as e:
                logger.warning(f"[DEBUG] Could not load state file: {e}")

        if self.enabled:
            self._ensure_dirs()
            logger.info(f"[DEBUG] Debug logging ENABLED at {self.debug_path}")
        else:
            logger.info(
                f"[DEBUG] Debug logging disabled (base path: {self.debug_path})"
            )

    def _ensure_dirs(self):
        """Create debug directories if they don't exist."""
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def set_enabled(self, enabled: bool):
        """Enable or disable debug logging and persist state."""
        self.enabled = enabled
        if enabled:
            self._ensure_dirs()
        try:
            self.state_file.write_text(str(enabled).lower())
            logger.info(f"[DEBUG] Debug logging {'ENABLED' if enabled else 'DISABLED'}")
        except Exception as e:
            logger.error(f"[DEBUG] Failed to save state: {e}")

    def log_facial_recognition(
        self,
        event_id: str,
        snapshot_base64: str,
        person_id: str,
        camera: str,
        confidence: float,
        zones: List[str],
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Log successful facial recognition from Frigate.

        Args:
            event_id: Frigate event ID
            snapshot_base64: Base64-encoded JPEG snapshot
            person_id: Identified person name
            camera: Camera name
            confidence: Detection confidence (0-1)
            zones: List of zones where person was detected
            timestamp: Event timestamp (seconds since epoch)

        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.enabled:
            return False

        ts = timestamp or __import__("time").time()
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        timestamp_str = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")

        try:
            # Save snapshot
            snapshot_path = (
                self.snapshots_dir
                / date_str
                / f"{timestamp_str}_{camera}_{person_id}.jpg"
            )
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)

            snapshot_bytes = base64.b64decode(snapshot_base64)
            snapshot_path.write_bytes(snapshot_bytes)

            # Log metadata
            log_file = self.logs_dir / f"{date_str}_facial_recognition.jsonl"
            log_entry = {
                "timestamp": datetime.fromtimestamp(ts).isoformat(),
                "event_id": event_id,
                "camera": camera,
                "person_id": person_id,
                "confidence": confidence,
                "zones": zones,
                "snapshot_file": os.path.basename(str(snapshot_path)),
                "source": "frigate_face_recognition",
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.debug(
                f"[DEBUG] Logged facial recognition for {person_id} at {camera}"
            )
            return True

        except Exception as e:
            logger.error(f"[DEBUG] Error logging facial recognition: {e}")
            return False

    def log_reid_match(
        self,
        event_id: str,
        snapshot_base64: str,
        matches: List[Dict],
        chosen_person_id: str,
        chosen_similarity: float,
        camera: str,
        zones: List[str],
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Log successful ReID match.

        Args:
            event_id: Frigate event ID
            snapshot_base64: Base64-encoded JPEG snapshot
            matches: List of dicts with person_id and similarity score
            chosen_person_id: The person identified
            chosen_similarity: Confidence of the match
            camera: Camera name
            zones: List of zones
            timestamp: Event timestamp

        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.enabled:
            return False

        ts = timestamp or __import__("time").time()
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        timestamp_str = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")

        try:
            # Save snapshot
            snapshot_path = (
                self.snapshots_dir
                / date_str
                / f"{timestamp_str}_{camera}_{chosen_person_id}_reid.jpg"
            )
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)

            snapshot_bytes = base64.b64decode(snapshot_base64)
            snapshot_path.write_bytes(snapshot_bytes)

            # Log metadata
            log_file = self.logs_dir / f"{date_str}_reid_matches.jsonl"
            log_entry = {
                "timestamp": datetime.fromtimestamp(ts).isoformat(),
                "event_id": event_id,
                "camera": camera,
                "chosen_person_id": chosen_person_id,
                "chosen_similarity": chosen_similarity,
                "all_matches": matches,
                "zones": zones,
                "snapshot_file": os.path.basename(str(snapshot_path)),
                "source": "reid_model",
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.debug(
                f"[DEBUG] Logged ReID match for {chosen_person_id} (similarity={chosen_similarity:.3f})"
            )
            return True

        except Exception as e:
            logger.error(f"[DEBUG] Error logging ReID match: {e}")
            return False

    def log_reid_no_match(
        self,
        event_id: str,
        snapshot_base64: str,
        matches: List[Dict],
        best_similarity: float,
        threshold: float,
        camera: str,
        zones: List[str],
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Log ReID searches that found no match above threshold (useful for debugging).

        Args:
            event_id: Frigate event ID
            snapshot_base64: Base64-encoded JPEG snapshot
            matches: All matches examined (for debugging)
            best_similarity: Highest similarity found
            threshold: Threshold that was not met
            camera: Camera name
            zones: List of zones
            timestamp: Event timestamp

        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.enabled:
            return False

        ts = timestamp or __import__("time").time()
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        timestamp_str = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")

        try:
            # Save snapshot
            snapshot_path = (
                self.snapshots_dir / date_str / f"{timestamp_str}_{camera}_no_match.jpg"
            )
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)

            snapshot_bytes = base64.b64decode(snapshot_base64)
            snapshot_path.write_bytes(snapshot_bytes)

            # Log metadata
            log_file = self.logs_dir / f"{date_str}_reid_matches.jsonl"
            log_entry = {
                "timestamp": datetime.fromtimestamp(ts).isoformat(),
                "event_id": event_id,
                "camera": camera,
                "person_id": None,
                "chosen_similarity": None,
                "all_matches": matches,
                "best_similarity": best_similarity,
                "threshold": threshold,
                "match_found": False,
                "zones": zones,
                "snapshot_file": os.path.basename(str(snapshot_path)),
                "source": "reid_model",
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.debug(
                f"[DEBUG] Logged ReID no-match (best={best_similarity:.3f}, threshold={threshold:.3f})"
            )
            return True

        except Exception as e:
            logger.error(f"[DEBUG] Error logging ReID no-match: {e}")
            return False

    def log_correlation_issue(
        self,
        camera: str,
        active_persons_count: int,
        queue_state: List[Dict],
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Log multi-person correlation issues (when MQTT snapshot correlation is ambiguous).

        Args:
            camera: Camera name
            active_persons_count: Number of active persons on camera
            queue_state: State of the detection queue for context
            timestamp: Event timestamp

        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.enabled:
            return False

        ts = timestamp or __import__("time").time()
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

        try:
            log_file = self.logs_dir / f"{date_str}_correlation_issues.jsonl"
            log_entry = {
                "timestamp": datetime.fromtimestamp(ts).isoformat(),
                "camera": camera,
                "active_persons_count": active_persons_count,
                "queue_state": queue_state,
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.debug(
                f"[DEBUG] Logged correlation issue on {camera} ({active_persons_count} persons)"
            )
            return True

        except Exception as e:
            logger.error(f"[DEBUG] Error logging correlation issue: {e}")
            return False

    def cleanup_old_logs(self) -> int:
        """
        Remove debug logs older than retention_days.

        Returns:
            Number of days deleted
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0

        try:
            # Clean up snapshots by date
            if self.snapshots_dir.exists():
                for date_dir in self.snapshots_dir.iterdir():
                    if date_dir.is_dir():
                        try:
                            dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                            if dir_date < cutoff_date:
                                shutil.rmtree(date_dir)
                                deleted_count += 1
                                logger.info(
                                    f"[DEBUG] Cleaned up snapshots from {date_dir.name}"
                                )
                        except ValueError:
                            # Directory name doesn't match date format, skip
                            pass

            # Clean up old log files
            if self.logs_dir.exists():
                for log_file in self.logs_dir.glob("*.jsonl"):
                    try:
                        # Extract date from filename (format: YYYY-MM-DD_*.jsonl)
                        date_str = log_file.name.split("_")[0]
                        file_date = datetime.strptime(date_str, "%Y-%m-%d")
                        if file_date < cutoff_date:
                            log_file.unlink()
                            logger.info(f"[DEBUG] Deleted log file {log_file.name}")
                    except (ValueError, IndexError):
                        # File doesn't match expected format, skip
                        pass

            if deleted_count > 0:
                logger.info(
                    f"[DEBUG] Cleanup completed: deleted {deleted_count} days of snapshots (older than {self.retention_days} days)"
                )

            return deleted_count

        except Exception as e:
            logger.error(f"[DEBUG] Error during cleanup: {e}")
            return 0

    def get_storage_usage(self) -> Dict:
        """
        Calculate current storage usage of debug logs.

        Returns:
            Dict with 'total_mb', 'snapshots_mb', 'logs_mb'
        """
        try:
            total_size = 0

            if self.snapshots_dir.exists():
                for file in self.snapshots_dir.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size

            if self.logs_dir.exists():
                for file in self.logs_dir.glob("*.jsonl"):
                    if file.is_file():
                        total_size += file.stat().st_size

            total_mb = total_size / (1024 * 1024)

            return {
                "total_mb": round(total_mb, 2),
                "total_bytes": total_size,
            }

        except Exception as e:
            logger.error(f"[DEBUG] Error calculating storage usage: {e}")
            return {"total_mb": 0, "total_bytes": 0}
