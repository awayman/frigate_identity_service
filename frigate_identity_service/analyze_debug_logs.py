#!/usr/bin/env python3
"""
Analyze Frigate Identity Service debug logs to diagnose misidentifications.

This script processes debug log files created by the identity service and generates
comprehensive reports with statistics, visualizations, and snapshot comparisons.

Usage:
    python analyze_debug_logs.py --debug-path /data/debug --start-date 2026-02-25 --output-dir ./reports

Output:
    - HTML report with embedded snapshots and metrics
    - CSV summary for spreadsheet analysis
    - JSON aggregated metrics
"""

import argparse
import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class DebugLogAnalyzer:
    """Analyzes debug logs from frigate_identity_service."""

    def __init__(self, debug_path: str):
        """
        Initialize analyzer.

        Args:
            debug_path: Path to debug directory (/data/debug)
        """
        self.debug_path = Path(debug_path)
        self.snapshots_dir = self.debug_path / "snapshots"
        self.logs_dir = self.debug_path / "logs"

        if not self.logs_dir.exists():
            raise ValueError(
                f"Debug logs directory not found at {self.logs_dir}. "
                "Is debug logging enabled?"
            )

        logger.info(f"Analyzing debug logs from {self.debug_path}")

    def load_logs(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Load all debug log files for the given date range.

        Args:
            start_date: Start date (YYYY-MM-DD) or None for all
            end_date: End date (YYYY-MM-DD) or None for all
            timestamp: "2026-02-25T14:30:22.123456"

        Returns:
            Dict with keys: facial_recognition, reid_matches, correlation_issues
        """
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = None

        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        else:
            end_dt = None

        logs = {
            "facial_recognition": [],
            "reid_matches": [],
            "correlation_issues": [],
        }

        # Load facial recognition logs
        for log_file in self.logs_dir.glob("*_facial_recognition.jsonl"):
            if start_dt or end_dt:
                file_date = self._extract_date_from_filename(log_file.name)
                if file_date:
                    if start_dt and file_date < start_dt:
                        continue
                    if end_dt and file_date >= end_dt:
                        continue

            with open(log_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            logs["facial_recognition"].append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {log_file}")

        # Load ReID match logs
        for log_file in self.logs_dir.glob("*_reid_matches.jsonl"):
            if start_dt or end_dt:
                file_date = self._extract_date_from_filename(log_file.name)
                if file_date:
                    if start_dt and file_date < start_dt:
                        continue
                    if end_dt and file_date >= end_dt:
                        continue

            with open(log_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            logs["reid_matches"].append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {log_file}")

        # Load correlation issue logs
        for log_file in self.logs_dir.glob("*_correlation_issues.jsonl"):
            if start_dt or end_dt:
                file_date = self._extract_date_from_filename(log_file.name)
                if file_date:
                    if start_dt and file_date < start_dt:
                        continue
                    if end_dt and file_date >= end_dt:
                        continue

            with open(log_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            logs["correlation_issues"].append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {log_file}")

        logger.info(
            f"Loaded {len(logs['facial_recognition'])} facial recognition events, "
            f"{len(logs['reid_matches'])} ReID events, "
            f"{len(logs['correlation_issues'])} correlation issues"
        )

        return logs

    def analyze_metrics(self, logs: Dict[str, List[Dict]]) -> Dict:
        """
        Calculate statistics from the logs.

        Returns:
            Dict with metrics
        """
        facial_recs = logs["facial_recognition"]
        reid_matches = logs["reid_matches"]
        reid_no_matches = [r for r in reid_matches if r.get("match_found") is False]
        reid_with_matches = [
            r for r in reid_matches if r.get("match_found") is not False
        ]
        correlation_issues = logs["correlation_issues"]

        # Group by camera and person
        persons_by_camera = defaultdict(set)
        cameras = set()
        for event_list in [facial_recs, reid_with_matches]:
            for event in event_list:
                camera = event.get("camera", "unknown")
                person_id = event.get(
                    "person_id", event.get("chosen_person_id", "unknown")
                )
                cameras.add(camera)
                persons_by_camera[camera].add(person_id)

        # Calculate confidence distributions
        facial_rec_confidences = [
            e.get("confidence", 0) for e in facial_recs if "confidence" in e
        ]
        reid_confidences = [
            e.get("chosen_similarity", 0)
            for e in reid_with_matches
            if "chosen_similarity" in e
        ]

        # Identify potential misidentifications (ReID near threshold)
        near_threshold = [
            e for e in reid_with_matches if 0.55 <= e.get("chosen_similarity", 0) < 0.70
        ]

        metrics = {
            "total_events": len(facial_recs) + len(reid_matches),
            "facial_recognition_events": len(facial_recs),
            "reid_matching_events": len(reid_with_matches),
            "reid_no_match_events": len(reid_no_matches),
            "reid_no_match_rate": (
                len(reid_no_matches) / len(reid_matches) if reid_matches else 0
            ),
            "correlation_issues_count": len(correlation_issues),
            "unique_cameras": len(cameras),
            "cameras": list(cameras),
            "unique_persons": sum(len(p) for p in persons_by_camera.values()),
            "persons_by_camera": {k: list(v) for k, v in persons_by_camera.items()},
            "facial_recognition_confidences": {
                "mean": (
                    sum(facial_rec_confidences) / len(facial_rec_confidences)
                    if facial_rec_confidences
                    else 0
                ),
                "min": min(facial_rec_confidences, default=0),
                "max": max(facial_rec_confidences, default=1),
            },
            "reid_confidences": {
                "mean": (
                    sum(reid_confidences) / len(reid_confidences)
                    if reid_confidences
                    else 0
                ),
                "min": min(reid_confidences, default=0),
                "max": max(reid_confidences, default=1),
            },
            "potential_misidentifications_count": len(near_threshold),
            "potential_misidentifications": [
                {
                    "timestamp": e.get("timestamp"),
                    "camera": e.get("camera"),
                    "person_id": e.get("chosen_person_id"),
                    "confidence": e.get("chosen_similarity"),
                    "all_matches": e.get("all_matches"),
                }
                for e in near_threshold
            ],
        }

        return metrics

    def generate_html_report(
        self, metrics: Dict, output_path: str, include_snapshots: bool = True
    ):
        """
        Generate an HTML report with metrics and optional snapshots.

        Args:
            metrics: Metrics dict from analyze_metrics()
            output_path: Path to write HTML file
            include_snapshots: Whether to embed snapshots in HTML
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Frigate Identity Debug Report</title>",
            "<style>",
            "  body { font-family: Arial, sans-serif; margin: 20px; }",
            "  h1 { color: #333; }",
            "  .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; border-left: 4px solid #2196F3; }",
            "  .metric-value { font-size: 24px; font-weight: bold; color: #2196F3; }",
            "  .metric-label { color: #666; font-size: 12px; }",
            "  table { border-collapse: collapse; width: 100%; }",
            "  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "  th { background-color: #2196F3; color: white; }",
            "  .confidence-low { color: #ff9800; font-weight: bold; }",
            "  .confidence-medium { color: #2196F3; }",
            "  .confidence-high { color: #4caf50; }",
            "  .snapshot { max-width: 400px; margin: 10px 0; border: 1px solid #ddd; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Frigate Identity Service Debug Report</h1>",
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]

        # Summary metrics
        html_parts.extend(
            [
                "<h2>Summary</h2>",
                f"<div class='metric'><div class='metric-label'>Total Events</div><div class='metric-value'>{metrics['total_events']}</div></div>",
                f"<div class='metric'><div class='metric-label'>Facial Recognition Events</div><div class='metric-value'>{metrics['facial_recognition_events']}</div></div>",
                f"<div class='metric'><div class='metric-label'>ReID Matches</div><div class='metric-value'>{metrics['reid_matching_events']}</div></div>",
                f"<div class='metric'><div class='metric-label'>ReID No-Match Rate</div><div class='metric-value'>{metrics['reid_no_match_rate']:.1%}</div></div>",
                f"<div class='metric'><div class='metric-label'>Unique Cameras</div><div class='metric-value'>{metrics['unique_cameras']}</div></div>",
                f"<div class='metric'><div class='metric-label'>Unique Persons</div><div class='metric-value'>{metrics['unique_persons']}</div></div>",
                f"<div class='metric'><div class='metric-label'>Multi-Person Correlation Issues</div><div class='metric-value'>{metrics['correlation_issues_count']}</div></div>",
            ]
        )

        # Confidence analysis
        html_parts.extend(
            [
                "<h2>Confidence Analysis</h2>",
                "<h3>Facial Recognition Confidence</h3>",
                f"<p>Mean: {metrics['facial_recognition_confidences']['mean']:.3f}, "
                f"Min: {metrics['facial_recognition_confidences']['min']:.3f}, "
                f"Max: {metrics['facial_recognition_confidences']['max']:.3f}</p>",
                "<h3>ReID Confidence</h3>",
                f"<p>Mean: {metrics['reid_confidences']['mean']:.3f}, "
                f"Min: {metrics['reid_confidences']['min']:.3f}, "
                f"Max: {metrics['reid_confidences']['max']:.3f}</p>",
            ]
        )

        # Potential misidentifications
        if metrics["potential_misidentifications_count"] > 0:
            html_parts.extend(
                [
                    "<h2>Potential Misidentifications (Confidence 0.55-0.70)</h2>",
                    f"<p>{metrics['potential_misidentifications_count']} events found near threshold</p>",
                    "<table>",
                    "<tr><th>Timestamp</th><th>Camera</th><th>Person</th><th>Confidence</th><th>Top Matches</th></tr>",
                ]
            )

            for m in metrics["potential_misidentifications"][:50]:  # Limit to 50
                matches_str = ", ".join(
                    [
                        f"{match['person_id']} ({match['similarity']:.3f})"
                        for match in m.get("all_matches", [])[:3]
                    ]
                )
                html_parts.append(
                    f"<tr><td>{m.get('timestamp', 'N/A')}</td>"
                    f"<td>{m.get('camera', 'N/A')}</td>"
                    f"<td>{m.get('person_id', 'N/A')}</td>"
                    f"<td><span class='confidence-low'>{m.get('confidence', 0):.3f}</span></td>"
                    f"<td>{matches_str}</td></tr>"
                )

            html_parts.append("</table>")

        # Cameras and persons
        html_parts.extend(
            [
                "<h2>Cameras and Persons</h2>",
                "<table>",
                "<tr><th>Camera</th><th>Unique Persons</th></tr>",
            ]
        )

        for camera in sorted(metrics["cameras"]):
            persons = metrics["persons_by_camera"].get(camera, [])
            html_parts.append(
                f"<tr><td>{camera}</td><td>{len(persons)}: {', '.join(sorted(persons))}</td></tr>"
            )

        html_parts.extend(
            [
                "</table>",
                "</body>",
                "</html>",
            ]
        )

        html_content = "\n".join(html_parts)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report written to {output_path}")

    def generate_csv_report(self, logs: Dict[str, List[Dict]], output_path: str):
        """
        Generate CSV summary for spreadsheet analysis.

        Args:
            logs: Logs dict from load_logs()
            output_path: Path to write CSV file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "Timestamp",
                    "Type",
                    "Camera",
                    "Person ID",
                    "Confidence",
                    "Source",
                    "Notes",
                ]
            )

            # Facial recognition events
            for event in logs["facial_recognition"]:
                writer.writerow(
                    [
                        event.get("timestamp", ""),
                        "Facial Recognition",
                        event.get("camera", ""),
                        event.get("person_id", ""),
                        f"{event.get('confidence', 0):.3f}",
                        "Frigate",
                        "",
                    ]
                )

            # ReID events
            for event in logs["reid_matches"]:
                match_found = event.get("match_found", True)
                chosen_sim = event.get("chosen_similarity")
                if chosen_sim is None:
                    chosen_sim = 0
                writer.writerow(
                    [
                        event.get("timestamp", ""),
                        "ReID Match" if match_found else "ReID No-Match",
                        event.get("camera", ""),
                        event.get("chosen_person_id", "N/A"),
                        f"{chosen_sim:.3f}",
                        "ReID Model",
                        (
                            f"Best match: {event.get('best_similarity', 0):.3f} (threshold: {event.get('threshold', 0):.3f})"
                            if not match_found
                            else ""
                        ),
                    ]
                )

        logger.info(f"CSV report written to {output_path}")

    def generate_json_metrics(self, metrics: Dict, output_path: str):
        """
        Export aggregated metrics as JSON.

        Args:
            metrics: Metrics dict from analyze_metrics()
            output_path: Path to write JSON file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"JSON metrics written to {output_path}")

    @staticmethod
    def _extract_date_from_filename(filename: str) -> Optional[datetime]:
        """Extract date from filename format YYYY-MM-DD_*.jsonl"""
        try:
            date_str = filename.split("_")[0]
            return datetime.strptime(date_str, "%Y-%m-%d")
        except (IndexError, ValueError):
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Frigate Identity Service debug logs"
    )
    parser.add_argument(
        "--debug-path",
        default="/data/debug",
        help="Path to debug directory (default: /data/debug)",
    )
    parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD) for analysis (default: all)",
    )
    parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD) for analysis (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="./debug_reports",
        help="Output directory for reports (default: ./debug_reports)",
    )

    args = parser.parse_args()

    try:
        analyzer = DebugLogAnalyzer(args.debug_path)

        # Load logs
        logs = analyzer.load_logs(args.start_date, args.end_date)

        # Analyze
        metrics = analyzer.analyze_metrics(logs)

        # Generate reports
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analyzer.generate_html_report(
            metrics, str(output_dir / "report.html"), include_snapshots=True
        )
        analyzer.generate_csv_report(logs, str(output_dir / "summary.csv"))
        analyzer.generate_json_metrics(metrics, str(output_dir / "metrics.json"))

        logger.info(f"All reports generated in {output_dir}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
