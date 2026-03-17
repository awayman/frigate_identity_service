"""HTML report generator for real Frigate integration tests.

Generates interactive HTML reports showing:
- Snapshots fetched from Frigate API
- Matched person IDs
- Confidence scores
- Summary statistics
"""

import base64
from urllib.parse import quote
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a person identification match."""

    event_id: str
    snapshot_bytes: bytes
    matched_person: str
    confidence: float
    camera: str = "unknown"
    timestamp: Optional[str] = None
    is_mismatch: bool = False
    expected_person: Optional[str] = None
    source: str = "unknown"  # "facial_recognition", "reid_model", or "unknown"
    alternatives: Optional[List[tuple]] = (
        None  # List of (person, confidence) alternatives
    )


class MatchReport:
    """Accumulates and generates HTML reports for person identification matches."""

    def __init__(self, frigate_host: Optional[str] = None):
        """Initialize an empty match report."""
        self.matches: List[MatchResult] = []
        self.mismatches: List[MatchResult] = []
        self.notes: List[str] = []
        self.total_processed = 0
        self.frigate_host = frigate_host.rstrip("/") if frigate_host else None

    def add_note(self, message: str):
        """Add a note/warning message to be displayed in the HTML report."""
        if not message:
            return
        self.notes.append(message)

    def _frigate_snapshot_url(self, event_id: str) -> Optional[str]:
        """Build a direct Frigate snapshot URL for an event, if host is configured."""
        if not self.frigate_host:
            return None
        safe_event_id = quote(event_id, safe="")
        return f"{self.frigate_host}/api/events/{safe_event_id}/snapshot.jpg?crop=1&quality=95"

    def add_match(
        self,
        event_id: str,
        snapshot_bytes: bytes,
        person_id: str,
        confidence: float,
        camera: str = "unknown",
        timestamp: Optional[str] = None,
        source: str = "unknown",
        alternatives: Optional[List[tuple]] = None,
    ):
        """Add a successful person identification match to the report.

        Args:
            event_id: Frigate event ID
            snapshot_bytes: Raw JPEG image bytes
            person_id: Identified person ID
            confidence: Confidence score (0.0-1.0)
            camera: Camera name where detection occurred
            timestamp: ISO timestamp of the detection
            source: Identification source ("facial_recognition", "reid_model", or "unknown")
            alternatives: Optional list of (person_name, confidence) tuples for alternative matches
        """
        match = MatchResult(
            event_id=event_id,
            snapshot_bytes=snapshot_bytes,
            matched_person=person_id,
            confidence=confidence,
            camera=camera,
            timestamp=timestamp or datetime.now().isoformat(),
            source=source,
            alternatives=alternatives,
        )
        self.matches.append(match)
        self.total_processed += 1
        logger.info(
            f"Added match: {person_id} (confidence: {confidence:.2%}) on {camera} [source: {source}]"
        )

    def add_mismatch(
        self,
        event_id: str,
        snapshot_bytes: bytes,
        matched_person: str,
        confidence: float,
        expected_person: str,
        camera: str = "unknown",
        timestamp: Optional[str] = None,
        source: str = "unknown",
    ):
        """Add a misidentified person (ground truth available).

        Args:
            event_id: Frigate event ID
            snapshot_bytes: Raw JPEG image bytes
            matched_person: Person ID that was matched
            confidence: Confidence score (0.0-1.0)
            expected_person: Correct person ID (ground truth)
            camera: Camera name where detection occurred
            timestamp: ISO timestamp of the detection
            source: Identification source ("facial_recognition", "reid_model", or "unknown")
        """
        mismatch = MatchResult(
            event_id=event_id,
            snapshot_bytes=snapshot_bytes,
            matched_person=matched_person,
            confidence=confidence,
            camera=camera,
            timestamp=timestamp or datetime.now().isoformat(),
            is_mismatch=True,
            expected_person=expected_person,
            source=source,
        )
        self.mismatches.append(mismatch)
        self.total_processed += 1
        logger.warning(
            f"Mismatch: expected {expected_person}, got {matched_person} "
            f"(confidence: {confidence:.2%}) [source: {source}]"
        )

    def generate_html(self, output_path: Path) -> Path:
        """Generate an interactive HTML report.

        Creates a self-contained HTML file with:
        - Inline base64-encoded images (no external file references)
        - Person identification results with confidence badges
        - Summary statistics (total processed, accuracy, etc.)
        - Color-coded results (green for matches, red for mismatches)

        Args:
            output_path: Path where HTML file should be saved

        Returns:
            Path to the generated HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        accuracy = self._calculate_accuracy()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = self._build_html(accuracy, timestamp)

        output_path.write_text(html, encoding="utf-8")
        logger.info(f"Generated HTML report: {output_path}")

        return output_path

    def _calculate_accuracy(self) -> float:
        """Calculate accuracy as (matches / total_processed)."""
        if self.total_processed == 0:
            return 0.0
        return len(self.matches) / self.total_processed

    def _build_html(self, accuracy: float, timestamp: str) -> str:
        """Build the HTML content for the report."""
        accuracy_pct = accuracy * 100

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Frigate Identity Test Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f5f5f5;
            border-bottom: 1px solid #ddd;
        }}
        
        .stat {{
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            margin-top: 5px;
            color: #666;
            font-size: 0.9em;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .match-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }}
        
        .match-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background: white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .match-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }}
        
        .match-card.mismatch {{
            border-left: 4px solid #e74c3c;
        }}
        
        .match-card.match {{
            border-left: 4px solid #27ae60;
        }}
        
        .match-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: #f0f0f0;
        }}
        
        .match-info {{
            padding: 15px;
        }}
        
        .match-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .match-person {{
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        
        .badge.match {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge.mismatch {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .badge.source-facial {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .badge.source-reid {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge.source-unknown {{
            background: #e2e3e5;
            color: #383d41;
        }}
        
        .confidence {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: auto;
        }}
        
        .confidence-high {{
            background: #d4edda;
            color: #155724;
        }}
        
        .confidence-medium {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .confidence-low {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .match-details {{
            font-size: 0.85em;
            color: #666;
            line-height: 1.6;
        }}
        
        .match-details p {{
            margin: 5px 0;
        }}
        
        .mismatch-expected {{
            color: #e74c3c;
            font-weight: bold;
            background: #ffe6e6;
            padding: 5px 8px;
            border-radius: 4px;
            display: inline-block;
            font-size: 0.85em;
        }}
        
        .event-id {{
            font-family: 'Courier New', monospace;
            font-size: 0.75em;
            color: #999;
            margin-top: 5px;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 40px;
            color: #999;
        }}
        
        .empty-state p {{
            font-size: 1.1em;
        }}

        .notice-banner {{
            margin: 0 0 20px 0;
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid #d39e00;
            background: #fff3cd;
            color: #856404;
            font-size: 0.95em;
        }}

        .notice-banner strong {{
            margin-right: 6px;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .match-grid {{
                grid-template-columns: 1fr;
            }}
            
            .stats {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Frigate Identity Test Report</h1>
            <p>Real image integration test results</p>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{self.total_processed}</div>
                <div class="stat-label">Processed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.matches)}</div>
                <div class="stat-label">Matches</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.mismatches)}</div>
                <div class="stat-label">Mismatches</div>
            </div>
            <div class="stat">
                <div class="stat-value">{accuracy_pct:.1f}%</div>
                <div class="stat-label">Accuracy</div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <p style="text-align: center; color: #999; font-size: 0.9em;">
                    Generated {timestamp}
                </p>
            </div>
"""

        # Add notes/warnings section (if any)
        if self.notes:
            html += '<div class="section">\n'
            for note in self.notes:
                html += (
                    '<div class="notice-banner">'
                    "<strong>⚠ Notice:</strong> "
                    f"{note}"
                    "</div>\n"
                )
            html += "</div>\n"

        # Add matches section
        if self.matches:
            html += '<div class="section">\n'
            html += "<h2>✓ Successful Matches</h2>\n"
            html += '<div class="match-grid">\n'
            for match in self.matches:
                html += self._build_match_card(match)
            html += "</div>\n</div>\n"

        # Add mismatches section
        if self.mismatches:
            html += '<div class="section">\n'
            html += "<h2>✗ Mismatches</h2>\n"
            html += '<div class="match-grid">\n'
            for mismatch in self.mismatches:
                html += self._build_match_card(mismatch)
            html += "</div>\n</div>\n"

        # Add empty state if no results
        if not self.matches and not self.mismatches:
            html += """<div class="empty-state">
                <p>No results to display</p>
            </div>"""

        html += """
        </div>
    </div>
</body>
</html>
"""
        return html

    def _build_match_card(self, result: MatchResult) -> str:
        """Build HTML for a single match card."""
        # Convert image bytes to base64 data URI
        img_base64 = base64.b64encode(result.snapshot_bytes).decode("utf-8")
        img_src = f"data:image/jpeg;base64,{img_base64}"

        # Determine confidence badge color
        confidence_class = (
            "confidence-high"
            if result.confidence >= 0.7
            else "confidence-medium"
            if result.confidence >= 0.5
            else "confidence-low"
        )

        # Determine source badge
        source_display = {
            "facial_recognition": ("👤 Facial Recognition", "source-facial"),
            "reid_model": ("🔍 ReID Match", "source-reid"),
            "simulated_learning": ("🎓 Learned (Simulated)", "source-facial"),
            "unknown": ("❓ Unknown", "source-unknown"),
        }.get(result.source, ("Unknown", "source-unknown"))

        source_text, source_class = source_display

        # Build card HTML
        card_class = "mismatch" if result.is_mismatch else "match"
        badge_class = "mismatch" if result.is_mismatch else "match"
        label = "✗ Mismatch" if result.is_mismatch else "✓ Match"

        frigate_snapshot_url = self._frigate_snapshot_url(result.event_id)
        image_html = f'<img src="{img_src}" alt="Snapshot" class="match-image">'
        event_html = f'<div class="event-id">Event: {result.event_id}</div>'

        if frigate_snapshot_url:
            image_html = (
                f'<a href="{frigate_snapshot_url}" target="_blank" rel="noopener noreferrer" '
                f'title="Open event snapshot in Frigate">{image_html}</a>'
            )
            event_html = (
                f'<div class="event-id">Event: '
                f'<a href="{frigate_snapshot_url}" target="_blank" rel="noopener noreferrer">'
                f"{result.event_id}</a></div>"
            )

        html = f"""<div class="match-card {card_class}">
    {image_html}
    <div class="match-info">
        <div class="match-header">
            <div class="match-person">{result.matched_person}</div>
            <span class="badge {badge_class}">{label}</span>
        </div>
        <div style="margin-bottom: 10px;">
            <span class="confidence {confidence_class}">{result.confidence:.1%}</span>
            <span class="badge {source_class}" style="margin-left: 5px;">{source_text}</span>
        </div>
        <div class="match-details">
            <p><strong>Camera:</strong> {result.camera}</p>
            <p><strong>Time:</strong> {result.timestamp}</p>
"""

        if result.is_mismatch and result.expected_person:
            html += f'            <p><strong>Expected:</strong> <span class="mismatch-expected">{result.expected_person}</span></p>\n'

        html += f"""            {event_html}
    """

        # Add alternatives section if available (for ReID matches)
        if result.alternatives and len(result.alternatives) > 0:
            html += '            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee;">\n'
            html += '                <p style="font-size: 0.75em; color: #999; margin-bottom: 5px;"><strong>Alternative matches:</strong></p>\n'
            for alt_person, alt_conf in result.alternatives:
                conf_class = (
                    "confidence-high"
                    if alt_conf >= 0.7
                    else "confidence-medium"
                    if alt_conf >= 0.5
                    else "confidence-low"
                )
                html += f'                <p style="font-size: 0.75em; margin: 2px 0;">• {alt_person}: <span class="{conf_class}" style="padding: 2px 6px; font-size: 0.9em;">{alt_conf:.1%}</span></p>\n'
            html += "            </div>\n"

        html += """        </div>
    </div>
</div>
"""
        return html
