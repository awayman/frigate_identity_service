"""Quick summary of the test report."""
import re
from pathlib import Path

report_path = Path("tests/output/real_frigate_report.html")
html = report_path.read_text(encoding='utf-8')

# Extract person names and their identification sources
persons_facial = []
persons_reid = []
persons_unknown = []

# Find all person names with their source badges
matches = re.findall(r'<div class="match-person">([^<]+)</div>.*?<span class="badge source-(\w+)"[^>]*>([^<]+)</span>', html, re.DOTALL)

for person, source_class, source_text in matches:
    if 'facial' in source_class:
        persons_facial.append(person)
    elif 'reid' in source_class:
        persons_reid.append(person)
    elif 'unknown' in source_class:
        persons_unknown.append(person)

print("=" * 80)
print("TEST REPORT SUMMARY")
print("=" * 80)
print(f"\nPhase 1 - Facial Recognition (Learning):")
print(f"  Total events: {len(persons_facial)}")
if persons_facial:
    from collections import Counter
    counts = Counter(persons_facial)
    print(f"  Unique persons: {len(counts)}")
    for person, count in sorted(counts.items()):
        print(f"    - {person}: {count} event(s)")

print(f"\nPhase 2 - ReID Matching:")
print(f"  Total events: {len(persons_reid)}")
if persons_reid:
    from collections import Counter
    counts = Counter(persons_reid)
    print(f"  Unique persons matched: {len(counts)}")
    for person, count in sorted(counts.items()):
        print(f"    - {person}: {count} match(es)")

if persons_unknown:
    print(f"\nUnknown/No Match:")
    print(f"  Total: {len(persons_unknown)}")

print(f"\n" + "=" * 80)
print(f"TOTAL: {len(persons_facial) + len(persons_reid) + len(persons_unknown)} events processed")
print("=" * 80)
