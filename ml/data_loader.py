# =============================================================================
# data_loader.py — Merges the three route NOTAM files into a single dataset
#
# Usage:
#   python data_loader.py
#
# Output:
#   data/merged_notams.txt  — all NOTAMs with route tags injected
#   Prints a summary report of what was loaded
# =============================================================================

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ml.config import (
    RAW_FILES,
    MERGED_FILE,
    NOTAM_SEPARATOR,
    ROUTE_METADATA,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DataLoader")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ParsedNotam:
    """A single NOTAM parsed from a route file."""
    id: str
    number: str
    type: str
    issued: str
    selection_code: str
    location: str
    effective_start: str
    effective_end: str
    classification: str
    account_id: str
    last_updated: str
    icao_location: str
    text: str
    # Injected by the loader — not present in raw files
    route: str = ""
    departure: str = ""
    destination: str = ""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_notam_file(filepath: Path, route: str) -> list[ParsedNotam]:
    """
    Parses a single route NOTAM file into a list of ParsedNotam objects.
    Injects route/departure/destination metadata into each NOTAM.

    Args:
        filepath: Path to the .txt file
        route:    Route key e.g. "OKC_JFK"

    Returns:
        List of ParsedNotam objects
    """
    if not filepath.exists():
        logger.warning(f"File not found, skipping: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split(NOTAM_SEPARATOR) if b.strip()]
    notams = []
    failed = 0

    meta = ROUTE_METADATA[route]

    for block in blocks:
        fields = {}
        text_lines = []
        in_text = False

        for line in block.split("\n"):
            if in_text:
                text_lines.append(line)
            elif line.startswith("Text:"):
                in_text = True
                value = line[len("Text:"):].strip()
                if value:
                    text_lines.append(value)
            else:
                if ": " in line:
                    key, _, value = line.partition(": ")
                    fields[key.strip()] = value.strip()

        fields["Text"] = " ".join(text_lines).strip()

        # Skip empty blocks
        if not fields.get("ID"):
            continue

        try:
            notams.append(ParsedNotam(
                id=fields.get("ID", ""),
                number=fields.get("Number", ""),
                type=fields.get("Type", ""),
                issued=fields.get("Issued", ""),
                selection_code=fields.get("Selection Code", ""),
                location=fields.get("Location", ""),
                effective_start=fields.get("Effective Start", ""),
                effective_end=fields.get("Effective End", ""),
                classification=fields.get("Classification", ""),
                account_id=fields.get("Account ID", ""),
                last_updated=fields.get("Last Updated", ""),
                icao_location=fields.get("ICAO Location", ""),
                text=fields.get("Text", ""),
                route=route,
                departure=meta["departure"],
                destination=meta["destination"],
            ))
        except Exception as e:
            failed += 1
            logger.warning(f"Failed to parse block in {route}: {e}")

    logger.info(f"[{route}] Parsed {len(notams)} NOTAMs ({failed} failed) from {filepath.name}")
    return notams


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(notams: list[ParsedNotam]) -> list[ParsedNotam]:
    """
    Removes duplicate NOTAMs by ID, keeping the first occurrence.
    Duplicates appear because the same en-route NOTAM can be fetched
    for multiple routes (e.g. a Kansas City center NOTAM appears in
    both OKC->JFK and OKC->ORD).
    """
    seen = set()
    unique = []
    duplicates = 0

    for notam in notams:
        if notam.id not in seen:
            seen.add(notam.id)
            unique.append(notam)
        else:
            duplicates += 1

    logger.info(f"Deduplication: removed {duplicates} duplicates, {len(unique)} unique NOTAMs remain")
    return unique


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_merged(notams: list[ParsedNotam], output_path: Path):
    """
    Writes the merged, deduplicated NOTAMs to a single file.
    Injects Route, Departure, and Destination fields so the label
    generator has full flight context.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for notam in notams:
            f.write(f"ID: {notam.id}\n")
            f.write(f"Number: {notam.number}\n")
            f.write(f"Type: {notam.type}\n")
            f.write(f"Issued: {notam.issued}\n")
            f.write(f"Selection Code: {notam.selection_code}\n")
            f.write(f"Location: {notam.location}\n")
            f.write(f"Effective Start: {notam.effective_start}\n")
            f.write(f"Effective End: {notam.effective_end}\n")
            f.write(f"Classification: {notam.classification}\n")
            f.write(f"Account ID: {notam.account_id}\n")
            f.write(f"Last Updated: {notam.last_updated}\n")
            f.write(f"ICAO Location: {notam.icao_location}\n")
            f.write(f"Route: {notam.route}\n")
            f.write(f"Departure: {notam.departure}\n")
            f.write(f"Destination: {notam.destination}\n")
            f.write(f"Text: {notam.text}\n")
            f.write(NOTAM_SEPARATOR + "\n")

    logger.info(f"Wrote {len(notams)} merged NOTAMs to {output_path}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(notams: list[ParsedNotam]):
    """Prints a breakdown of the merged dataset — useful for Checkpoint 0.3."""
    print("\n" + "=" * 60)
    print("  MERGED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total NOTAMs: {len(notams)}")

    # By route
    print("\n  By route:")
    route_counts: dict[str, int] = {}
    for n in notams:
        route_counts[n.route] = route_counts.get(n.route, 0) + 1
    for route, count in sorted(route_counts.items()):
        print(f"    {route}: {count}")

    # By classification
    print("\n  By classification:")
    class_counts: dict[str, int] = {}
    for n in notams:
        c = n.classification.replace("Classification.", "")
        class_counts[c] = class_counts.get(c, 0) + 1
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls}: {count}")

    # By selection code (top 10)
    print("\n  Top 10 selection codes:")
    code_counts: dict[str, int] = {}
    for n in notams:
        code = n.selection_code if n.selection_code and n.selection_code != "None" else "(none)"
        code_counts[code] = code_counts.get(code, 0) + 1
    top_codes = sorted(code_counts.items(), key=lambda x: -x[1])[:10]
    for code, count in top_codes:
        print(f"    {code}: {count}")

    # Empty text warning
    empty_text = sum(1 for n in notams if not n.text.strip())
    if empty_text:
        print(f"\n  ⚠️  WARNING: {empty_text} NOTAMs have empty text fields")
    else:
        print(f"\n  ✓  All NOTAMs have non-empty text fields")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_all() -> list[ParsedNotam]:
    """
    Loads, merges, and deduplicates all three route files.
    Returns the final list of unique ParsedNotam objects.
    """
    all_notams = []

    for route, filepath in RAW_FILES.items():
        notams = parse_notam_file(filepath, route)
        all_notams.extend(notams)

    unique_notams = deduplicate(all_notams)
    return unique_notams


if __name__ == "__main__":
    notams = load_all()
    print_summary(notams)
    write_merged(notams, MERGED_FILE)
    print(f"Done. Merged file written to: {MERGED_FILE}")