# =============================================================================
# label_generator.py — Async LLM labeler with 10 concurrent API calls
#
# Uses asyncio + aiohttp to run 10 API calls in parallel instead of
# sequentially. On an M2 Pro this reduces ~100 min runtime to ~10 min.
#
# Usage:
#   python3 -m ml.label_generator
#
# Output:
#   data/labeled/notams_labeled.csv   — labeled dataset ready for ML training
#   data/labeled/nlp_cache.json       — cache so re-runs skip already-labeled NOTAMs
#
# Safe to interrupt and resume — cache is saved after every BATCH_SIZE completions.
# =============================================================================

import asyncio
import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from ml.config import (
    ANTHROPIC_MODEL,
    MAX_TOKENS,
    API_DELAY_SECONDS,
    MAX_RETRIES,
    RETRY_BACKOFF,
    CRITICALITY_LEVELS,
    LABELED_CSV,
    NLP_CACHE_FILE,
    MERGED_FILE,
    NOTAM_SEPARATOR,
    LLM_TEMPERATURE
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger("LabelGenerator")

# ---------------------------------------------------------------------------
# Concurrency settings
# ---------------------------------------------------------------------------

MAX_CONCURRENT = 2    # Parallel API calls
BATCH_SIZE     = 20    # Save cache every N completions


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParsedNotam:
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
    route: str
    departure: str
    destination: str
    text: str


@dataclass
class NotamLabel:
    notam_id: str
    criticality: str    # HIGH / MEDIUM / LOW / INFO
    score: int          # 0-100
    reason: str


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_merged_file(filepath: Path) -> list[ParsedNotam]:
    """Parses merged_notams.txt produced by data_loader.py."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Merged file not found: {filepath}\n"
            f"Run data_loader.py first."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split(NOTAM_SEPARATOR) if b.strip()]
    notams = []
    failed = 0

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
                route=fields.get("Route", ""),
                departure=fields.get("Departure", ""),
                destination=fields.get("Destination", ""),
                text=fields.get("Text", ""),
            ))
        except Exception as e:
            failed += 1
            logger.warning(f"Failed to parse block: {e}")

    logger.info(f"Parsed {len(notams)} NOTAMs ({failed} failed)")
    return notams


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an experienced airline dispatcher with 20 years of experience
reading and evaluating FAA NOTAMs for commercial flight operations. You have deep knowledge
of ICAO NOTAM format, FAA Order 7930.2, and standard flight operations procedures.

Your job is to evaluate each NOTAM for operational criticality to the specific flight provided.
You must return a JSON object with exactly these fields:
- "criticality": one of "HIGH", "MEDIUM", "LOW", or "INFO"
- "score": an integer from 0 to 100 representing criticality (100 = most critical)
- "reason": a concise one-sentence explanation (max 20 words)

Criticality guidelines:
- HIGH (75-100): Directly impacts flight safety or operations. Runway/airport closures,
  ILS/GPS outages at departure or destination, TFRs, national defense airspace, MEA changes
  on the route, VOR outages affecting navigation.
- MEDIUM (40-74): Notable but not immediately flight-critical. Taxiway closures,
  non-precision approach outages, obstacle warnings, lighting outages on active runways,
  airspace restrictions away from the flight path.
- LOW (15-39): Minor operational impact. Peripheral equipment outages,
  non-critical lighting, minor construction away from active areas, ramp closures.
- INFO (0-14): Informational only. Administrative NOTAMs, services availability,
  non-safety-related facility info, documentation updates.

Return ONLY valid JSON. No preamble, no markdown, no explanation outside the JSON object."""


def build_user_prompt(notam: ParsedNotam) -> str:
    classification = notam.classification.replace("Classification.", "")
    notam_type = notam.type.replace("NotamType.", "")
    return f"""Flight: {notam.departure} -> {notam.destination}

NOTAM:
ID: {notam.id}
Type: {notam_type}
Location: {notam.location} (ICAO: {notam.icao_location})
Classification: {classification}
Selection Code: {notam.selection_code}
Effective: {notam.effective_start} to {notam.effective_end}
Text: {notam.text}

Return JSON only."""


# ---------------------------------------------------------------------------
# Async label generator
# ---------------------------------------------------------------------------

class LabelGenerator:

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Check your .env file.")
        self.cache: dict[str, dict] = self._load_cache()
        self._cache_dirty = False
        self._completed = 0
        self._semaphore = None
        self._lock = None

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict:
        if NLP_CACHE_FILE.exists():
            with open(NLP_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} cached labels from {NLP_CACHE_FILE.name}")
            return cache
        return {}

    def _save_cache(self):
        NLP_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(NLP_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)
        self._cache_dirty = False

    # ------------------------------------------------------------------
    # Single async API call with retry
    # ------------------------------------------------------------------

    async def _label_one(
        self,
        session: aiohttp.ClientSession,
        notam: ParsedNotam,
        total: int,
    ) -> NotamLabel:
        """Labels a single NOTAM. Semaphore caps concurrency at MAX_CONCURRENT."""
        async with self._semaphore:
            last_error = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    async with session.post(
                        self.API_URL,
                        headers={
                            "Content-Type": "application/json",
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01",
                        },
                        json={
                            "model": ANTHROPIC_MODEL,
                            "max_tokens": MAX_TOKENS,
                            "temperature": LLM_TEMPERATURE,
                            "system": SYSTEM_PROMPT,
                            "messages": [
                                {"role": "user", "content": build_user_prompt(notam)}
                            ],
                        },
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:

                        if resp.status in (429, 500, 529):
                            wait = API_DELAY_SECONDS * (RETRY_BACKOFF ** attempt)
                            logger.warning(
                                f"HTTP {resp.status} for {notam.id} "
                                f"(attempt {attempt}/{MAX_RETRIES}), "
                                f"retrying in {wait:.1f}s..."
                            )
                            await asyncio.sleep(wait)
                            continue

                        resp.raise_for_status()
                        data = await resp.json()

                    raw_text = data["content"][0]["text"].strip()
                    raw_text = re.sub(r"^```json\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)

                    parsed = json.loads(raw_text)

                    criticality = parsed.get("criticality", "INFO").upper()
                    if criticality not in CRITICALITY_LEVELS:
                        criticality = "INFO"

                    label = NotamLabel(
                        notam_id=notam.id,
                        criticality=criticality,
                        score=max(0, min(100, int(parsed.get("score", 0)))),
                        reason=parsed.get("reason", "")[:200],
                    )

                    # Thread-safe cache update
                    async with self._lock:
                        self.cache[notam.id] = {
                            "criticality": label.criticality,
                            "score": label.score,
                            "reason": label.reason,
                        }
                        self._completed += 1
                        self._cache_dirty = True

                        if self._completed % BATCH_SIZE == 0:
                            self._save_cache()
                            logger.info(
                                f"Progress: {self._completed}/{total} "
                                f"({self._completed/total*100:.1f}%)"
                            )

                    return label

                except (json.JSONDecodeError, KeyError) as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(API_DELAY_SECONDS)

                except Exception as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(API_DELAY_SECONDS * (RETRY_BACKOFF ** attempt))

            logger.error(f"All retries exhausted for {notam.id}: {last_error}")
            return NotamLabel(
                notam_id=notam.id,
                criticality="INFO",
                score=0,
                reason=f"LABELING_FAILED: {str(last_error)[:100]}",
            )

    # ------------------------------------------------------------------
    # Async orchestrator
    # ------------------------------------------------------------------

    async def _run_async(self, notams: list[ParsedNotam]) -> list[NotamLabel]:
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self._lock = asyncio.Lock()

        uncached = [n for n in notams if n.id not in self.cache]
        cached_count = len(notams) - len(uncached)

        logger.info(
            f"Labeling {len(notams)} NOTAMs: "
            f"{cached_count} from cache, {len(uncached)} via API"
        )
        if uncached:
            est = (len(uncached) / MAX_CONCURRENT) * 1.5 / 60
            logger.info(f"Estimated time with {MAX_CONCURRENT} concurrent calls: ~{est:.0f} min")

        # Resolve cached results immediately
        results_map: dict[str, NotamLabel] = {}
        for notam in notams:
            if notam.id in self.cache:
                c = self.cache[notam.id]
                results_map[notam.id] = NotamLabel(
                    notam_id=notam.id,
                    criticality=c["criticality"],
                    score=c["score"],
                    reason=c["reason"],
                )

        # Fire concurrent tasks for uncached NOTAMs
        if uncached:
            start = time.time()
            connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [
                    self._label_one(session, notam, len(notams))
                    for notam in uncached
                ]
                api_results = await asyncio.gather(*tasks)

            elapsed = time.time() - start
            logger.info(
                f"Done in {elapsed:.0f}s "
                f"({len(uncached) / elapsed:.1f} NOTAMs/sec)"
            )

            for label in api_results:
                results_map[label.notam_id] = label

            if self._cache_dirty:
                self._save_cache()

        return [results_map[n.id] for n in notams if n.id in results_map]

    def label_all(self, notams: list[ParsedNotam]) -> list[NotamLabel]:
        """Synchronous entry point — wraps the async pipeline."""
        return asyncio.run(self._run_async(notams))

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def save_csv(self, notams: list[ParsedNotam], labels: list[NotamLabel]):
        """Saves labeled dataset as CSV — the single input to all ML layers."""
        LABELED_CSV.parent.mkdir(parents=True, exist_ok=True)
        label_map = {l.notam_id: l for l in labels}

        fieldnames = [
            "id", "route", "departure", "destination",
            "icao_location", "classification", "selection_code",
            "text", "criticality", "score", "reason",
        ]

        with open(LABELED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for notam in notams:
                label = label_map.get(notam.id)
                if not label:
                    continue
                writer.writerow({
                    "id":             notam.id,
                    "route":          notam.route,
                    "departure":      notam.departure,
                    "destination":    notam.destination,
                    "icao_location":  notam.icao_location,
                    "classification": notam.classification.replace("Classification.", ""),
                    "selection_code": notam.selection_code,
                    "text":           notam.text,
                    "criticality":    label.criticality,
                    "score":          label.score,
                    "reason":         label.reason,
                })

        logger.info(f"Saved labeled CSV to {LABELED_CSV}")

    # ------------------------------------------------------------------
    # Summary (Checkpoint 1.3)
    # ------------------------------------------------------------------

    def print_summary(self, labels: list[NotamLabel]):
        print("\n" + "=" * 60)
        print("  LABEL DISTRIBUTION SUMMARY")
        print("=" * 60)

        counts: dict[str, int] = {level: 0 for level in CRITICALITY_LEVELS}
        failed = 0

        for label in labels:
            if label.reason.startswith("LABELING_FAILED"):
                failed += 1
            counts[label.criticality] = counts.get(label.criticality, 0) + 1

        total = len(labels)
        for level in CRITICALITY_LEVELS:
            count = counts.get(level, 0)
            pct = (count / total * 100) if total else 0
            bar = "█" * int(pct / 2)
            print(f"  {level:<8} {count:>5} ({pct:5.1f}%)  {bar}")

        if failed:
            print(f"\n  ⚠️  {failed} failed — re-run to retry automatically")
        else:
            print(f"\n  ✓  All {total} NOTAMs labeled successfully")

        for level in CRITICALITY_LEVELS:
            if counts.get(level, 0) < 50:
                print(f"\n  ⚠️  WARNING: {level} only has {counts[level]} examples.")

        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    notams = parse_merged_file(MERGED_FILE)

    generator = LabelGenerator()
    labels = generator.label_all(notams)

    generator.save_csv(notams, labels)
    generator.print_summary(labels)

    print(f"Done. Labeled CSV -> {LABELED_CSV}")