# =============================================================================
# ml_scorer.py — ML-powered NOTAM scorer for integration with driver.py
#
# This module is the bridge between the trained ML models and the existing
# NOTAM pipeline. It runs alongside (not replacing) the existing rule-based
# system. Differences between the two are flagged for review.
#
# Usage from driver.py:
#   from ml.ml_scorer import MLScorer
#   scorer = MLScorer()
#   results = scorer.score(notams)
# =============================================================================

import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib

from ml.config import (
    BASELINE_MODEL_PATH,
    EMBEDDING_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    EMBEDDINGS_CACHE_PATH,
    ANOMALY_MODEL_PATH,
    EMBEDDING_MODEL_NAME,
    CRITICALITY_LEVELS,
    CRITICALITY_ORDER,
)

logger = logging.getLogger("MLScorer")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MLResult:
    """ML scoring result for a single NOTAM."""
    notam_id:           str
    text:               str

    # Layer 1 — TF-IDF + Logistic Regression
    lr_criticality:     str
    lr_confidence:      float   # max class probability

    # Layer 2 — Embeddings + Random Forest
    rf_criticality:     str
    rf_confidence:      float

    # Layer 3 — Anomaly Detection
    anomaly_score:      float
    is_anomaly:         bool

    # Consensus: do both classifiers agree?
    models_agree:       bool

    # Final recommended criticality (RF wins on disagreement — higher HIGH F1)
    final_criticality:  str


# ---------------------------------------------------------------------------
# ML Scorer
# ---------------------------------------------------------------------------

class MLScorer:
    """
    Loads all three trained models and scores a batch of NOTAM texts.
    Models are lazy-loaded on first use.
    """

    ANOMALY_THRESHOLD_PERCENTILE = 95   # Top 5% flagged as anomalous

    def __init__(self):
        self._tfidf     = None
        self._lr        = None
        self._rf        = None
        self._iso       = None
        self._embedder  = None
        self._models_loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        """Lazy-loads all models on first call to score()."""
        if self._models_loaded:
            return

        logger.info("Loading ML models...")

        missing = []
        for path in [BASELINE_MODEL_PATH, EMBEDDING_MODEL_PATH,
                     TFIDF_VECTORIZER_PATH, ANOMALY_MODEL_PATH]:
            if not path.exists():
                missing.append(path.name)

        if missing:
            raise FileNotFoundError(
                f"Missing trained models: {missing}\n"
                f"Run notebooks/03_model_training.ipynb and "
                f"notebooks/04_anomaly_detection.ipynb first."
            )

        self._tfidf = joblib.load(TFIDF_VECTORIZER_PATH)
        self._lr    = joblib.load(BASELINE_MODEL_PATH)
        self._rf    = joblib.load(EMBEDDING_MODEL_PATH)
        self._iso   = joblib.load(ANOMALY_MODEL_PATH)

        # Load sentence transformer
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self._models_loaded = True
        logger.info("All models loaded.")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, notams: list) -> list[MLResult]:
        """
        Scores a list of NOTAM objects from the existing pipeline.
        Expects objects with .id and .text attributes.

        Args:
            notams: List of NOTAM objects (ParsedNotam from data_loader)

        Returns:
            List of MLResult objects in same order as input
        """
        self._load_models()

        if not notams:
            return []

        # Filter out cancelled NOTAMs — they are no longer operationally relevant
        notams = [n for n in notams if 'CANCELED' not in str(n.text).upper()]
        texts = [str(n.text) for n in notams]
        ids   = [str(n.id)   for n in notams]

        logger.info(f"Scoring {len(texts)} NOTAMs with ML pipeline...")

        # --- Layer 1: TF-IDF + LR ---
        tfidf_matrix    = self._tfidf.transform(texts)
        lr_preds        = self._lr.predict(tfidf_matrix)
        lr_proba        = self._lr.predict_proba(tfidf_matrix)
        lr_confidences  = lr_proba.max(axis=1)

        # --- Layer 2: Embeddings + RF ---
        embeddings      = self._embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=len(texts) > 100,
            device=self._get_device(),
        )
        rf_preds        = self._rf.predict(embeddings)
        rf_proba        = self._rf.predict_proba(embeddings)
        rf_confidences  = rf_proba.max(axis=1)

        # --- Layer 3: Anomaly Detection ---
        raw_scores      = -self._iso.decision_function(embeddings)
        threshold       = np.percentile(raw_scores, self.ANOMALY_THRESHOLD_PERCENTILE)
        is_anomaly      = raw_scores >= threshold

        # --- Assemble results ---
        results = []
        for i in range(len(notams)):
            lr_crit = str(lr_preds[i])
            rf_crit = str(rf_preds[i])
            agree   = lr_crit == rf_crit

            # RF wins on disagreement — it has higher HIGH F1 (81% vs 79%)
            final = rf_crit

            results.append(MLResult(
                notam_id=ids[i],
                text=texts[i],
                lr_criticality=lr_crit,
                lr_confidence=round(float(lr_confidences[i]), 3),
                rf_criticality=rf_crit,
                rf_confidence=round(float(rf_confidences[i]), 3),
                anomaly_score=round(float(raw_scores[i]), 4),
                is_anomaly=bool(is_anomaly[i]),
                models_agree=agree,
                final_criticality=final,
            ))

        n_disagree = sum(1 for r in results if not r.models_agree)
        n_anomaly  = sum(1 for r in results if r.is_anomaly)
        logger.info(
            f"Scored {len(results)} NOTAMs | "
            f"{n_disagree} model disagreements | "
            f"{n_anomaly} anomalies flagged"
        )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_device(self) -> str:
        """Returns best available device for sentence-transformers."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return 'mps'    # M2 Pro GPU
            if torch.cuda.is_available():
                return 'cuda'
        except ImportError:
            pass
        return 'cpu'

    def save_brief(
        self,
        results: list[MLResult],
        filepath: str,
        departure: str = "",
        destination: str = "",
    ) -> str:
        """
        Saves a pilot-facing NOTAM brief to a text file, sorted by ML criticality.

        Order: HIGH → MEDIUM → LOW → INFO
        Anomalies are flagged inline within their criticality section.

        Args:
            results:     List of MLResult objects from score()
            filepath:    Output file path (e.g. 'data/raw/OKC_DEN_ml.txt')
            departure:   Departure airport code for header
            destination: Destination airport code for header

        Returns:
            The filepath written to.
        """
        from pathlib import Path
        import datetime

        W     = 72
        lines = []

        total    = len(results)
        by_crit  = {level: [] for level in CRITICALITY_LEVELS}
        for r in results:
            by_crit[r.final_criticality].append(r)

        anomalies = [r for r in results if r.is_anomaly]
        route     = f"{departure} -> {destination}" if departure and destination else "NOTAM BRIEF"
        now       = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%MZ")

        SECTION_LABELS = {
            'HIGH':   '[ HIGH ]   Action required',
            'MEDIUM': '[ MEDIUM ] Situational awareness',
            'LOW':    '[ LOW ]    Informational',
            'INFO':   '[ INFO ]   Advisory only',
        }

        # ------------------------------------------------------------------
        # File header
        # ------------------------------------------------------------------
        lines += [
            "=" * W,
            f"  NOTAM BRIEF  |  {route}  |  {now} UTC",
            f"  {total} NOTAMs scored by ML pipeline",
            "-" * W,
        ]
        for level in CRITICALITY_LEVELS:
            count = len(by_crit[level])
            pct   = count / total * 100 if total else 0
            lines.append(f"  {level:<8} {count:>4}  ({pct:4.1f}%)")
        lines += [
            f"  Anomalies    {len(anomalies):>4}  ({len(anomalies)/total*100:4.1f}%)",
            "=" * W,
            "",
        ]

        # ------------------------------------------------------------------
        # One section per criticality level, sorted HIGH → INFO
        # ------------------------------------------------------------------
        for level in CRITICALITY_LEVELS:
            section_notams = by_crit[level]
            lines += [
                "=" * W,
                f"  {SECTION_LABELS[level]}  —  {len(section_notams)} NOTAMs",
                "=" * W,
            ]

            if not section_notams:
                lines += ["  None.", ""]
                continue

            for r in section_notams:
                anomaly_flag = "  [ANOMALY]" if r.is_anomaly else ""
                conf         = f"conf={r.rf_confidence:.0%}"
                lines += [
                    "-" * W,
                    f"  {r.notam_id}  |  {level}{anomaly_flag}  |  {conf}",
                    "-" * W,
                    r.text.strip(),
                    "",
                ]

        # ------------------------------------------------------------------
        # Write to file
        # ------------------------------------------------------------------
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"ML brief saved to {filepath}")
        return filepath