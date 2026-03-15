# ML Architecture — NOTAM Criticality Classifier

> This section documents the machine learning pipeline added to the NOTAM
> prioritization system. The ML component augments the existing rule-based
> sorter with three layers of learned intelligence.

---

## Motivation

The original system relies on FAA Q-codes (selection codes) to categorize
NOTAMs. An analysis of our dataset revealed that **59.2% of NOTAMs have no
Q-code**, meaning rule-based systems fail silently on the majority of real
flight data. Text-based ML is not an enhancement — it is a necessity.

---

## Architecture Overview

```
Raw NOTAM Text
      │
      ├──► Layer 1: TF-IDF + Logistic Regression  ──► Criticality Label (baseline)
      │
      ├──► Layer 2: Sentence Embeddings (all-MiniLM-L6-v2)
      │              └──► Random Forest Classifier  ──► Criticality Label (primary)
      │              └──► Isolation Forest           ──► Anomaly Flag
      │
      └──► Final Output: Criticality + Anomaly Score + Model Agreement Flag
```

### Layer 1 — TF-IDF + Logistic Regression (Baseline)
Converts NOTAM text to a bag-of-words feature matrix using Term Frequency–
Inverse Document Frequency weighting with unigrams and bigrams. A Logistic
Regression classifier with `class_weight='balanced'` learns which terms predict
each criticality level. Fast, interpretable, and sets the performance benchmark.

### Layer 2 — Sentence Embeddings + Random Forest (Primary)
Uses the `all-MiniLM-L6-v2` sentence transformer to encode each NOTAM as a
384-dimensional semantic vector. Two NOTAMs with equivalent meaning but different
wording (e.g. "RWY CLSD" vs "Runway closed") receive similar embeddings. A
Random Forest classifier operates on these vectors. Achieves higher HIGH
criticality F1 than the baseline — the metric that matters most for safety.

### Layer 3 — Isolation Forest (Anomaly Detection)
Operates on the same 384-dimensional embeddings from Layer 2. NOTAMs that are
difficult to isolate from their neighbours are normal; those easily separated
are flagged as anomalous. Identifies unusual events that don't fit established
patterns — rare airspace restrictions, atypical operational prohibitions, and
complex multi-event NOTAMs.

### Label Generation — Weak Supervision
Training labels were generated using Claude (Anthropic) acting as an experienced
airline dispatcher. Each NOTAM was scored for operational criticality relative
to its specific flight route, producing HIGH / MEDIUM / LOW / INFO labels with
a confidence score and reason. This technique — using an LLM to bootstrap
labels for a downstream ML model — is known as **weak supervision**.

---

## Dataset

| Property | Value |
|----------|-------|
| Total NOTAMs | 3,945 |
| Flight routes | OKC→JFK, OKC→ORD, OKC→DEN |
| Unique airports covered | 80+ |
| NOTAMs without Q-code | 59.2% |
| Label source | LLM weak supervision (Claude) |

**Class distribution:**

| Class | Count | % |
|-------|-------|---|
| HIGH | 319 | 8.1% |
| MEDIUM | 833 | 21.1% |
| LOW | 2,143 | 54.3% |
| INFO | 650 | 16.5% |

Class imbalance ratio: **6.7x** (LOW vs HIGH).
All models use `class_weight='balanced'` to compensate.

---

## Results

| Model | Accuracy | Macro F1 | Weighted F1 | HIGH F1 |
|-------|----------|----------|-------------|---------|
| TF-IDF + Logistic Regression | — | 75.8% | — | 79.4% |
| Sentence Embeddings + Random Forest | — | 74.4% | — | **81.0%** |

**Key finding:** The TF-IDF baseline outperforms embeddings on overall Macro F1,
which makes sense for NOTAM text — it is highly structured aviation jargon where
keyword presence is strongly predictive. However, sentence embeddings achieve
higher F1 specifically on HIGH criticality NOTAMs (+1.6%), which is the metric
that matters most for aviation safety. The Random Forest is used as the primary
classifier in the final pipeline for this reason.

---

## Anomaly Detection Results

- **198 NOTAMs flagged** as anomalous (5.0% of dataset)
- **85% of top anomalies** were content-driven (not rare-airport artifacts)
- Notable catches: unusual operational restrictions at military facilities,
  complex instrument approach procedure changes, atypical airspace events

---

## Known Limitations

**1. Rare Airport Conflation**
The Isolation Forest operates on embeddings that encode airport context
alongside NOTAM content. NOTAMs from airports with fewer than ~5 appearances
in the dataset sit in sparse embedding regions and may be flagged as anomalous
due to airport rarity rather than content unusualness.
*Future fix: strip airport identifiers from text before embedding, or filter
airports below a frequency threshold before fitting.*

**2. Weak Supervision Label Noise**
LLM-generated labels have natural ambiguity at class boundaries (e.g. ILS
outage at an alternate airport — MEDIUM or LOW?). This introduces noise that
affects classifier performance and anomaly scoring.
*Future fix: human-in-the-loop review of ~200 borderline cases to create a
gold-standard validation set.*

**3. Fixed Contamination Hyperparameter**
The 5% contamination setting is an assumption. Real anomaly rates in FAA NOTAM
data are unknown. The model flags exactly 5% regardless of the true rate.
*Future fix: knee-point detection on the score distribution for data-driven
threshold selection.*

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_label_generation.ipynb` | LLM labeling pipeline and quality check |
| `notebooks/02_eda.ipynb` | Exploratory data analysis |
| `notebooks/03_model_training.ipynb` | Layer 1 and Layer 2 training and evaluation |
| `notebooks/04_anomaly_detection.ipynb` | Layer 3 anomaly detection |

---

## Reproducing Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Merge route files
python3 -m ml.data_loader

# 3. Generate labels (requires ANTHROPIC_API_KEY in .env)
python3 -m ml.label_generator

# 4. Run notebooks in order
jupyter notebook notebooks/02_eda.ipynb
jupyter notebook notebooks/03_model_training.ipynb
jupyter notebook notebooks/04_anomaly_detection.ipynb
```
