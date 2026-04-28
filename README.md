# Engagement Index AI Model

Machine learning pipeline for predicting customer engagement scores using behavioral and transactional features across 30, 60, and 90-day windows.

---

## Overview

The Engagement Index model classifies members into four engagement tiers (0–3) based on historical behavioral signals. It helps identify disengaged members early and enables proactive outreach strategies.

| Tier | Label              | Description                                      |
|------|--------------------|--------------------------------------------------|
| 3    | Highly Engaged     | Active across 3+ pillars, care-active, no emergency |
| 2    | Moderately Engaged | Active across 2–3 pillars with some care/device activity |
| 1    | Low Engagement     | Limited activity across 1–2 pillars              |
| 0    | Disengaged         | No activity, 3+ consecutive inactive months      |

---

## Project Structure

```
engagement_index_AI_model/
│
├── engagement_index_feature_engineering.py   # Feature engineering pipeline (raw → engineered features)
├── engagement_index_feature_list.txt         # Full feature catalog with descriptions
│
├── engagement_index_training_30d.py          # Model training script — 30-day horizon
├── engagement_index_training_60d.py          # Model training script — 60-day horizon
├── engagement_index_training_90d.py          # Model training script — 90-day horizon
│
├── engagement_index_scoring_30d.py           # Scoring script — 30-day horizon
├── engagement_index_scoring_60d.py           # Scoring script — 60-day horizon
├── engagement_index_scoring_90d.py           # Scoring script — 90-day horizon
│
├── 202411_to_202603_dtc_engagement_index.parquet  # Base feature data (Nov 2024 – Mar 2026)
│
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
└── getting_started.md                        # Step-by-step usage guide
```

---

## Pipeline Architecture

```
Raw Data (parquet/csv)
        │
        ▼
engagement_index_feature_engineering.py
        │
        ▼
Engineered Feature File (with target columns)
        │
        ├──▶ engagement_index_training_30d.py  ──▶ Model artifact (.pkl) + Validation reports
        ├──▶ engagement_index_training_60d.py  ──▶ Model artifact (.pkl) + Validation reports
        └──▶ engagement_index_training_90d.py  ──▶ Model artifact (.pkl) + Validation reports
                                                              │
                                                              ▼
                                               engagement_index_scoring_30d/60d/90d.py
                                                              │
                                                              ▼
                                               Scored output file (Engagement_Score, Engagement_Tier, Engaged_Flag)
```

---

## Features

The model uses **46 engineered features** organized across 7 behavioral pillars:

| Pillar | Description                          | Example Features                             |
|--------|--------------------------------------|----------------------------------------------|
| 1      | Device Engagement                    | `buttons_count_6m`, `steps_trend`            |
| 2      | Care / System Interaction            | `monitoring_outreach_6m`, `case_count_12m`   |
| 3      | Emergency / Risk Events              | `had_emergency_event_6m`, `emergency_count_6m` |
| 4      | Communication / Email Engagement     | `email_open_rate_avg_6m`, `email_active_flag_12m` |
| 5      | Portal Engagement                    | `has_active_portal_flag`, `portal_recency_score` |
| 6      | Recency & Consistency (Cross-Pillar) | `consecutive_inactive_months`, `active_months_ratio_6m` |
| 7      | Member Profile (Static Context)      | `product_engagement_profile`, `age`          |

See [`engagement_index_feature_list.txt`](engagement_index_feature_list.txt) for the full feature catalog.

---

## Model

- **Algorithm:** XGBoost (default), with support for RandomForest, GradientBoosting, LogisticRegression
- **Target:** Multi-class engagement tier (0–3), binarized for threshold selection (engaged = tier > 0)
- **Threshold selection:** Validation-only, adaptive per predict month — caps FNR ≤ 0.30, then minimizes FPR
- **Split:** Stratified 70/30 train/validation split

### Output Columns (Scored File)

| Column            | Description                                         |
|-------------------|-----------------------------------------------------|
| `account_number`  | Unique member identifier                            |
| `account_id`      | Salesforce account ID                               |
| `anchor_month`    | Feature window anchor month                         |
| `Engagement_Score`| Predicted probability of engagement (0.0–1.0)       |
| `Engagement_Tier` | Human-readable tier: `Low`, `Moderate`, `High`      |
| `Engaged_Flag`    | Binary flag (1 = engaged, 0 = not engaged)          |

---

## Valid Anchor Months & Target Availability

| Anchor  | t30d Label | t60d Label | t90d Label |
|---------|------------|------------|------------|
| 2025-11 | 2025-12 ✓  | 2026-01 ✓  | 2026-02 ✓  |
| 2025-12 | 2026-01 ✓  | 2026-02 ✓  | 2026-03 ✓  |
| 2026-01 | 2026-02 ✓  | 2026-03 ✓  | 2026-04 ✗  |
| 2026-02 | 2026-03 ✓  | 2026-04 ✗  | 2026-05 ✗  |
| 2026-03 | 2026-04 ✗  | 2026-05 ✗  | 2026-06 ✗  |

---

## Quick Start

See [`getting_started.md`](getting_started.md) for full step-by-step instructions.

```bash
# Step 1: Feature engineering
python engagement_index_feature_engineering.py \
    --input_file 202411_to_202603_dtc_engagement_index.parquet \
    --anchor_month 01_2026

# Step 2: Train 30d model
python engagement_index_training_30d.py \
    --train_file engagement_index_feature_with_target_30d.parquet \
    --anchor_month 01_2026

# Step 3: Score members
python engagement_index_scoring_30d.py \
    --model_file engagement_index_model_30d_02_2026.pkl \
    --score_file engagement_index_feature.parquet \
    --anchor_month 01_2026
```

---

## Requirements

Python 3.10+ is required. Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## License

Internal use only. Not for public distribution.
