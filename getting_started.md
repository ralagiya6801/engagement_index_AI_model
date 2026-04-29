# Getting Started — Engagement Index AI Model

This guide walks you through the full pipeline: feature engineering → training → scoring.

---

## Prerequisites

- Python 3.10 or higher
- pip

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Input Data Requirements

| File | Description |
|------|-------------|
| `202411_to_202603_dtc_engagement_index.parquet` | Long-format base feature file (one row per account per observation month) |
| `portal_engagement_snapshot.csv` | Portal engagement snapshot (one row per account, optional) |

### Required columns in the base feature file

| Column | Description |
|--------|-------------|
| `account_number` | Unique customer identifier |
| `account_id` | Salesforce account ID |
| `age` | Member age |
| `obs_month` | Observation month (YYYY-MM format) |
| `product_name` | Medical Guardian device product name |

---

## Step 1: Feature Engineering

Run the feature engineering script to generate the engineered feature file with engagement targets.

```bash
python engagement_index_feature_engineering.py \
    --input_file 202411_to_202603_dtc_engagement_index.parquet \
    --anchor_month 01_2026
```

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_file` | *(required)* | Path to long-format base feature parquet/csv file |
| `--portal_file` | `None` | Path to portal engagement snapshot CSV |
| `--anchor_month` | `01_2026` | Anchor month in `MM_YYYY` format |
| `--output_dir` | Auto-generated | Directory to save engineered feature file |

**Output files:**

- `engagement_index_feature_with_target_30d.parquet` — Features + 30d target

---

## Step 2: Train the Model

Train the engagement model for your desired prediction horizon (30d).

### 30-day horizon

```bash
python engagement_index_training_30d.py \
    --train_file engagement_index_feature_with_target_30d.parquet \
    --anchor_month 01_2026
```
```

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--algorithm` | `XGBoost` | Model algorithm: `XGBoost`, `RandomForest`, `GradientBoosting`, `LogisticRegression` |
| `--val_size` | `0.3` | Validation split fraction |
| `--max_validation_fpr` | `0.30` | Maximum allowed False Positive Rate on validation |
| `--max_validation_fnr` | `0.30` | Maximum allowed False Negative Rate on validation |
| `--min_threshold` | `0.03` | Minimum decision threshold floor |
| `--random_state` | `42` | Random seed for reproducibility |
| `--output_dir` | Auto-generated | Directory to save validation outputs |
| `--model_file` | Auto-generated | Output path for saved model artifact (`.pkl`) |
| `--retrain_model` | `None` | Path to an existing `.pkl` to load and retrain |

**Output files (saved to `engagement_index_30d_training_output_<predict_month>/`):**

| File | Description |
|------|-------------|
| `engagement_index_model_30d_<predict_month>.pkl` | Trained model artifact |
| `ROC_AUC_Val.png` | ROC-AUC curve on validation set |
| `CM_Val.png` | Confusion matrix heatmap |
| `CM_Val.txt` | Confusion matrix with FPR/FNR metrics |
| `Performance_Val.txt` | Full classification report |
| `Training_Metadata.csv` | Training run parameters and metrics |

---

## Step 3: Score Members

Use the trained model to score members for the next period.

### 30-day scoring

```bash
python engagement_index_scoring_30d.py \
    --model_file engagement_index_model_30d_02_2026.pkl \
    --score_file engagement_index_feature.parquet \
    --anchor_month 01_2026
```

```

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--score_threshold` | Model adaptive threshold | Override the decision threshold manually |
| `--output_file` | Auto-generated | Output file path for scored members |
| `--output_dir` | Auto-generated | Directory to save scored file and metadata |

**Output files (saved to `engagement_index_30d_scoring_output_<predict_month>/`):**

| File | Description |
|------|-------------|
| `engagement_index_scored_30d_<predict_month>.parquet` | Scored members file |
| `Scoring_Report_Sample.csv` | First 100 scored members preview |
| `Scoring_Metadata.csv` | Scoring run summary (threshold, % engaged, score stats) |

---

## Output Score Columns

| Column | Description |
|--------|-------------|
| `account_number` | Unique member identifier |
| `account_id` | Salesforce account ID |
| `anchor_month` | Feature window anchor month |
| `Engagement_Score` | Predicted probability of engagement (0.0 – 1.0) |
| `Engagement_Tier` | Human-readable: `Low` (< 0.40), `Moderate` (0.40–0.70), `High` (≥ 0.70) |
| `Engaged_Flag` | Binary flag: `1` = engaged, `0` = not engaged |

---

## Anchor Month Format

All scripts use `MM_YYYY` format for anchor month (e.g., `01_2026` for January 2026).

The **predict month** is automatically derived:
- 30d model → anchor + 1 month

---

## Retraining an Existing Model

To retrain from an existing model artifact:

```bash
python engagement_index_training_30d.py \
    --train_file engagement_index_feature_with_target_30d.parquet \
    --anchor_month 02_2026 \
    --retrain_model engagement_index_model_30d_02_2026.pkl
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Target column not found` | Wrong `--train_file` for the horizon | Use `_30d`, `_60d`, or `_90d` feature file matching the script |
| `Missing features` | Score file missing columns from training | Re-run feature engineering on the scoring data |
| `No labeled rows found` | Target column is all NaN | Check anchor month — future months may have no labels yet |
| `FileNotFoundError` | Input file path is wrong | Verify file path and extension (`.parquet` or `.csv`) |
