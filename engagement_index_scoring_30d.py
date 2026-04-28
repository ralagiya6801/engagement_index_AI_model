"""
Engagement Index Scoring - 30d

Self-contained script: model class is embedded; no external model module required.
Loads a trained 30d model artifact and scores members from an engineered feature
file, outputting Engagement_Score and Engagement_Tier columns.

Outputs:
  - Scored output file (CSV/Parquet) with Engagement_Score and Engagement_Tier columns
  - Summary report in output directory
"""

import argparse
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE  # noqa: F401 — needed for pickle compat
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# ── Engagement Index Model (inference-side) ────────────────────────────────────

class EngagementIndexModel:
    """
    Engagement Index Forecasting Model — inference-only view.
    Full class definition retained so pickle artifacts load correctly.
    """

    def __init__(self, algorithm="XGBoost", use_temporal_features=True, use_smote=None):
        self.algorithm = algorithm
        self.use_temporal_features = use_temporal_features
        self.use_smote = use_smote if use_smote is not None else (not use_temporal_features)
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.use_scaling = algorithm == "LogisticRegression"  # QA1-3: disabled for tree models
        self.optimal_threshold = 0.08
        self.adaptive_thresholds = {}

    def _create_model(self, class_weights=None):
        """Reconstruct model skeleton — needed by load()."""
        if self.algorithm == "XGBoost":
            return xgb.XGBClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.01,
                subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                min_child_weight=1, reg_alpha=0.05, reg_lambda=0.5,
                random_state=42,
                eval_metric="mlogloss",    # multi-class log loss (4 tiers: 0-3)
                tree_method="hist", max_bin=512,
                min_split_loss=0.05,
                early_stopping_rounds=100,
                missing=np.nan,
            )
        if self.algorithm == "RandomForest":
            return RandomForestClassifier(
                n_estimators=800, max_depth=None, min_samples_split=2,
                min_samples_leaf=1, max_features="sqrt",
                class_weight="balanced_subsample", criterion="entropy",
                bootstrap=True, oob_score=True, random_state=42, n_jobs=-1,
            )
        if self.algorithm == "GradientBoosting":
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=7, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
        if self.algorithm == "LogisticRegression":
            return LogisticRegression(
                max_iter=2000, random_state=42, class_weight="balanced",
                C=0.1, solver="liblinear",
            )
        raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def get_adaptive_threshold(self, predict_month):
        """Return stored threshold for month; falls back to default (QA1-4)."""
        thr = self.adaptive_thresholds.get(predict_month, self.optimal_threshold)
        kind = "recall-optimized" if predict_month in self.adaptive_thresholds else "default"
        print(f"    Using {kind} threshold for {predict_month}: {thr:.4f}")
        return thr

    def save(self, filepath):
        """Persist model to pickle (saves data dict, not class instance)."""
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "calibrated_model": self.calibrated_model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "algorithm": self.algorithm,
                "use_scaling": self.use_scaling,
                "optimal_threshold": self.optimal_threshold,
                "adaptive_thresholds": self.adaptive_thresholds,
                "use_temporal_features": self.use_temporal_features,
                "use_smote": self.use_smote,
            }, f)

    @classmethod
    def load(cls, filepath):
        """Load model from pickle."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        instance = cls(
            data["algorithm"],
            use_temporal_features=data.get("use_temporal_features", True),
            use_smote=data.get("use_smote", None),
        )
        for key in ["model", "calibrated_model", "scaler", "feature_names",
                    "use_scaling", "optimal_threshold", "adaptive_thresholds"]:
            if key in data:
                setattr(instance, key, data[key])
        return instance


# ── Scoring helpers ────────────────────────────────────────────────────────────

def get_engagement_tier(score: float) -> str:
    """Map a continuous engagement score (0-1 probability) to a descriptive tier."""
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Moderate"
    return "Low"


def predict_engagement_compat(model: EngagementIndexModel, df: pd.DataFrame) -> np.ndarray:
    """
    Score members using stored feature list and scaling settings.
    Returns probability of positive engagement class.
    """
    x_aligned = df[model.feature_names].copy()
    for col in x_aligned.select_dtypes(include=["object", "str"]).columns:
        x_aligned[col] = pd.to_numeric(x_aligned[col], errors="coerce")
    if model.use_scaling:
        x_aligned = x_aligned.fillna(0)
        x_scaled = model.scaler.transform(x_aligned)
        x_scaled = pd.DataFrame(x_scaled, columns=x_aligned.columns)
    else:
        x_scaled = x_aligned
        predictor = model.calibrated_model if model.calibrated_model else model.model
        # P(engaged) = 1 - P(tier 0)  — works for both binary and multi-class models
        return 1.0 - predictor.predict_proba(x_scaled)[:, 0]


def detect_file_format(file_path_base: str) -> str:
    """Detect CSV or Parquet file when extension is omitted."""
    base = file_path_base.rsplit(".", 1)[0] if "." in file_path_base else file_path_base
    for ext in [".parquet", ".csv", ".CSV"]:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return file_path_base


def read_dataframe(file_path: str) -> pd.DataFrame:
    """Read scoring dataframe from Parquet or CSV."""
    actual_path = detect_file_format(file_path)
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"Scoring file not found: {file_path}")
    if actual_path.lower().endswith(".parquet"):
        print(f"Reading Parquet: {actual_path}")
        return pd.read_parquet(actual_path)
    print(f"Reading CSV: {actual_path}")
    return pd.read_csv(actual_path)


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """Save scored dataframe to Parquet or CSV based on file extension."""
    if file_path.lower().endswith(".parquet"):
        df.to_parquet(file_path, index=False)
        print(f"Saved Parquet: {file_path}")
    else:
        df.to_csv(file_path, index=False)
        print(f"Saved CSV: {file_path}")


# ── Scoring pipeline ───────────────────────────────────────────────────────────

def score_members(
    model_file: str,
    score_file: str,
    output_file: str,
    output_dir: str,
    predict_month: str,
    score_threshold: float | None,
) -> str:
    """Load model, score members, write outputs, return output path."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_file}")
    model = EngagementIndexModel.load(model_file)
    print(f"  Algorithm: {model.algorithm}")
    print(f"  Features stored: {len(model.feature_names or [])}")
    print(f"  Adaptive thresholds: {model.adaptive_thresholds}")

    df = read_dataframe(score_file)
    print(f"Scoring file rows: {len(df):,}")

    if model.feature_names is None:
        raise ValueError("Model has no stored feature_names. Re-train the model first.")

    missing_features = [f for f in model.feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Scoring file is missing {len(missing_features)} features "
            f"that the model was trained on: {missing_features[:10]}"
        )

    y_proba = predict_engagement_compat(model, df)

    if score_threshold is not None:
        threshold = score_threshold
        print(f"Using user-supplied threshold: {threshold:.4f}")
    else:
        threshold = model.get_adaptive_threshold(predict_month)

    df["Engagement_Score"] = np.round(y_proba, 6)
    df["Engagement_Tier"] = [get_engagement_tier(p) for p in y_proba]
    df["Engaged_Flag"] = (y_proba >= threshold).astype(int)

    output_cols = ["account_number", "account_id", "anchor_month",
                   "Engagement_Score", "Engagement_Tier", "Engaged_Flag"]
    output_cols = [c for c in output_cols if c in df.columns]
    save_dataframe(df[output_cols], output_file)

    engaged_count = int(df["Engaged_Flag"].sum())
    total_count = len(df)
    print(f"Engaged members: {engaged_count:,} / {total_count:,} "
          f"({engaged_count / total_count * 100:.1f}%)")
    print(f"Score distribution:\n{df['Engagement_Tier'].value_counts().to_string()}")

    optional_cols = ["account_number", "account_id", "anchor_month",
                     "Engagement_Score", "Engagement_Tier", "Engaged_Flag"]
    report_cols = [c for c in optional_cols if c in df.columns]
    if report_cols:
        df[report_cols].head(100).to_csv(
            os.path.join(output_dir, "Scoring_Report_Sample.csv"), index=False
        )

    pd.DataFrame([
        {"key": "model_file",       "value": model_file},
        {"key": "score_file",       "value": score_file},
        {"key": "predict_month",    "value": predict_month},
        {"key": "algorithm",        "value": model.algorithm},
        {"key": "threshold",        "value": f"{threshold:.6f}"},
        {"key": "n_scored",         "value": str(total_count)},
        {"key": "n_engaged",        "value": str(engaged_count)},
        {"key": "pct_engaged",      "value": f"{engaged_count / total_count * 100:.2f}"},
        {"key": "score_mean",       "value": f"{y_proba.mean():.6f}"},
        {"key": "score_median",     "value": f"{np.median(y_proba):.6f}"},
        {"key": "score_std",        "value": f"{y_proba.std():.6f}"},
    ]).to_csv(os.path.join(output_dir, "Scoring_Metadata.csv"), index=False)

    return output_file


def _add_months(anchor_month: str, n: int) -> str:
    """Advance an MM_YYYY anchor month by n months → MM_YYYY predict month."""
    mm, yyyy = anchor_month.split("_")
    total = int(mm) + n
    year = int(yyyy) + (total - 1) // 12
    month = ((total - 1) % 12) + 1
    return f"{month:02d}_{year}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scoring pipeline for 30d engagement index model."
    )
    parser.add_argument("--model_file", type=str, default=None,
                        help="Path to trained model artifact (.pkl). Defaults to engagement_index_model_30d_<predict_month>.pkl.")
    parser.add_argument("--score_file", type=str,
                        default="engagement_index_feature.parquet",
                        help="Path to engineered features file for scoring (no target required).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path for scored members. Defaults to engagement_index_30d_scoring_output_<predict_month>/engagement_index_scored_30d_<predict_month>.parquet.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save scored file and metadata. Defaults to engagement_index_30d_scoring_output_<predict_month>.")
    parser.add_argument("--anchor_month", type=str, default="01_2026",
                        help="Anchor month in MM_YYYY format. Predict month is derived as anchor + 1 month.")
    parser.add_argument("--score_threshold", type=float, default=None,
                        help="Override decision threshold (default: use model adaptive threshold).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # predict_month = anchor + 1 month (30d horizon)
    predict_month = _add_months(args.anchor_month, 1)
    model_file = args.model_file or f"engagement_index_model_30d_{predict_month}.pkl"
    output_dir = args.output_dir or f"engagement_index_30d_scoring_output_{predict_month}"
    output_file = args.output_file or os.path.join(
        output_dir, f"engagement_index_scored_30d_{predict_month}.parquet"
    )
    print(f"Anchor month: {args.anchor_month}  →  Predict month: {predict_month}")
    try:
        score_members(
            model_file=model_file,
            score_file=args.score_file,
            output_file=output_file,
            output_dir=output_dir,
            predict_month=predict_month,
            score_threshold=args.score_threshold,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
