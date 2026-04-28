"""
Engagement Index Training - 30d (Training/Validation Only)

Self-contained script: model class and all reporting helpers are embedded.
Trains a model from an engineered feature dataset that includes
target_engagement_t30d and saves a .pkl artifact for scoring.

Outputs:
  - Trained model artifact (.pkl) compatible with engagement_index_scoring_30d.py
  - Validation metrics and plots in output directory
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
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_palette("husl")
    plt.style.use("seaborn-v0_8-darkgrid")
except Exception:
    try:
        plt.style.use("seaborn-darkgrid")
    except Exception:
        plt.style.use("default")
    HAS_SEABORN = False


# ── Engagement Index Model ─────────────────────────────────────────────────────

class EngagementIndexModel:
    """Engagement Index Forecasting Model with adaptive thresholds and NaN-aware features."""

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
        """Create the underlying sklearn/XGBoost model instance."""
        if self.algorithm == "XGBoost":
            return xgb.XGBClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.01,
                subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
                min_child_weight=1, reg_alpha=0.05, reg_lambda=0.5,
                random_state=42,
                eval_metric="mlogloss",    # multi-class log loss (4 tiers: 0-3)
                tree_method="hist", max_bin=512,
                min_split_loss=0.05,       # QA3-4: single param, no gamma conflict
                early_stopping_rounds=100,  # QA3-7: early stopping to prevent overfitting
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

    def _calculate_optimal_threshold(self, X, y, target_recall=0.70):
        """
        Calculate threshold from validation data only (QA1-4: no hardcoded threshold).
        Targets minimum recall then maximises precision.
        Binarises multi-class targets: engaged = tier > 0, not engaged = tier 0.
        """
        # Binarise: any engaged tier (1/2/3) → 1, not engaged (0) → 0
        y_binary = (np.asarray(y) > 0).astype(int)

        if len(np.unique(y_binary)) < 2:
            print("    WARNING: Only one class after binarisation, using default threshold")
            return self.optimal_threshold

        y_proba = self.predict_proba(X)
        p_min, p_max = np.percentile(y_proba, [0.1, 95])
        thresholds = np.linspace(max(p_min, 0.001), min(p_max, 0.5), 300)
        print(f"    Searching {len(thresholds)} thresholds, target recall >= {target_recall:.1%}")

        valid = []
        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            if y_pred.sum() == 0:
                continue
            rec = recall_score(y_binary, y_pred, zero_division=0)
            prec = precision_score(y_binary, y_pred, zero_division=0)
            if rec >= target_recall:
                valid.append({"threshold": thr, "recall": rec, "precision": prec,
                               "f2": fbeta_score(y_binary, y_pred, beta=2.0, zero_division=0)})

        if not valid:
            print(f"    WARNING: No threshold achieves {target_recall:.1%} recall; maximising recall")
            best_thr = self.optimal_threshold
            best_rec = -1.0
            for thr in thresholds:
                y_pred = (y_proba >= thr).astype(int)
                if y_pred.sum() == 0:
                    continue
                rec = recall_score(y_binary, y_pred, zero_division=0)
                if rec > best_rec:
                    best_rec, best_thr = rec, thr
        else:
            print(f"    Found {len(valid)} thresholds meeting target recall")
            valid.sort(key=lambda x: x["precision"], reverse=True)
            best_thr = valid[0]["threshold"]

        y_best = (y_proba >= best_thr).astype(int)
        print(
            f"    Optimal: {best_thr:.4f}  "
            f"Recall={recall_score(y_binary, y_best, zero_division=0):.4f}  "
            f"Precision={precision_score(y_binary, y_best, zero_division=0):.4f}  "
            f"F2={fbeta_score(y_binary, y_best, beta=2.0, zero_division=0):.4f}"
        )
        return best_thr

    def train(self, X_train, y_train, X_val=None, y_val=None,
              sample_weights=None, predict_month=None):
        """Train model with proper validation — all QA fixes applied."""
        self.feature_names = list(X_train.columns)

        classes = np.unique(y_train)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        if len(class_weights) == 2:
            class_weights[1] *= 9.5
        print(f"    Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"    Class weights: {dict(zip(classes, class_weights))}")

        if sample_weights is None:
            quality_col = next(
                (c for c in ["historical_data_quality", "data_quality_score"] if c in X_train.columns),
                None,
            )
            if quality_col:
                completeness = X_train[quality_col].values
            else:
                feat_cols = [c for c in X_train.columns
                             if c not in {"age", "data_quality_score", "historical_data_quality"}]
                completeness = X_train[feat_cols].notna().sum(axis=1) / max(len(feat_cols), 1)
            sample_weights = 1.0 + completeness
            print(f"    Sample weights: min={sample_weights.min():.2f}, "
                  f"max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")

        # QA1-5: SMOTE disabled to prevent temporal mixing
        if not self.use_smote:
            print("    SMOTE disabled (prevents temporal mixing)")

        for col in X_train.select_dtypes(include=["object", "str"]).columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        if X_val is not None:
            for col in X_val.select_dtypes(include=["object", "str"]).columns:
                X_val[col] = pd.to_numeric(X_val[col], errors="coerce")

        # QA1-3: scaling disabled for tree-based models
        X_tr = self.scaler.fit_transform(X_train) if self.use_scaling else X_train
        if self.use_scaling:
            X_tr = pd.DataFrame(X_tr, columns=X_train.columns)

        self.model = self._create_model(class_weights if self.algorithm == "XGBoost" else None)

        if X_val is not None and y_val is not None and self.algorithm == "XGBoost":
            X_vl = self.scaler.transform(X_val) if self.use_scaling else X_val
            if self.use_scaling:
                X_vl = pd.DataFrame(X_vl, columns=X_val.columns)
            self.model.fit(X_tr, y_train, sample_weight=sample_weights,
                           eval_set=[(X_vl, y_val)], verbose=False)
            print(f"    Early stopping: Best iteration = {self.model.best_iteration}")
        else:
            self.model.fit(X_tr, y_train, sample_weight=sample_weights)

        # QA3-5: calibration disabled (avoids overconfident probs on training data)
        self.calibrated_model = None
        print("    Calibration: DISABLED (using native XGBoost probabilities)")

        if X_val is not None and y_val is not None:
            print("    Calculating optimal threshold from validation data...")
            opt_thr = self._calculate_optimal_threshold(X_val, y_val, target_recall=0.70)
            if predict_month:
                self.adaptive_thresholds[predict_month] = opt_thr
                print(f"    Stored adaptive threshold for {predict_month}: {opt_thr:.4f}")
            else:
                self.optimal_threshold = opt_thr

        print(f"  Model trained: {self.algorithm}")
        return self.adaptive_thresholds.get(predict_month, self.optimal_threshold)

    def predict_proba(self, X):
        """Return predicted probabilities (positive class)."""
        X_aligned = X[self.feature_names].copy()
        for col in X_aligned.select_dtypes(include=["object", "str"]).columns:
            X_aligned[col] = pd.to_numeric(X_aligned[col], errors="coerce")
        if self.use_scaling:
            X_aligned = X_aligned.fillna(0)
        X_sc = self.scaler.transform(X_aligned) if self.use_scaling else X_aligned
        if self.use_scaling:
            X_sc = pd.DataFrame(X_sc, columns=X_aligned.columns)
        predictor = self.calibrated_model if self.calibrated_model else self.model
        # P(engaged) = 1 - P(tier 0)  — works for both binary and multi-class models
        return 1.0 - predictor.predict_proba(X_sc)[:, 0]

    def get_adaptive_threshold(self, predict_month):
        """Return stored threshold for month; falls back to default (QA1-4)."""
        thr = self.adaptive_thresholds.get(predict_month, self.optimal_threshold)
        kind = "recall-optimized" if predict_month in self.adaptive_thresholds else "default"
        print(f"    Using {kind} threshold for {predict_month}: {thr:.4f}")
        return thr

    def predict(self, X, threshold=None, predict_month=None):
        """Predict binary engagement label."""
        if threshold is None:
            threshold = self.get_adaptive_threshold(predict_month) if predict_month else self.optimal_threshold
        return (self.predict_proba(X) >= threshold).astype(int)

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


# ── Reporting helpers ──────────────────────────────────────────────────────────

def plot_roc_auc(y_true, y_proba, title, output_file):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "darkorange", lw=2, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, output_file):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                    xticklabels=["Not Engaged", "Engaged"],
                    yticklabels=["Not Engaged", "Engaged"])
    else:
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, cm[i, j], ha="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.xticks([0, 1], ["Not Engaged", "Engaged"])
        plt.yticks([0, 1], ["Not Engaged", "Engaged"])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_txt(y_true, y_pred, y_proba, output_file, phase="Validation"):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "FPR":       fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "FNR":       fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }
    with open(output_file, "w") as f:
        f.write("=" * 80 + f"\nCONFUSION MATRIX - {phase.upper()}\n" + "=" * 80 + "\n\n")
        f.write(f"{'':30} {'Predicted: Not Engaged':22} {'Predicted: Engaged':20}\n")
        f.write(f"{'Actual: Not Engaged':30} {tn:20,d} {fp:20,d}\n")
        f.write(f"{'Actual: Engaged':30} {fn:20,d} {tp:20,d}\n\n")
        f.write("METRICS:\n" + "-" * 80 + "\n")
        for name, val in metrics.items():
            f.write(f"{name:15}: {val:.4f}\n")
        f.write("\nINTERPRETATION:\n")
        f.write(f"False Positives: {fp:,} over-engaged flags ({metrics['FPR'] * 100:.1f}%)\n")
        f.write(f"False Negatives: {fn:,} missed engagements ({metrics['FNR'] * 100:.1f}%)\n")


def evaluate_model(y_true, y_pred, y_proba, output_file):
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
    }
    with open(output_file, "w") as f:
        f.write("=" * 60 + "\nPERFORMANCE SUMMARY\n" + "=" * 60 + "\n\n")
        for name, val in metrics.items():
            f.write(f"{name:15}: {val:.4f}\n")
        f.write("\n" + classification_report(y_true, y_pred,
                                             target_names=["Not Engaged", "Engaged"]))
    print(f"  Accuracy: {metrics['Accuracy']:.4f}, ROC-AUC: {metrics['ROC-AUC']:.4f}")


# ── Training pipeline ──────────────────────────────────────────────────────────

def detect_file_format(file_path_base: str) -> str:
    """Detect CSV or Parquet file when extension is omitted."""
    base = file_path_base.rsplit(".", 1)[0] if "." in file_path_base else file_path_base
    for ext in [".parquet", ".csv", ".CSV"]:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return file_path_base


def read_dataframe(file_path: str) -> pd.DataFrame:
    """Read training dataframe from Parquet or CSV."""
    actual_path = detect_file_format(file_path)
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"Training file not found: {file_path}")
    if actual_path.lower().endswith(".parquet"):
        print(f"Reading Parquet: {actual_path}")
        return pd.read_parquet(actual_path)
    print(f"Reading CSV: {actual_path}")
    return pd.read_csv(actual_path)


def _infer_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Infer feature columns by excluding identifiers and all target columns.

    All three horizon targets are always excluded regardless of which one is active.
    This prevents cross-horizon label leakage even if a combined file is passed in.
    """
    exclude_cols = {
        target_col,
        "target_engagement_t30d",
        "target_engagement_t60d",
        "target_engagement_t90d",
        "account_number",
        "account_id",
        "anchor_month",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding ID/target columns.")
    return feature_cols


def _sanitize_target(y: pd.Series, target_col: str) -> pd.Series:
    """Ensure valid engagement tier values (0–3)."""
    y_numeric = pd.to_numeric(y, errors="coerce")
    if y_numeric.isna().any():
        raise ValueError(
            f"Target column '{target_col}' contains null or non-numeric values "
            "after labeled-row filtering. Expected engagement tier values 0-3."
        )
    unique_vals = sorted(y_numeric.unique().tolist())
    if not set(unique_vals).issubset({0, 1, 2, 3}):
        raise ValueError(
            f"Target column '{target_col}' must contain engagement tiers 0-3. "
            f"Found values: {unique_vals}"
        )
    return y_numeric.astype(int)


def _select_threshold_with_error_caps(
    y_true: pd.Series,
    y_proba: np.ndarray,
    max_fpr: float = 0.30,
    max_fnr: float = 0.25,
    min_threshold: float = 0.03,
    fpr_weight: float = 3.0,
    fnr_weight: float = 1.0,
) -> tuple[float, dict]:
    """
    QA-safe threshold selection from validation set only (QA1-4 / QA3-6).
    Primary goal: enforce FNR cap, then minimise FPR.
    """
    # Binarise: any engaged tier (1/2/3) → 1, not engaged (tier 0) → 0
    y_arr = (np.asarray(y_true) > 0).astype(int)
    p_arr = np.asarray(y_proba).astype(float)

    unique_scores = np.unique(np.round(p_arr, 6))
    thresholds = np.unique(np.concatenate(([0.0], unique_scores, [1.0])))
    thresholds = thresholds[thresholds >= float(min_threshold)]
    if thresholds.size == 0:
        thresholds = np.array([float(min_threshold)])

    evaluated: list[dict] = []
    for thr in thresholds:
        y_pred = (p_arr >= thr).astype(int)
        cm = confusion_matrix(y_arr, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        entry = {
            "threshold": float(thr),
            "fpr": float(fpr), "fnr": float(fnr),
            "recall": float(recall_score(y_arr, y_pred, zero_division=0)),
            "precision": float(precision_score(y_arr, y_pred, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "target_distance": fpr_weight * abs(fpr - max_fpr) + fnr_weight * abs(fnr - max_fnr),
        }
        evaluated.append(entry)

    feasible = [e for e in evaluated if e["fnr"] <= max_fnr]
    if feasible:
        chosen = min(feasible, key=lambda e: (
            e["fpr"], abs(e["fnr"] - max_fnr), -e["precision"], -e["threshold"]
        ))
        chosen["selection_mode"] = "fnr_cap_then_min_fpr"
        return float(chosen["threshold"]), chosen

    chosen = min(evaluated, key=lambda e: (
        e["fnr"], abs(e["fpr"] - max_fpr), e["target_distance"], -e["precision"], -e["threshold"]
    ))
    chosen["selection_mode"] = "min_fnr_fallback"
    return float(chosen["threshold"]), chosen


def train_model(
    train_file: str,
    target_col: str,
    output_dir: str,
    model_file: str,
    predict_month: str,
    algorithm: str,
    val_size: float,
    random_state: int,
    max_validation_fpr: float,
    max_validation_fnr: float,
    min_threshold: float,
    fpr_weight: float,
    fnr_weight: float,
    retrain_model: str | None,
) -> str:
    """Train model, validate, save artifacts, and return model path."""
    os.makedirs(output_dir, exist_ok=True)

    df = read_dataframe(train_file)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in input file. "
            "Training requires features + engagement tier target."
        )

    target_numeric = pd.to_numeric(df[target_col], errors="coerce")
    labeled_mask = target_numeric.notna()
    unlabeled_count = int((~labeled_mask).sum())
    if unlabeled_count > 0:
        print(f"Dropping unlabeled rows (missing {target_col}): {unlabeled_count:,}")
    df_labeled = df.loc[labeled_mask].reset_index(drop=True)
    if df_labeled.empty:
        raise ValueError(
            f"No labeled rows found in '{target_col}'. "
            "Cannot train without at least some non-null labels."
        )

    feature_cols = _infer_feature_columns(df_labeled, target_col)
    X = df_labeled[feature_cols].copy()
    y = _sanitize_target(df_labeled[target_col], target_col)

    print(f"Rows (input): {len(df):,}")
    print(f"Rows (labeled): {len(df_labeled):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution: {dict(y.value_counts().sort_index())}")
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"Target column '{target_col}' has only one class. "
            "Need at least two classes to train a classifier."
        )

    # QA3-1: stratified split to preserve class distribution
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    idx = np.arange(len(df_labeled))
    train_idx, val_idx = next(splitter.split(idx, y))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    print(f"Train rows: {len(X_train):,}, distribution: {dict(y_train.value_counts().sort_index())}")
    print(f"Val rows:   {len(X_val):,}, distribution: {dict(y_val.value_counts().sort_index())}")

    if retrain_model:
        print(f"Loading existing model for retraining: {retrain_model}")
        model = EngagementIndexModel.load(retrain_model)
        model.algorithm = algorithm
    else:
        model = EngagementIndexModel(
            algorithm=algorithm,
            use_temporal_features=True,
            use_smote=False,  # QA1-5: SMOTE disabled
        )

    model.train(X_train, y_train, X_val=X_val, y_val=y_val, predict_month=predict_month)

    y_val_proba = model.predict_proba(X_val)
    # Binarise for all threshold selection and binary reporting:
    # engaged = any tier > 0, not engaged = tier 0
    y_val_bin = (y_val > 0).astype(int)

    # QA1-4 / QA3-6: threshold selected from validation only, never test data
    capped_threshold, threshold_stats = _select_threshold_with_error_caps(
        y_true=y_val_bin,
        y_proba=y_val_proba,
        max_fpr=max_validation_fpr,
        max_fnr=max_validation_fnr,
        min_threshold=min_threshold,
        fpr_weight=fpr_weight,
        fnr_weight=fnr_weight,
    )
    model.adaptive_thresholds[predict_month] = capped_threshold
    print(
        f"Validation threshold update "
        f"(max FPR={max_validation_fpr:.2f}, max FNR={max_validation_fnr:.2f}, "
        f"min_thr={min_threshold:.3f}, fpr_w={fpr_weight:.2f}, fnr_w={fnr_weight:.2f}): "
        f"mode={threshold_stats.get('selection_mode', 'unknown')}, "
        f"threshold={capped_threshold:.4f}, "
        f"FPR={threshold_stats['fpr']:.4f}, FNR={threshold_stats['fnr']:.4f}, "
        f"Recall={threshold_stats['recall']:.4f}, Precision={threshold_stats['precision']:.4f}"
    )
    if threshold_stats["fnr"] > max_validation_fnr:
        print(
            f"WARNING: Could not satisfy FNR cap <= {max_validation_fnr:.2f}. "
            f"Best achievable FNR={threshold_stats['fnr']:.4f} on validation."
        )

    y_val_pred = model.predict(X_val, predict_month=predict_month)
    val_auc = roc_auc_score(y_val_bin, y_val_proba) if len(np.unique(y_val_bin)) > 1 else 0.0
    print(f"Validation ROC-AUC: {val_auc:.4f}")

    plot_roc_auc(y_val_bin, y_val_proba,
                 f"ROC-AUC Validation - 30d ({predict_month})",
                 os.path.join(output_dir, "ROC_AUC_Val.png"))
    plot_confusion_matrix(y_val_bin, y_val_pred,
                          f"CM Validation - 30d ({predict_month})",
                          os.path.join(output_dir, "CM_Val.png"))
    save_confusion_matrix_txt(y_val_bin, y_val_pred, y_val_proba,
                              os.path.join(output_dir, "CM_Val.txt"),
                              f"Validation 30d {predict_month}")
    evaluate_model(y_val_bin, y_val_pred, y_val_proba,
                   os.path.join(output_dir, "Performance_Val.txt"))

    model.save(model_file)
    print(f"Saved model: {model_file}")

    pd.DataFrame([
        {"key": "train_file",               "value": train_file},
        {"key": "target_col",               "value": target_col},
        {"key": "predict_month",            "value": predict_month},
        {"key": "algorithm",                "value": algorithm},
        {"key": "val_size",                 "value": str(val_size)},
        {"key": "random_state",             "value": str(random_state)},
        {"key": "n_rows_input",             "value": str(len(df))},
        {"key": "n_rows_labeled",           "value": str(len(df_labeled))},
        {"key": "n_rows_unlabeled",         "value": str(unlabeled_count)},
        {"key": "n_features",               "value": str(len(feature_cols))},
        {"key": "val_roc_auc",              "value": f"{val_auc:.6f}"},
        {"key": "threshold_predict_month",  "value": f"{capped_threshold:.6f}"},
        {"key": "threshold_fpr",            "value": f"{threshold_stats['fpr']:.6f}"},
        {"key": "threshold_fnr",            "value": f"{threshold_stats['fnr']:.6f}"},
        {"key": "threshold_recall",         "value": f"{threshold_stats['recall']:.6f}"},
        {"key": "threshold_precision",      "value": f"{threshold_stats['precision']:.6f}"},
        {"key": "max_validation_fpr",       "value": f"{max_validation_fpr:.6f}"},
        {"key": "max_validation_fnr",       "value": f"{max_validation_fnr:.6f}"},
        {"key": "min_threshold",            "value": f"{min_threshold:.6f}"},
        {"key": "fpr_weight",               "value": f"{fpr_weight:.6f}"},
        {"key": "fnr_weight",               "value": f"{fnr_weight:.6f}"},
        {"key": "threshold_target_distance","value": f"{threshold_stats.get('target_distance', np.nan):.6f}"},
        {"key": "threshold_selection_mode", "value": str(threshold_stats.get("selection_mode", "unknown"))},
    ]).to_csv(os.path.join(output_dir, "Training_Metadata.csv"), index=False)

    return model_file


def _add_months(anchor_month: str, n: int) -> str:
    """Advance an MM_YYYY anchor month by n months → MM_YYYY predict month."""
    mm, yyyy = anchor_month.split("_")
    total = int(mm) + n
    year = int(yyyy) + (total - 1) // 12
    month = ((total - 1) % 12) + 1
    return f"{month:02d}_{year}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training/validation pipeline for 30d engagement index model."
    )
    parser.add_argument("--train_file", type=str,
                        default="engagement_index_feature_with_target_30d.parquet",
                        help="Path to engineered training file containing only target_engagement_t30d.")
    parser.add_argument("--target_col", type=str, default="target_engagement_t30d",
                        help="Engagement tier target column name (0-3).")
    parser.add_argument("--anchor_month", type=str, default="01_2026",
                        help="Anchor month in MM_YYYY format. Predict month is derived as anchor + 1 month.")
    parser.add_argument("--algorithm", type=str, default="XGBoost",
                        choices=["XGBoost", "RandomForest", "GradientBoosting", "LogisticRegression"],
                        help="Model algorithm.")
    parser.add_argument("--val_size", type=float, default=0.3,
                        help="Validation split fraction.")
    parser.add_argument("--max_validation_fpr", type=float, default=0.30,
                        help="Maximum allowed validation FPR when selecting decision threshold.")
    parser.add_argument("--max_validation_fnr", type=float, default=0.30,
                        help="Maximum allowed validation FNR when selecting decision threshold.")
    parser.add_argument("--min_threshold", type=float, default=0.03,
                        help="Minimum decision threshold floor to avoid ultra-low cutoffs.")
    parser.add_argument("--fpr_weight", type=float, default=1.0,
                        help="Penalty weight for FPR-cap violation in weighted tradeoff fallback.")
    parser.add_argument("--fnr_weight", type=float, default=3.0,
                        help="Penalty weight for FNR-cap violation in weighted tradeoff fallback.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for stratified split.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save validation outputs and metadata. "
                             "Defaults to engagement_index_30d_training_output_{predict_month}.")
    parser.add_argument("--model_file", type=str, default=None,
                        help="Output path for saved model artifact (.pkl). "
                             "Defaults to engagement_index_model_30d_{predict_month}.pkl.")
    parser.add_argument("--retrain_model", type=str, default=None,
                        help="Path to existing model (.pkl) to load and retrain.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # predict_month = anchor + 1 month (30d horizon)
    predict_month = _add_months(args.anchor_month, 1)
    output_dir = args.output_dir or f"engagement_index_30d_training_output_{predict_month}"
    model_file = args.model_file or f"engagement_index_model_30d_{predict_month}.pkl"
    print(f"Anchor month: {args.anchor_month}  →  Predict month: {predict_month}")
    try:
        train_model(
            train_file=args.train_file,
            target_col=args.target_col,
            output_dir=output_dir,
            model_file=model_file,
            predict_month=predict_month,
            algorithm=args.algorithm,
            val_size=args.val_size,
            random_state=args.random_state,
            max_validation_fpr=args.max_validation_fpr,
            max_validation_fnr=args.max_validation_fnr,
            min_threshold=args.min_threshold,
            fpr_weight=args.fpr_weight,
            fnr_weight=args.fnr_weight,
            retrain_model=args.retrain_model,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

