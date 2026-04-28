"""engagement_index_feature_engineering.py

Feature engineering for the Engagement Index AI model.

Input:
  - Long-format base feature file (one row per account per obs_month)
  - Portal engagement snapshot (one row per account)

Output:
  - Engineered feature file with 7 pillar features + 3 target columns:
      target_engagement_t30d, target_engagement_t60d, target_engagement_t90d
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Product lookup ─────────────────────────────────────────────────────────────

PRODUCT_CATEGORY_MAP: Dict[str, str] = {
    "Active Guardian":               "Mobile",
    "Active Guardian Black":         "Mobile",
    "Active Guardian White":         "Mobile",
    "Classic Guardian":              "In-home",
    "Elite 911":                     "Mobile",
    "Family Guardian":               "Unknown",
    "Freedom Guardian 2 Black":      "Wearable",
    "Freedom Guardian 2 White":      "Wearable",
    "Freedom Guardian Black":        "Wearable",
    "Freedom Guardian Black Refurb": "Wearable",
    "Freedom Guardian White":        "Wearable",
    "Freedom Guardian White Refurb": "Wearable",
    "Home 2.0":                      "In-home",
    "Home Guardian":                 "In-home",
    "Home Guardian Combo":           "In-home",
    "Home Guardian LT":              "In-home",
    "MG Move":                       "Wearable",
    "MG Move XL":                    "Wearable",
    "MGHome Landline":               "In-home",
    "MGMini Black":                  "Wearable",
    "MGMini Lite ATT":               "Wearable",
    "MGMini Lite VZ":                "Wearable",
    "MGMini Lite VZ Black":          "Wearable",
    "MGMini Pearl":                  "Wearable",
    "MGMini Rose Gold":              "Wearable",
    "MGMini Silver":                 "Wearable",
    "MH Classic Cellular":           "In-home",
    "MH Classic Landline":           "In-home",
    "MH Micro":                      "Unknown",
    "MH Solo":                       "Unknown",
    "Mini Guardian Black":           "Mobile",
    "Mini Guardian Silver":          "Mobile",
    "Mini Guardian White":           "Mobile",
    "Mobile 2.0":                    "Mobile",
    "Mobile Guardian":               "Mobile",
    "On-The-Go Guardian":            "Mobile",
    "Premium Guardian":              "Mobile",
}

_CURRENT_PRODUCTS = {
    "MG Move", "MG Move XL", "MGHome Landline",
    "MGMini Black", "MGMini Lite ATT", "MGMini Lite VZ",
    "MGMini Lite VZ Black", "MGMini Pearl", "MGMini Rose Gold", "MGMini Silver",
}

PRODUCT_PROFILE_ENCODING: Dict[str, int] = {
    "Unknown": 0, "Mobile": 1, "In-home": 2, "Wearable": 3,
}

_PRODUCT_NAME_ENCODING: Dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(sorted(PRODUCT_CATEGORY_MAP.keys()))
}

# ── Constants ──────────────────────────────────────────────────────────────────

EPS = 1e-6
LOW_STEPS_THRESHOLD = 2000
MIN_ANCHOR_MONTH = pd.Timestamp("2025-11-01")

STATIC_OUTPUT_COLUMNS = ["account_number", "account_id", "age", "anchor_month"]

ENGAGEMENT_FILE_PATTERN = re.compile(
    r"^engagement_index_feature_(?P<month>\d{2})_(?P<year>\d{4})\.(parquet|xlsx|xls|csv)$",
    re.IGNORECASE,
)

COLS_TO_DROP = ["brand", "prev_avg_daily_steps", "steps_delta"]

ACTIVITY_COLS = [
    "buttons_count",
    "assist_count",
    "fall_count",
    "er_dispatch_count",
    "help_sent_count",
    "subscriber_reached_count",
    "monitoring_outreach_count",
    "case_count",
]


@dataclass(frozen=True)
class MonthInfo:
    key: str
    timestamp: pd.Timestamp


# ── Path resolution ────────────────────────────────────────────────────────────

def _resolve_input_path(raw_path: str) -> str:
    if os.path.exists(raw_path):
        return raw_path

    root, ext = os.path.splitext(raw_path)
    candidates: List[str] = []
    if ext:
        candidates.append(raw_path)
    else:
        for candidate_ext in [".parquet", ".xlsx", ".xls", ".csv"]:
            candidates.append(root + candidate_ext)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    parent_dir = os.path.dirname(raw_path) or "."
    base_name = os.path.basename(raw_path)
    if os.path.isdir(parent_dir):
        entries = os.listdir(parent_dir)
        entry_map = {e.lower(): e for e in entries}
        if base_name.lower() in entry_map:
            return os.path.join(parent_dir, entry_map[base_name.lower()])
        stem = os.path.splitext(base_name)[0].lower()
        for entry in entries:
            entry_stem, entry_ext = os.path.splitext(entry)
            if entry_stem.lower() == stem and entry_ext.lower() in {".parquet", ".xlsx", ".xls", ".csv"}:
                return os.path.join(parent_dir, entry)

    raise FileNotFoundError(f"Input file not found: {raw_path}")


# ── File loading ───────────────────────────────────────────────────────────────

def load_input(file_path: str) -> pd.DataFrame:
    """Load long-format base feature file; enforce required static columns."""
    resolved = _resolve_input_path(file_path)
    ext = os.path.splitext(resolved)[1].lower()

    if ext == ".parquet":
        df = pd.read_parquet(resolved)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(resolved)
    elif ext == ".csv":
        df = pd.read_csv(resolved)
    else:
        raise ValueError(f"Unsupported extension '{ext}'. Supported: .parquet, .xlsx, .xls, .csv")

    print(f"Input file resolved: {resolved}")

    required = ["account_number", "account_id", "age", "obs_month", "product_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required static columns: {missing}")

    df = df.copy().reset_index(drop=True)
    df["account_number"] = df["account_number"].astype(str).str.strip()
    df["account_id"] = df["account_id"].astype(str).str.strip()

    for col in COLS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped column: {col}")

    df["obs_month_dt"] = pd.to_datetime(
        df["obs_month"].astype(str).str.strip() + "-01", errors="coerce"
    )
    invalid = int(df["obs_month_dt"].isna().sum())
    if invalid > 0:
        print(f"Warning: {invalid} rows with unparseable obs_month excluded.")
        df = df[df["obs_month_dt"].notna()].copy()

    # Pre-convert numeric columns once to avoid repeated per-group to_numeric calls.
    skip_cols = {"account_number", "account_id", "obs_month", "obs_month_dt",
                 "product_name", "email_last_send_month",
                 "email_last_open_month", "email_last_click_month"}
    for col in df.columns:
        if col not in skip_cols and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(
        f"Input rows: {len(df)}, "
        f"obs_month range: {df['obs_month_dt'].min().strftime('%Y-%m')} "
        f"to {df['obs_month_dt'].max().strftime('%Y-%m')}"
    )
    return df


def load_portal(portal_path: str) -> pd.DataFrame:
    """Load portal engagement snapshot file."""
    resolved = _resolve_input_path(portal_path)
    ext = os.path.splitext(resolved)[1].lower()

    if ext == ".parquet":
        pdf = pd.read_parquet(resolved)
    elif ext == ".csv":
        pdf = pd.read_csv(resolved)
    elif ext in {".xlsx", ".xls"}:
        pdf = pd.read_excel(resolved)
    else:
        raise ValueError(f"Unsupported portal file extension: {ext}")

    if "account_number" not in pdf.columns:
        raise ValueError("Portal file missing required column: account_number")

    pdf = pdf.copy()
    pdf["account_number"] = pdf["account_number"].astype(str).str.strip()
    print(f"Portal file loaded: {resolved}, rows: {len(pdf)}")
    return pdf


def detect_latest_engagement_file(input_dir: str) -> str:
    """Detect latest engagement_index_feature_MM_YYYY file in directory."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    candidates: List[Tuple[int, str]] = []
    for entry in os.listdir(input_dir):
        match = ENGAGEMENT_FILE_PATTERN.match(entry)
        if not match:
            continue
        mm = int(match.group("month"))
        yyyy = int(match.group("year"))
        candidates.append((yyyy * 100 + mm, os.path.join(input_dir, entry)))

    if not candidates:
        raise FileNotFoundError(
            "No files detected with pattern 'engagement_index_feature_MM_YYYY.<ext>' in: "
            f"{input_dir}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    latest = candidates[0][1]
    print(f"Auto-detected latest engagement index file: {latest}")
    return latest


# ── Window helpers ─────────────────────────────────────────────────────────────

def _get_window_df(df: pd.DataFrame, anchor_ts: pd.Timestamp, months_back: int) -> pd.DataFrame:
    """Return rows in [anchor - months_back + 1 months, anchor] (both inclusive)."""
    start = (anchor_ts - pd.DateOffset(months=months_back - 1)).replace(day=1)
    end = anchor_ts.replace(day=1)
    return df[(df["obs_month_dt"] >= start) & (df["obs_month_dt"] <= end)]


def _col_sum(wdf: pd.DataFrame, col: str) -> pd.Series:
    """Sum a column per account_id; NaN if column absent or all-NaN."""
    if col not in wdf.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(wdf[col], errors="coerce")
    return wdf.assign(**{col: s}).groupby("account_id")[col].sum(min_count=1)


def _col_mean(wdf: pd.DataFrame, col: str) -> pd.Series:
    """Mean of a column per account_id."""
    if col not in wdf.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(wdf[col], errors="coerce")
    return wdf.assign(**{col: s}).groupby("account_id")[col].mean()


def _any_positive(wdf: pd.DataFrame, *cols: str) -> pd.Series:
    """1 if any of cols > 0 in any row per account_id, else 0."""
    present = [c for c in cols if c in wdf.columns]
    if not present:
        return pd.Series(0, index=wdf["account_id"].unique(), dtype=np.int8)
    flag = (
        wdf[present].apply(pd.to_numeric, errors="coerce").gt(0).any(axis=1)
    )
    return wdf.assign(_flag=flag).groupby("account_id")["_flag"].any().astype(np.int8)


# ── Product feature helpers ────────────────────────────────────────────────────

def _clean_product_name(name: object) -> str:
    if pd.isna(name):
        return ""
    return re.sub(r"\s*\(.*?\)", "", str(name)).strip()


def _build_product_features(df_full: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.DataFrame:
    """Return product features per account using most recent product_name as of anchor."""
    all_account_ids = df_full["account_id"].unique()
    base = pd.DataFrame({"account_id": all_account_ids})

    if "product_name" not in df_full.columns:
        base["product_name_encoded"] = 0
        base["has_product_name_flag"] = 0
        base["product_engagement_profile"] = 0
        return base

    df_up = df_full[df_full["obs_month_dt"] <= anchor_ts].copy()
    df_up = df_up.sort_values("obs_month_dt")

    latest = (
        df_up[df_up["product_name"].notna() & (df_up["product_name"].astype(str).str.strip() != "")]
        .groupby("account_id")["product_name"]
        .last()
        .reset_index()
    )
    latest["product_name_clean"] = latest["product_name"].apply(_clean_product_name)
    latest["product_engagement_profile"] = (
        latest["product_name_clean"]
        .map(PRODUCT_CATEGORY_MAP)
        .fillna("Unknown")
        .map(PRODUCT_PROFILE_ENCODING)
        .fillna(0)
        .astype(np.int8)
    )
    latest["product_name_encoded"] = (
        latest["product_name_clean"]
        .map(_PRODUCT_NAME_ENCODING)
        .fillna(0)
        .astype(np.int16)
    )
    latest["has_product_name_flag"] = np.int8(1)

    result = base.merge(
        latest[["account_id", "product_name_encoded", "has_product_name_flag", "product_engagement_profile"]],
        on="account_id",
        how="left",
    )
    result["product_name_encoded"] = result["product_name_encoded"].fillna(0).astype(np.int16)
    result["has_product_name_flag"] = result["has_product_name_flag"].fillna(0).astype(np.int8)
    result["product_engagement_profile"] = result["product_engagement_profile"].fillna(0).astype(np.int8)
    return result


# ── Recency helpers ────────────────────────────────────────────────────────────

def _months_between_date_col(series: pd.Series, ref_ts: pd.Timestamp) -> pd.Series:
    """
    Compute months between a date column and ref_ts.
    Handles both MM_YYYY (e.g. '12_2025') and YYYY-MM (e.g. '2025-12') formats.
    Safely handles NaN/float values.
    """
    mm_yyyy = re.compile(r"^(\d{1,2})_(\d{4})$")

    def _normalize(v: object) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        s = str(v).strip()
        m = mm_yyyy.match(s)
        if m:
            return f"{m.group(2)}-{m.group(1).zfill(2)}"
        return s[:7]

    normalized = series.apply(_normalize)
    parsed = pd.to_datetime(normalized + "-01", format="%Y-%m-%d", errors="coerce")
    diff = (ref_ts.year - parsed.dt.year) * 12 + (ref_ts.month - parsed.dt.month)
    return diff.where(parsed.notna(), other=np.nan)


def _consecutive_inactive_months(df_up: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.Series:
    """
    For each account, count consecutive months of zero activity ending at anchor.
    Uses vectorised pivot + cumprod for efficiency.
    """
    activity_cols_present = [c for c in ACTIVITY_COLS if c in df_up.columns]
    if not activity_cols_present:
        return pd.Series(0, index=df_up["account_id"].unique(), dtype=int)

    tmp = df_up.copy()
    tmp["_active"] = (
        tmp[activity_cols_present]
        .apply(pd.to_numeric, errors="coerce")
        .gt(0)
        .any(axis=1)
        .astype(int)
    )

    monthly = (
        tmp.groupby(["account_id", "obs_month_dt"])["_active"]
        .max()
        .unstack("obs_month_dt")
        .fillna(0)
    )
    monthly = monthly[sorted(monthly.columns, reverse=True)]

    arr = monthly.values
    streaks = (arr == 0).cumprod(axis=1).sum(axis=1)
    return pd.Series(streaks.astype(int), index=monthly.index)


def _months_since_device_event(df_up: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.Series:
    """Months since last month where buttons_count > 0 or assist_count > 0."""
    device_cols = [c for c in ["buttons_count", "assist_count"] if c in df_up.columns]
    if not device_cols:
        return pd.Series(np.nan, index=df_up["account_id"].unique())

    tmp = df_up.copy()
    tmp["_dev"] = (
        tmp[device_cols].apply(pd.to_numeric, errors="coerce").gt(0).any(axis=1)
    )
    last_active = (
        tmp[tmp["_dev"]].groupby("account_id")["obs_month_dt"].max()
    )
    diff = (
        (anchor_ts.year - last_active.dt.year) * 12
        + (anchor_ts.month - last_active.dt.month)
    )
    return diff


# ── Target computation ─────────────────────────────────────────────────────────

def _compute_engagement_tier(
    df_label: pd.DataFrame,
    portal_df: Optional[pd.DataFrame],
    df_full_up_to_label: pd.DataFrame,
    label_ts: pd.Timestamp,
    account_ids: np.ndarray,
) -> pd.Series:
    """
    Compute engagement tier (0–3) for each account from label-month base features.
    Returns a float Series indexed by position matching account_ids.
    """
    result = pd.Series(np.nan, index=range(len(account_ids)), dtype=float)

    if df_label.empty:
        return result

    # Aggregate label month per account
    cnt_cols = [
        "buttons_count", "avg_daily_steps", "monitoring_outreach_count",
        "case_count", "er_dispatch_count", "assist_count", "fall_count",
        "email_open_count", "email_click_count",
    ]
    agg_dict = {c: "sum" for c in cnt_cols if c in df_label.columns}
    if "avg_daily_steps" in agg_dict:
        agg_dict["avg_daily_steps"] = "mean"

    label_agg = df_label.groupby("account_id").agg(agg_dict)

    def _get(col: str) -> pd.Series:
        if col in label_agg.columns:
            return pd.to_numeric(label_agg[col], errors="coerce").reindex(account_ids).fillna(0)
        return pd.Series(0.0, index=account_ids)

    buttons    = _get("buttons_count")
    avg_steps  = _get("avg_daily_steps")
    outreach   = _get("monitoring_outreach_count")
    case_cnt   = _get("case_count")
    er_disp    = _get("er_dispatch_count")
    assist     = _get("assist_count")
    fall       = _get("fall_count")
    email_open = _get("email_open_count")
    email_clk  = _get("email_click_count")

    # Portal signal (snapshot)
    if portal_df is not None:
        acc_map = (
            df_full_up_to_label[["account_number", "account_id"]]
            .drop_duplicates("account_number")
            .set_index("account_number")["account_id"]
        )
        p = portal_df.copy()
        p["account_id"] = p["account_number"].map(acc_map)
        p = p.dropna(subset=["account_id"]).set_index("account_id")
        if "portal_days_since_last_login" in p.columns:
            portal_days = pd.to_numeric(p["portal_days_since_last_login"], errors="coerce").reindex(account_ids)
        else:
            portal_days = pd.Series(np.nan, index=account_ids)
    else:
        portal_days = pd.Series(np.nan, index=account_ids)

    # Pillar flags
    device_active  = ((buttons > 0) | (avg_steps > LOW_STEPS_THRESHOLD)).astype(int)
    care_active    = ((outreach > 2) | (case_cnt > 0)).astype(int)
    email_active   = ((email_open > 0) | (email_clk > 0)).astype(int)
    portal_active  = (portal_days <= 180).fillna(0).astype(int)
    emergency_flag = ((er_disp > 0) | (assist > 0) | (fall > 0)).astype(int)

    active_pillar_count = device_active + care_active + email_active + portal_active

    # Hard disengagement override checks (computed relative to label month)
    df_6m_label  = _get_window_df(df_full_up_to_label, label_ts, 6)
    df_12m_label = _get_window_df(df_full_up_to_label, label_ts, 12)

    btn_6m = (
        _col_sum(df_6m_label, "buttons_count").reindex(account_ids).fillna(0)
        if not df_6m_label.empty else pd.Series(0.0, index=account_ids)
    )
    email_12m = (
        _col_sum(df_12m_label, "email_open_count").reindex(account_ids).fillna(0)
        if not df_12m_label.empty else pd.Series(0.0, index=account_ids)
    )

    consec = _consecutive_inactive_months(df_full_up_to_label, label_ts).reindex(account_ids).fillna(0)

    hard_disengaged = (
        (consec >= 3)
        | ((portal_days > 365) & (email_12m == 0) & (btn_6m == 0))
    )

    # Assign tiers (ascending so higher tiers overwrite lower)
    tier = pd.Series(0, index=account_ids, dtype=np.int8)

    t1 = (
        ((active_pillar_count == 2) & (care_active == 0))
        | ((active_pillar_count == 1) & (emergency_flag == 0))
        | ((active_pillar_count == 0) & (emergency_flag == 1))
    )
    tier[t1.values] = 1

    t2 = (
        ((active_pillar_count >= 3) & (emergency_flag == 1))
        | ((active_pillar_count == 2) & (care_active == 1))
        | ((active_pillar_count == 2) & (device_active == 1) & (portal_active == 1))
    )
    tier[t2.values] = 2

    t3 = (
        (active_pillar_count >= 3)
        & (care_active == 1)
        & ((device_active == 1) | (portal_active == 1))
        & (emergency_flag == 0)
    )
    tier[t3.values] = 3

    tier[hard_disengaged.values] = 0

    # Only expose tiers for accounts present in the label month
    in_label = label_agg.index
    mask_in_label = pd.Series(account_ids).isin(in_label).values
    result_vals = tier.astype(float).values.copy()
    result_vals[~mask_in_label] = np.nan

    return pd.Series(result_vals, dtype=float)


# ── Main feature builder ───────────────────────────────────────────────────────

def _build_features_for_anchor(
    df: pd.DataFrame,
    portal_df: Optional[pd.DataFrame],
    anchor_ts: pd.Timestamp,
    include_targets: bool,
) -> pd.DataFrame:
    """Build all 7-pillar engineered features for a single anchor month."""

    anchor_str = anchor_ts.strftime("%Y-%m")

    df_6m   = _get_window_df(df, anchor_ts, 6)
    df_12m  = _get_window_df(df, anchor_ts, 12)
    df_full = df[df["obs_month_dt"] <= anchor_ts].copy()

    # Base accounts — latest age per account as of anchor
    out = (
        df_full.sort_values("obs_month_dt")
        .groupby("account_id", as_index=False)
        .agg(account_number=("account_number", "last"), age=("age", "last"))
    )
    out["anchor_month"] = anchor_str
    out = out[["account_number", "account_id", "age", "anchor_month"]].copy()

    account_ids = out["account_id"].values

    def _add(name: str, series: pd.Series, fill=np.nan, dtype=None) -> None:
        aligned = series.reindex(account_ids)
        aligned.index = out.index
        if fill is not np.nan:
            aligned = aligned.fillna(fill)
        out[name] = aligned
        if dtype is not None:
            out[name] = out[name].astype(dtype)

    # ── Pillar 1: Device Engagement ───────────────────────────────────────────

    btn6   = _col_sum(df_6m,  "buttons_count")
    btn12  = _col_sum(df_12m, "buttons_count")
    steps6 = _col_mean(df_6m,  "avg_daily_steps")
    steps12= _col_mean(df_12m, "avg_daily_steps")

    _add("buttons_count_6m",    btn6)
    _add("buttons_count_12m",   btn12)
    _add("buttons_trend",       btn6 / (btn12 + EPS))
    _add("avg_daily_steps_6m",  steps6)
    _add("avg_daily_steps_12m", steps12)
    _add("steps_trend",         steps6 / (steps12 + EPS))

    if "avg_daily_steps" in df_6m.columns:
        _sa_flag = pd.to_numeric(df_6m["avg_daily_steps"], errors="coerce") > LOW_STEPS_THRESHOLD
        steps_active = (
            df_6m.assign(_sa=_sa_flag).groupby("account_id")["_sa"].any().astype(np.int8)
        )
    else:
        steps_active = pd.Series(0, index=pd.Index(account_ids), dtype=np.int8)
    _add("steps_active_flag_6m", steps_active, fill=0, dtype=np.int8)

    # ── Pillar 2: Care / System Interaction ───────────────────────────────────

    out6m  = _col_sum(df_6m,  "monitoring_outreach_count")
    out12m = _col_sum(df_12m, "monitoring_outreach_count")
    case6m = _col_sum(df_6m,  "case_count")
    case12m= _col_sum(df_12m, "case_count")

    _add("monitoring_outreach_6m",    out6m)
    _add("monitoring_outreach_12m",   out12m)
    _add("case_count_6m",             case6m)
    _add("case_count_12m",            case12m)
    _add("outreach_trend",            out6m / (out12m + EPS))
    _add("total_care_interactions_6m", out6m.add(case6m, fill_value=0))

    # ── Pillar 3: Emergency / Risk Events ────────────────────────────────────

    _add("had_emergency_event_6m",
         _any_positive(df_6m, "er_dispatch_count", "fall_count"), fill=0, dtype=np.int8)
    _add("had_emergency_event_12m",
         _any_positive(df_12m, "er_dispatch_count", "fall_count"), fill=0, dtype=np.int8)
    _add("assist_event_flag_6m",
         _any_positive(df_6m, "assist_count"), fill=0, dtype=np.int8)
    _add("help_sent_flag_6m",
         _any_positive(df_6m, "help_sent_count"), fill=0, dtype=np.int8)

    er6    = _col_sum(df_6m, "er_dispatch_count").fillna(0)
    asst6  = _col_sum(df_6m, "assist_count").fillna(0)
    fall6  = _col_sum(df_6m, "fall_count").fillna(0)
    canc6  = _col_sum(df_6m, "dispatch_cancelled_count").fillna(0)

    _add("emergency_count_6m", er6 + asst6 + fall6, fill=0)
    dispatch_rate = (1 - canc6 / (er6 + EPS)).where(er6 > 0, other=np.nan)
    _add("dispatch_completion_rate_6m", dispatch_rate)

    # ── Pillar 4: Communication / Email Engagement ────────────────────────────

    def _email_opened_flag(wdf: pd.DataFrame) -> pd.Series:
        if "email_open_count" not in wdf.columns:
            return pd.Series(0, index=wdf["account_id"].unique(), dtype=np.int8)
        _flag = pd.to_numeric(wdf["email_open_count"], errors="coerce") > 0
        return wdf.assign(_flag=_flag).groupby("account_id")["_flag"].any().astype(np.int8)

    _add("email_active_flag_6m",  _email_opened_flag(df_6m),  fill=0, dtype=np.int8)
    _add("email_active_flag_12m", _email_opened_flag(df_12m), fill=0, dtype=np.int8)

    _add("email_open_rate_avg_6m",  _col_mean(df_6m, "email_open_rate_pct"))
    _add("email_click_rate_avg_6m", _col_mean(df_6m, "email_click_rate_pct"))

    opens6  = _col_sum(df_6m, "email_open_count").fillna(0)
    clicks6 = _col_sum(df_6m, "email_click_count").fillna(0)
    _add("email_click_to_open_ratio_6m", (clicks6 / (opens6 + EPS)).where(opens6 > 0, other=np.nan))

    def _email_engagement_score(wdf: pd.DataFrame) -> pd.Series:
        accs = wdf["account_id"].unique()
        score = pd.Series(0, index=accs, dtype=np.int8)

        def _any_pos(col: str) -> pd.Series:
            if col not in wdf.columns:
                return pd.Series(0, index=accs, dtype=np.int8)
            _f = pd.to_numeric(wdf[col], errors="coerce") > 0
            return (
                wdf.assign(_f=_f).groupby("account_id")["_f"]
                .any().astype(np.int8).reindex(accs, fill_value=0)
            )

        delivered = _any_pos("email_delivered_count")
        opened    = _any_pos("email_open_count")
        clicked   = _any_pos("email_click_count")

        score = score.where(delivered == 0, np.int8(1))
        score = score.where(opened    == 0, np.int8(2))
        score = score.where(clicked   == 0, np.int8(3))
        return score.astype(np.int8)

    _add("email_engagement_score_6m", _email_engagement_score(df_6m), fill=0, dtype=np.int8)

    if "email_last_click_month" in df_full.columns:
        last_click = df_full.groupby("account_id")["email_last_click_month"].last()
        _add("months_since_last_email_click", _months_between_date_col(last_click, anchor_ts))
    else:
        out["months_since_last_email_click"] = np.nan

    if "email_last_open_month" in df_full.columns:
        last_open = df_full.groupby("account_id")["email_last_open_month"].last()
        _add("months_since_last_email_open", _months_between_date_col(last_open, anchor_ts))
    else:
        out["months_since_last_email_open"] = np.nan

    if "email_open_count" in df_12m.columns:
        _eo_pos = pd.to_numeric(df_12m["email_open_count"], errors="coerce") > 0
        ever_opened = df_12m.assign(_eo=_eo_pos).groupby("account_id")["_eo"].any()
        never_engaged = (~ever_opened).astype(np.int8)
    else:
        never_engaged = pd.Series(1, index=pd.Index(account_ids), dtype=np.int8)
    _add("email_never_engaged_flag", never_engaged, fill=1, dtype=np.int8)

    # ── Pillar 5: Portal Engagement ───────────────────────────────────────────

    if portal_df is not None:
        acc_map = (
            df_full[["account_number", "account_id"]]
            .drop_duplicates("account_number")
            .set_index("account_number")["account_id"]
        )
        p = portal_df.copy()
        p["account_id"] = p["account_number"].map(acc_map)
        p = p.dropna(subset=["account_id"]).set_index("account_id")

        if "has_active_portal_flag" in p.columns:
            _add("has_active_portal_flag",
                 pd.to_numeric(p["has_active_portal_flag"], errors="coerce").astype(np.int8),
                 fill=0, dtype=np.int8)
        else:
            out["has_active_portal_flag"] = np.int8(0)

        if "portal_days_since_last_login" in p.columns:
            pdays = pd.to_numeric(p["portal_days_since_last_login"], errors="coerce")
            _add("portal_days_since_last_login", pdays)

            def _recency_bucket(d: float) -> int:
                if pd.isna(d): return 0
                if d < 7:   return 5
                if d < 30:  return 4
                if d < 90:  return 3
                if d < 180: return 2
                if d < 365: return 1
                return 0

            _add("portal_recency_score",   pdays.map(_recency_bucket).astype(np.int8), fill=0, dtype=np.int8)
            _add("portal_active_6m_flag",  (pdays <= 180).astype(np.int8), fill=0, dtype=np.int8)
            _add("portal_active_30d_flag", (pdays <= 30).astype(np.int8),  fill=0, dtype=np.int8)
        else:
            out["portal_days_since_last_login"] = np.nan
            out["portal_recency_score"]   = np.int8(0)
            out["portal_active_6m_flag"]  = np.int8(0)
            out["portal_active_30d_flag"] = np.int8(0)
    else:
        out["has_active_portal_flag"]      = np.int8(0)
        out["portal_days_since_last_login"] = np.nan
        out["portal_recency_score"]         = np.int8(0)
        out["portal_active_6m_flag"]        = np.int8(0)
        out["portal_active_30d_flag"]       = np.int8(0)

    # ── Pillar 6: Recency & Consistency (Cross-Pillar) ────────────────────────

    activity_cols_present = [c for c in ACTIVITY_COLS if c in df.columns]

    def _active_month_count(wdf: pd.DataFrame) -> pd.Series:
        if not activity_cols_present:
            return pd.Series(0, index=wdf["account_id"].unique(), dtype=int)
        _act = wdf[activity_cols_present].gt(0).any(axis=1)
        return (
            wdf.assign(_act=_act)
            .groupby(["account_id", "obs_month_dt"])["_act"]
            .max()
            .groupby(level="account_id")
            .sum()
        )

    active6m  = _active_month_count(df_6m)
    active12m = _active_month_count(df_12m)

    _add("active_months_6m",       active6m,          fill=0)
    _add("active_months_12m",      active12m,          fill=0)
    _add("active_months_ratio_6m", active6m / 6.0,     fill=0.0)

    consec_inactive = _consecutive_inactive_months(df_full, anchor_ts)
    _add("consecutive_inactive_months", consec_inactive, fill=0)

    msde = _months_since_device_event(df_full, anchor_ts)
    _add("months_since_any_device_event", msde)

    tenure = df_full.groupby("account_id")["obs_month_dt"].nunique()
    _add("tenure_months", tenure, fill=0)

    # ── Pillar 7: Member Profile (Static Context) ─────────────────────────────

    prod_feats = _build_product_features(df_full, anchor_ts).set_index("account_id")
    for col in ["product_name_encoded", "has_product_name_flag", "product_engagement_profile"]:
        _add(col, prod_feats[col], fill=0)

    # ── Target columns ────────────────────────────────────────────────────────

    if include_targets:
        for offset, col_name in [
            (1, "target_engagement_t30d"),
            (2, "target_engagement_t60d"),
            (3, "target_engagement_t90d"),
        ]:
            label_ts = (anchor_ts + pd.DateOffset(months=offset)).replace(day=1)
            df_label = df[df["obs_month_dt"] == label_ts]
            df_up_to_label = df[df["obs_month_dt"] <= label_ts]

            if df_label.empty:
                out[col_name] = np.nan
                print(f"  {col_name}: no data for label month {label_ts.strftime('%Y-%m')} → NaN")
            else:
                tier_vals = _compute_engagement_tier(
                    df_label=df_label,
                    portal_df=portal_df,
                    df_full_up_to_label=df_up_to_label,
                    label_ts=label_ts,
                    account_ids=account_ids,
                )
                out[col_name] = tier_vals.values
                non_null = int(tier_vals.notna().sum())
                dist = tier_vals.value_counts().sort_index().to_dict()
                print(
                    f"  {col_name}: label={label_ts.strftime('%Y-%m')}, "
                    f"non_null={non_null}, dist={dist}"
                )

    print(f"  anchor={anchor_str}, rows={len(out)}, features={len(out.columns)}")
    return out


def build_features(
    df: pd.DataFrame,
    portal_df: Optional[pd.DataFrame],
    window_months: int,
    include_targets: bool = True,
    anchor_month: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build engagement index features.

    For training (anchor_month=None): slides over all valid months >= 2025-11,
    producing one row per (account, anchor_month).
    For scoring (anchor_month specified): uses a single anchor month.
    """
    all_months = sorted(df["obs_month_dt"].unique())

    if anchor_month:
        raw = anchor_month.replace("_", "-")
        anchor_ts = pd.to_datetime(raw + "-01", errors="coerce")
        if pd.isna(anchor_ts):
            raise ValueError(f"Invalid --anchor_month '{anchor_month}'. Use YYYY-MM or MM_YYYY.")
        if anchor_ts < MIN_ANCHOR_MONTH:
            raise ValueError(
                f"Anchor month {anchor_ts.strftime('%Y-%m')} must be >= "
                f"{MIN_ANCHOR_MONTH.strftime('%Y-%m')}."
            )
        anchor_months = [anchor_ts]
    else:
        anchor_months = [m for m in all_months if m >= MIN_ANCHOR_MONTH]
        if not anchor_months:
            available = [m.strftime("%Y-%m") for m in all_months]
            raise ValueError(
                f"No months >= {MIN_ANCHOR_MONTH.strftime('%Y-%m')} found. "
                f"Available months: {available}"
            )

    frames = []
    for a_ts in anchor_months:
        print(f"Processing anchor month: {a_ts.strftime('%Y-%m')}")
        frame = _build_features_for_anchor(
            df=df,
            portal_df=portal_df,
            anchor_ts=a_ts,
            include_targets=include_targets,
        )
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)

    dup_key = ["account_id", "anchor_month"]
    dup_count = int(out[dup_key].duplicated(keep=False).sum())
    if dup_count > 0:
        print(f"Duplicate key check: found {dup_count} duplicates on {dup_key} — keeping first")
        out = out.drop_duplicates(subset=dup_key, keep="first")
    else:
        print(f"Duplicate key check: no duplicates on {dup_key}")

    feature_cols = [c for c in out.columns if c not in STATIC_OUTPUT_COLUMNS]
    out = out[STATIC_OUTPUT_COLUMNS + feature_cols]

    print(f"Total output rows: {len(out)}, total columns: {len(out.columns)}")
    return out


# ── Build and save ─────────────────────────────────────────────────────────────

def build_and_save_window(
    df: pd.DataFrame,
    portal_df: Optional[pd.DataFrame],
    window_months: int,
    output_dir: str,
    include_targets: bool = True,
    anchor_month: Optional[str] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    features = build_features(
        df=df,
        portal_df=portal_df,
        window_months=window_months,
        include_targets=include_targets,
        anchor_month=anchor_month,
    )
    suffix = "_with_target" if include_targets else ""
    file_name = f"engagement_index_feature{suffix}.parquet"
    output_path = os.path.join(output_dir, file_name)
    features.to_parquet(output_path, index=False)
    print(f"[window={window_months}] output file: {output_path}")
    return output_path


def build_and_save_window_variant(
    df: pd.DataFrame,
    portal_df: Optional[pd.DataFrame],
    window_months: int,
    output_dir: str,
    include_targets: bool,
    file_suffix: str = "",
    anchor_month: Optional[str] = None,
) -> str:
    """Save a specific variant (scoring or training) for a given window."""
    os.makedirs(output_dir, exist_ok=True)
    features = build_features(
        df=df,
        portal_df=portal_df,
        window_months=window_months,
        include_targets=include_targets,
        anchor_month=anchor_month,
    )
    file_name = f"engagement_index_feature{file_suffix}.parquet"
    output_path = os.path.join(output_dir, file_name)
    features.to_parquet(output_path, index=False)
    variant = "training" if include_targets else "scoring"
    print(f"[window={window_months}] {variant} output file: {output_path}")
    return output_path


def build_and_save_per_target_splits(
    df: pd.DataFrame,
    portal_df: Optional[pd.DataFrame],
    window_months: int,
    output_dir: str,
    anchor_month: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build features once (with all 3 target columns), then write three horizon-specific
    training files — each containing all feature columns plus only its own target:

      engagement_index_feature_with_target_30d.parquet  → target_engagement_t30d
      engagement_index_feature_with_target_60d.parquet  → target_engagement_t60d
      engagement_index_feature_with_target_90d.parquet  → target_engagement_t90d

    Keeping one target per file prevents data leakage: each training script sees only
    the target it is trained on and cannot inadvertently use future-horizon labels as
    features.

    Returns a dict mapping horizon label ('30d'/'60d'/'90d') -> saved output path.
    """
    os.makedirs(output_dir, exist_ok=True)

    features = build_features(
        df=df,
        portal_df=portal_df,
        window_months=window_months,
        include_targets=True,
        anchor_month=anchor_month,
    )

    all_target_cols = {"target_engagement_t30d", "target_engagement_t60d", "target_engagement_t90d"}
    non_target_cols = [c for c in features.columns if c not in all_target_cols]

    saved: Dict[str, str] = {}
    for horizon, target_col in [
        ("30d", "target_engagement_t30d"),
        ("60d", "target_engagement_t60d"),
        ("90d", "target_engagement_t90d"),
    ]:
        keep_cols = non_target_cols + ([target_col] if target_col in features.columns else [])
        file_name = f"engagement_index_feature_with_target_{horizon}.parquet"
        output_path = os.path.join(output_dir, file_name)
        features[keep_cols].to_parquet(output_path, index=False)
        n_labeled = int(features[target_col].notna().sum()) if target_col in features.columns else 0
        print(
            f"[window={window_months}] {horizon} training file: {output_path} "
            f"(labeled rows: {n_labeled:,})"
        )
        saved[horizon] = output_path

    return saved


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature engineering for the Engagement Index AI model."
    )
    parser.add_argument(
        "--input_file",
        default="202411_to_202603_dtc_engagement_index.parquet",
        help=(
            "Path to long-format base feature file (.parquet/.xlsx/.xls/.csv). "
            "Use 'auto' to detect latest engagement_index_feature_MM_YYYY file."
        ),
    )
    parser.add_argument(
        "--portal_file",
        default="portal_engagement_snapshot.csv",
        help="Path to portal engagement snapshot file.",
    )
    parser.add_argument(
        "--input_dir",
        default=".",
        help="Directory used for --input_file auto-detection.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to write output parquet files.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[6],
        help="Feature window sizes in months (default: 6).",
    )
    parser.add_argument(
        "--anchor_month",
        default=None,
        help=(
            "Anchor month in YYYY-MM or MM_YYYY format. "
            "Must be >= 2025-11. "
            "If omitted, slides over all valid months >= 2025-11."
        ),
    )
    parser.add_argument(
        "--include_target",
        action="store_true",
        help="Include target_engagement_t30d, t60d, t90d target columns.",
    )
    parser.add_argument(
        "--save_both_versions",
        action="store_true",
        default=True,
        help="Generate both scoring (no target) and training (with target) versions per window. Default: True.",
    )
    parser.add_argument(
        "--train_output_dir",
        default=None,
        help="Training output directory (used with --save_both_versions). Defaults to <output_dir>/train.",
    )
    parser.add_argument(
        "--score_output_dir",
        default=None,
        help="Scoring output directory (used with --save_both_versions). Defaults to <output_dir>/score.",
    )

    args = parser.parse_args()

    resolved_input = args.input_file
    if args.input_file.strip().lower() == "auto":
        resolved_input = detect_latest_engagement_file(args.input_dir)

    df = load_input(resolved_input)

    portal_df: Optional[pd.DataFrame] = None
    if args.portal_file:
        try:
            portal_df = load_portal(args.portal_file)
        except FileNotFoundError:
            print(
                f"Warning: portal file not found at '{args.portal_file}' "
                "— portal features will default to 0/NaN."
            )

    unique_windows = sorted(set(args.windows), reverse=True)
    print(f"Input rows: {len(df)}")
    print(f"Requested windows: {unique_windows}")
    print(f"Include targets: {args.include_target}")
    print(f"Save both versions: {args.save_both_versions}")
    print(f"Anchor month: {args.anchor_month or 'all valid months >= 2025-11'}")

    for w in unique_windows:
        if args.save_both_versions:
            score_dir = args.score_output_dir or os.path.join(args.output_dir, "score")
            train_dir = args.train_output_dir or os.path.join(args.output_dir, "train")
            # Scoring file: no target columns (used by scoring scripts)
            build_and_save_window_variant(
                df=df, portal_df=portal_df, window_months=w,
                output_dir=score_dir, include_targets=False, file_suffix="",
                anchor_month=args.anchor_month,
            )
            # Training files: one per horizon, each with only its own target column
            # (prevents cross-horizon target leakage into features)
            build_and_save_per_target_splits(
                df=df, portal_df=portal_df, window_months=w,
                output_dir=train_dir, anchor_month=args.anchor_month,
            )
        else:
            build_and_save_window(
                df=df, portal_df=portal_df, window_months=w,
                output_dir=args.output_dir,
                include_targets=args.include_target,
                anchor_month=args.anchor_month,
            )


if __name__ == "__main__":
    main()
