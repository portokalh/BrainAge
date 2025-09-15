#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HABS — DeltaBAG Analysis: FULL SCRIPT (cBAG_Analysis3.py)

- Predictors analyzed:
    * BAG_AB, cBAG_AB
    * DeltaBAGs derived here:
        - BAG_BrainAgeR = 06_BrainAgeR_PredictedAge − Age
        - cBAG_BrainAgeR = residual(BAG_BrainAgeR ~ Age)
        - BAG_DeepBrainNet = 06_DeepBrainNet_PredictedAge − Age
        - cBAG_DeepBrainNet = residual(BAG_DeepBrainNet ~ Age)

- Outcomes analyzed:
    * IMH_Diabetes, CDX_Diabetes (binary)
    * CDX_Cog (binary 0 = normal vs 1/2 = impaired)
    * IMH_Alzheimers (binary; alias IMH_Alzheimer supported)
    * APOE4_Positivity (binary)

- For each predictor×outcome (binary):
    * AUC with 95% bootstrap CI (n=2000)
    * Automatic orientation (computes AUC for x and -x, chooses higher; flag recorded)
    * Odds Ratio per +1 SD (from logistic regression)
    * Cohen's d and Cliff's delta
    * ROC curve PNG and violin+dots group plot PNG with p-values (MWU + Welch t)

- Additional outputs:
    * Histogram PNGs for each predictor
    * delta_bags.csv containing derived BAG_BrainAgeR / cBAG_BrainAgeR / BAG_DeepBrainNet / cBAG_DeepBrainNet (+ID if found)
    * summary_stats.csv with all stats
    * processing_log.txt with coverage details

Usage:
    python cBAG_Analysis3.py \
      --csv "/Users/alex/AlexBadea_MyGrants/BAG_R01_100325/code/data/HABS_metadata_with_PredictedAgeAB_BAG_short.csv" \
      --outdir "/Users/alex/AlexBadea_MyGrants/BAG_R01_100325/code/results_HABS"
"""
import argparse
import os
import sys
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu, ttest_ind

# -----------------------
# Defaults
# -----------------------
DEFAULT_CSV = "/Users/alex/AlexBadea_MyGrants/BAG_R01_100325/code/data/HABS_metadata_with_PredictedAgeAB_BAG_short.csv"
DEFAULT_OUTDIR = "/Users/alex/AlexBadea_MyGrants/BAG_R01_100325/code/results_HABS"

# We will ensure the derived predictors exist before analysis
PREDICTORS = [
    "BAG_AB", "cBAG_AB",
    "BAG_BrainAgeR", "cBAG_BrainAgeR",
    "BAG_DeepBrainNet", "cBAG_DeepBrainNet",
]

OUTCOMES = [
    "IMH_Diabetes",
    "CDX_Diabetes",
    "CDX_Cog",            # forced 0 vs (1,2)
    "IMH_Alzheimers",
    "IMH_Alzheimer",      # alias
    "APOE4_Positivity",
]

N_BOOT = 2000
RANDOM_SEED = 42

# -----------------------
# Utils
# -----------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log(msg: str, log_path: str):
    """Safe logger that always writes a newline and mirrors to stdout."""
    try:
        with open(log_path, "a") as f:
            f.write(str(msg).rstrip() + "\n")
    except Exception as e:
        print(f"[LOG-ERROR] {e}: {msg}")
    print(str(msg))


def find_col(df: pd.DataFrame, target: str) -> str:
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    key = target.lower().replace(" ", "_")
    return cols.get(key, "")


def find_age_column(df: pd.DataFrame) -> str:
    candidates = [
        "Age", "ChronologicalAge", "Chronological_Age", "AgeYears",
        "Age_Years", "Age_at_scan", "CHRONO_AGE", "AgeAtScan",
        "AGE", "age"
    ]
    for cand in candidates:
        col = find_col(df, cand)
        if col:
            return col
    return ""


def find_id_column(df: pd.DataFrame) -> str:
    candidates = [
        "SubjectID", "Subject_ID", "ID", "Id", "RID", "PTID",
        "Participant", "ParticipantID", "participant_id", "eid", "record_id"
    ]
    for cand in candidates:
        col = find_col(df, cand)
        if col:
            return col
    return ""


def describe_uniques(name: str, series: pd.Series, log_path: str, prefix: str = "global"):
    try:
        counts = series.value_counts(dropna=False)
        as_dict = {str(k): int(v) for k, v in counts.items()}
        log(f"[UNIQ-{prefix}] {name}: {as_dict}", log_path)
    except Exception as e:
        log(f"[UNIQ-{prefix}] {name}: <failed to summarize> ({e})", log_path)


def coerce_string_binary(s: pd.Series, name: str = "") -> pd.Series:
    """Map common string-coded binaries to 0/1. Leaves numeric as-is; non-mapped strings -> NaN.
    Special handling: CDX_Cog => numeric kept as-is (we do 0 vs 1+2 later).
    """
    if name and name.lower().replace(" ", "_") == "cdx_cog":
        return pd.to_numeric(s, errors="coerce")
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    mapping_pos = {"yes", "y", "true", "t", "pos", "+", "case", "diabetes", "alzheimers", "ad", "impairment", "mci", "1", "positive"}
    mapping_neg = {"no", "n", "false", "f", "neg", "-", "control", "none", "normal", "0", "negative"}
    def _map(v):
        if pd.isna(v):
            return np.nan
        sv = str(v).strip().lower()
        if sv in mapping_pos:
            return 1.0
        if sv in mapping_neg:
            return 0.0
        try:
            return float(sv)
        except Exception:
            return np.nan
    return s.map(_map)


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_sd


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    greater = 0
    lesser = 0
    for xi in x:
        greater += np.sum(xi > y)
        lesser += np.sum(xi < y)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    return (greater - lesser) / (n1 * n2)


def bootstrap_ci(values: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.array([v for v in values if not np.isnan(v)])
    if arr.size == 0:
        return (np.nan, np.nan)
    lo = np.percentile(arr, 100 * (alpha / 2))
    hi = np.percentile(arr, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


def standardize(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    mean = np.nanmean(x)
    return (x - mean) / sd

# --- Metrics helpers (annotation-free, single-line headers) ---
def binary_auc_with_ci(y, score, n_boot, seed):
    auc_base = roc_auc_score(y, score)
    rng = np.random.RandomState(seed)
    boot_vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        sb = score[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            boot_vals.append(roc_auc_score(yb, sb))
        except Exception:
            pass
    lo = float(np.percentile(boot_vals, 2.5)) if boot_vals else np.nan
    hi = float(np.percentile(boot_vals, 97.5)) if boot_vals else np.nan
    return float(auc_base), (lo, hi)

def binary_or_per_sd_with_ci(y, x_std, n_boot, seed):
    clf = LogisticRegression(solver="liblinear")
    clf.fit(x_std.reshape(-1, 1), y)
    or_base = float(np.exp(clf.coef_[0][0]))

    rng = np.random.RandomState(seed)
    boot_vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        xb = x_std[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            clf_b = LogisticRegression(solver="liblinear")
            clf_b.fit(xb.reshape(-1, 1), yb)
            boot_vals.append(np.exp(clf_b.coef_[0][0]))
        except Exception:
            pass
    lo = float(np.percentile(boot_vals, 2.5)) if boot_vals else np.nan
    hi = float(np.percentile(boot_vals, 97.5)) if boot_vals else np.nan
    return or_base, (lo, hi)

def multiclass_macro_auc_with_ci(y_raw, x, n_boot, seed):
    le = LabelEncoder()
    y = le.fit_transform(np.asarray(y_raw).astype(str))
    classes = np.unique(y)

    def macro_auc(y_int, x_feat):
        clf = LogisticRegression(multi_class="ovr", solver="liblinear")
        clf.fit(x_feat.reshape(-1, 1), y_int)
        probs = clf.predict_proba(x_feat)
        aucs = []
        for k in classes:
            y_bin = (y_int == k).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            aucs.append(roc_auc_score(y_bin, probs[:, k]))
        return float(np.mean(aucs)) if len(aucs) else np.nan

    base_auc = macro_auc(y, x)

    rng = np.random.RandomState(seed)
    boot_vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        xb = x[idx]
        try:
            boot_vals.append(macro_auc(yb, xb))
        except Exception:
            pass
    lo = float(np.percentile(boot_vals, 2.5)) if boot_vals else np.nan
    hi = float(np.percentile(boot_vals, 97.5)) if boot_vals else np.nan
    return base_auc, (lo, hi)

def choose_oriented_binary_metrics(y_bin, x, n_boot, seed):
    """Compute AUC for x and -x; return the orientation with higher AUC.
       Returns: auc, (lo, hi), flipped(bool), score_for_plot
    """
    auc_a, (lo_a, hi_a) = binary_auc_with_ci(y_bin, x, n_boot, seed)
    auc_b, (lo_b, hi_b) = binary_auc_with_ci(y_bin, -x, n_boot, seed)
    if np.isnan(auc_a) and not np.isnan(auc_b):
        return auc_b, (lo_b, hi_b), True, -x
    if np.isnan(auc_b) and not np.isnan(auc_a):
        return auc_a, (lo_a, hi_a), False, x
    if auc_b > auc_a:
        return auc_b, (lo_b, hi_b), True, -x
    return auc_a, (lo_a, hi_a), False, x

# Positive label preferences

def positive_label_for(outcome: str, uniques=None):
    key = outcome.lower().replace(" ", "_")
    if key in {"imh_diabetes", "cdx_diabetes", "imh_alzheimers", "apoe4_positivity"}:
        return 1
    return None

# Plotters

def plot_roc_binary(y, score, title, out_png, auc_val, lo, hi):
    fpr, tpr, _ = roc_curve(y, score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.3f} [95% CI {lo:.3f}-{hi:.3f}])")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_groups_box_violin(values, groups, title, out_png):
    vals = pd.Series(values).astype(float)
    cats = pd.Series(groups)
    m = ~vals.isna() & ~cats.isna()
    vals = vals[m]
    cats = cats[m]
    labels = list(pd.unique(cats))
    data = [vals[cats == c].values for c in labels]

    plt.figure()
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.boxplot(data, positions=np.arange(1, len(labels) + 1))
    for i, arr in enumerate(data, start=1):
        if len(arr) == 0:
            continue
        xj = i + (np.random.rand(len(arr)) - 0.5) * 0.2
        plt.scatter(xj, arr, alpha=0.6, s=12)
    plt.xticks(np.arange(1, len(labels) + 1), [str(l) for l in labels], rotation=15)
    plt.ylabel("Value")
    plt.title(title)

    if len(labels) == 2 and len(data[0]) > 1 and len(data[1]) > 1:
        try:
            u_p = mannwhitneyu(data[0], data[1], alternative="two-sided").pvalue
        except Exception:
            u_p = np.nan
        try:
            t_p = ttest_ind(data[0], data[1], equal_var=False).pvalue
        except Exception:
            t_p = np.nan
        y_max = np.nanmax(vals)
        y_min = np.nanmin(vals)
        h = (y_max - y_min) * 0.08 if np.isfinite(y_max) and np.isfinite(y_min) else 1.0
        y = y_max + h
        x1, x2 = 1, 2
        plt.plot([x1, x1, x2, x2], [y, y + h*0.2, y + h*0.2, y], linewidth=1)
        plt.text((x1 + x2) / 2, y + h*0.25, f"Mann-Whitney p={u_p:.3g}, t-test p={t_p:.3g}", ha="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -----------------------
# Derivation of deltaBAGs
# -----------------------

def compute_bag_and_cbag(df: pd.DataFrame, pred_name: str, age_col: str, bag_name: str, cbag_name: str, log_path: str):
    pred_col = find_col(df, pred_name)
    if pred_col == "":
        log(f"[WARN] Predicted age column '{pred_name}' not found; cannot compute {bag_name}/{cbag_name}.", log_path)
        return
    if age_col == "":
        log(f"[WARN] No chronological age column found; cannot compute {bag_name}/{cbag_name}.", log_path)
        return
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    age = pd.to_numeric(df[age_col], errors="coerce")
    bag = pred - age
    df[bag_name] = bag
    m = (~bag.isna()) & (~age.isna())
    if m.sum() >= 5:
        X = age[m].to_numpy().reshape(-1,1)
        y = bag[m].to_numpy()
        try:
            lr = LinearRegression()
            lr.fit(X, y)
            resid = y - lr.predict(X)
            cbag = pd.Series(np.nan, index=df.index)
            idxs = np.where(m)[0]
            cbag.iloc[idxs] = resid
            df[cbag_name] = cbag.values
            log(f"[DERIVED] Computed {bag_name} and {cbag_name}: non-NaN n={int(m.sum())}", log_path)
        except Exception as e:
            log(f"[WARN] Failed to residualize {bag_name} on Age: {e}", log_path)
            df[cbag_name] = np.nan
    else:
        log(f"[WARN] Not enough rows to residualize {bag_name} on Age (n={int(m.sum())}).", log_path)
        df[cbag_name] = np.nan

# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to CSV metadata")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n_boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    roc_dir = os.path.join(args.outdir, "roc_curves")
    grp_dir = os.path.join(args.outdir, "group_plots")
    ensure_dir(roc_dir)
    ensure_dir(grp_dir)
    log_path = os.path.join(args.outdir, "processing_log.txt")
    if os.path.exists(log_path):
        os.remove(log_path)

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"ERROR: could not read CSV: {e}")
        sys.exit(1)

    rows: List[Dict] = []

    # Derive deltaBAGs for BrainAgeR & DeepBrainNet
    age_col = find_age_column(df)
    if age_col == "":
        log("[WARN] No chronological age column found; derived BAGs will be skipped.", log_path)
    else:
        log(f"[INFO] Using chronological age column: '{age_col}'", log_path)
        compute_bag_and_cbag(df, "06_BrainAgeR_PredictedAge", age_col, "BAG_BrainAgeR", "cBAG_BrainAgeR", log_path)
        compute_bag_and_cbag(df, "06_DeepBrainNet_PredictedAge", age_col, "BAG_DeepBrainNet", "cBAG_DeepBrainNet", log_path)
        # Save deltaBAGs CSV
        id_col = find_id_column(df)
        delta_cols = [c for c in ["BAG_BrainAgeR", "cBAG_BrainAgeR", "BAG_DeepBrainNet", "cBAG_DeepBrainNet"] if c in df.columns]
        if delta_cols:
            cols_to_save = ([id_col] if id_col else []) + delta_cols
            delta_df = df[cols_to_save].copy() if cols_to_save else pd.DataFrame()
            delta_csv = os.path.join(args.outdir, "delta_bags.csv")
            delta_df.to_csv(delta_csv, index=False)
            log(f"[SAVED] deltaBAG CSV -> {delta_csv} (cols: {', '.join(cols_to_save)})", log_path)

    log("=== DATA COVERAGE REPORT (non-NaN when paired with predictor) ===", log_path)

    for pred in PREDICTORS:
        if pred not in df.columns:
            # silently skip if not present (e.g., no Age to derive)
            continue
        x_all = pd.to_numeric(df[pred], errors="coerce").values
        log(f"[UNIQ-global] {pred}: non-NaN={int(np.sum(~np.isnan(x_all)))}, NaN={int(np.sum(np.isnan(x_all)))}", log_path)

        for outcome in OUTCOMES:
            out_col = find_col(df, outcome)
            if out_col == "":
                log(f"[WARN] Outcome '{outcome}' not found; skipping.", log_path)
                continue
            y_ser = coerce_string_binary(df[out_col], name=outcome)
            describe_uniques(outcome, y_ser, log_path, prefix="global")

            m = (~pd.isna(y_ser)) & (~np.isnan(x_all))
            x = x_all[m]
            y = y_ser[m]
            describe_uniques(f"{outcome} | subset({pred})", pd.Series(y), log_path, prefix="subset")
            if len(y) < 3:
                log(f"[WARN] Too few non-NaN rows for '{pred}' vs '{outcome}'; skipping.", log_path)
                continue

            # CDX_Cog forced binary: 0 vs (1,2)
            if outcome.lower().replace(" ", "_") == "cdx_cog":
                mask = pd.Series(y).isin([0,1,2]).to_numpy()
                x_bin = x[mask]
                y_raw = pd.Series(y)[mask].astype(int)
                if len(y_raw) >= 3 and len(np.unique((y_raw > 0).to_numpy())) == 2:
                    y01 = (y_raw > 0).astype(int).to_numpy()
                    neg = int(np.sum(y01 == 0))
                    pos = int(np.sum(y01 == 1))
                    log(f"[COVERAGE] {pred} vs {outcome} (0 vs 1+2): n={len(y01)}, neg={neg}, pos={pos}", log_path)
                    try:
                        auc_base, (auc_lo, auc_hi), flipped, x_for_plot = choose_oriented_binary_metrics(y01, x_bin, args.n_boot, args.seed)
                        x_std = standardize(x_bin)
                        or_base, (or_lo, or_hi) = binary_or_per_sd_with_ci(y01, x_std, args.n_boot, args.seed)
                        g1 = x_bin[y01 == 0]; g2 = x_bin[y01 == 1]
                        d = cohens_d(g2, g1); cd = cliffs_delta(g2, g1)
                        roc_png = os.path.join(roc_dir, f"ROC_{pred}_{outcome}_0vs12.png")
                        plot_roc_binary(y01, x_for_plot, f"ROC: {pred} vs {outcome} (0 vs 1+2){' [flipped]' if flipped else ''}", roc_png, auc_base, auc_lo, auc_hi)
                        grp_png = os.path.join(grp_dir, f"Groups_{pred}_{outcome}_0vs12.png")
                        plot_groups_box_violin(x_bin, y01, f"{pred} by {outcome} (0 vs 1+2)", grp_png)
                        rows.append({
                            "predictor": pred,
                            "outcome": f"{outcome} (0 vs 1+2)",
                            "n": int(len(y01)),
                            "analysis_type": "binary",
                            "positive_class": "1+2",
                            "auc": auc_base,
                            "auc_ci_lo": auc_lo,
                            "auc_ci_hi": auc_hi,
                            "predictor_flipped_for_auc": flipped,
                            "or_per_+1SD": or_base,
                            "or_ci_lo": or_lo,
                            "or_ci_hi": or_hi,
                            "cohens_d": d,
                            "cliffs_delta": cd,
                            "plot_roc": roc_png,
                            "plot_groups": grp_png,
                        })
                    except Exception as e:
                        log(f"[WARN] Failed binary CDX_Cog 0 vs 1+2 analysis for {pred}: {e}", log_path)
                else:
                    log(f"[INFO] CDX_Cog lacks both classes after filtering; skipping 0 vs 1+2.", log_path)
                continue

            # Generic numeric handling (binary only)
            if pd.api.types.is_numeric_dtype(y):
                uniques = np.unique(pd.Series(y).to_numpy())
                if len(uniques) < 2:
                    log(f"[INFO] After NaN filtering with predictor '{pred}', outcome '{outcome}' has a single level; skipping.", log_path)
                    continue
                if len(uniques) == 2:
                    pos_label = positive_label_for(outcome, uniques)
                    if pos_label is None:
                        pos_label = int(np.max(uniques))
                    y_bin = (pd.Series(y).to_numpy() == pos_label).astype(int)
                    neg = int(np.sum(y_bin == 0)); pos = int(np.sum(y_bin == 1))
                    log(f"[COVERAGE] {pred} vs {outcome}: n={len(y_bin)}, neg={neg}, pos={pos}, pos_label={pos_label}", log_path)
                    try:
                        auc_base, (auc_lo, auc_hi), flipped, x_for_plot = choose_oriented_binary_metrics(y_bin, x, args.n_boot, args.seed)
                        x_std = standardize(x)
                        or_base, (or_lo, or_hi) = binary_or_per_sd_with_ci(y_bin, x_std, args.n_boot, args.seed)
                        g1 = x[y_bin == 0]; g2 = x[y_bin == 1]
                        d = cohens_d(g2, g1); cd = cliffs_delta(g2, g1)
                        roc_png = os.path.join(roc_dir, f"ROC_{pred}_{outcome}.png")
                        plot_roc_binary(y_bin, x_for_plot, f"ROC: {pred} vs {outcome} (pos={pos_label}{' flipped' if flipped else ''})", roc_png, auc_base, auc_lo, auc_hi)
                        grp_png = os.path.join(grp_dir, f"Groups_{pred}_{outcome}.png")
                        plot_groups_box_violin(x, y_bin, f"{pred} by {outcome} (pos={pos_label})", grp_png)
                        rows.append({
                            "predictor": pred,
                            "outcome": outcome,
                            "n": int(len(y_bin)),
                            "analysis_type": "binary",
                            "positive_class": pos_label,
                            "auc": auc_base,
                            "auc_ci_lo": auc_lo,
                            "auc_ci_hi": auc_hi,
                            "predictor_flipped_for_auc": flipped,
                            "or_per_+1SD": or_base,
                            "or_ci_lo": or_lo,
                            "or_ci_hi": or_hi,
                            "cohens_d": d,
                            "cliffs_delta": cd,
                            "plot_roc": roc_png,
                            "plot_groups": grp_png,
                        })
                    except Exception as e:
                        log(f"[WARN] Failed binary analysis for {pred} vs {outcome}: {e}", log_path)
                else:
                    log(f"[INFO] Outcome '{outcome}' is numeric with {len(uniques)} unique values; skipping AUC/effect sizes.", log_path)
                continue

            # Categorical (rare for listed outcomes)
            cats = pd.Categorical(pd.Series(y).astype(str))
            non_empty = [c for c in cats.categories if (cats == c).sum() > 0]
            if len(non_empty) < 2:
                log(f"[INFO] After NaN filtering with predictor '{pred}', outcome '{outcome}' has a single level; skipping.", log_path)
                continue
            if len(non_empty) == 2:
                counts = [(cats == non_empty[0]).sum(), (cats == non_empty[1]).sum()]
                pos_label_hint = positive_label_for(outcome)
                if pos_label_hint is not None:
                    # attempt to choose the label that maps to 1-like
                    pos_label = None
                    for lab in non_empty:
                        if str(lab).strip() in {"1", "1.0", "True", "true", "yes", "Yes"}:
                            pos_label = lab
                            break
                    if pos_label is None:
                        pos_label = non_empty[int(np.argmax(counts))]
                else:
                    pos_label = non_empty[int(np.argmax(counts))]
                y_bin = (cats == pos_label).astype(int)
                neg = int(np.sum(y_bin == 0)); pos = int(np.sum(y_bin == 1))
                log(f"[COVERAGE] {pred} vs {outcome}: n={len(y_bin)}, neg={neg}, pos={pos}, pos_label={pos_label}", log_path)
                try:
                    auc_base, (auc_lo, auc_hi), flipped, x_for_plot = choose_oriented_binary_metrics(y_bin, x, args.n_boot, args.seed)
                    x_std = standardize(x)
                    or_base, (or_lo, or_hi) = binary_or_per_sd_with_ci(y_bin, x_std, args.n_boot, args.seed)
                    g1 = x[y_bin == 0]; g2 = x[y_bin == 1]
                    d = cohens_d(g2, g1); cd = cliffs_delta(g2, g1)
                    roc_png = os.path.join(roc_dir, f"ROC_{pred}_{outcome}.png")
                    plot_roc_binary(y_bin, x_for_plot, f"ROC: {pred} vs {outcome} (pos={pos_label}{' flipped' if flipped else ''})", roc_png, auc_base, auc_lo, auc_hi)
                    grp_png = os.path.join(grp_dir, f"Groups_{pred}_{outcome}.png")
                    plot_groups_box_violin(x, y_bin, f"{pred} by {outcome} (pos={pos_label})", grp_png)
                    rows.append({
                        "predictor": pred,
                        "outcome": outcome,
                        "n": int(len(y_bin)),
                        "analysis_type": "binary",
                        "positive_class": str(pos_label),
                        "auc": auc_base,
                        "auc_ci_lo": auc_lo,
                        "auc_ci_hi": auc_hi,
                        "predictor_flipped_for_auc": flipped,
                        "or_per_+1SD": or_base,
                        "or_ci_lo": or_lo,
                        "or_ci_hi": or_hi,
                        "cohens_d": d,
                        "cliffs_delta": cd,
                        "plot_roc": roc_png,
                        "plot_groups": grp_png,
                    })
                except Exception as e:
                    log(f"[WARN] Failed binary-categorical analysis for {pred} vs {outcome}: {e}", log_path)

    # Predictor histograms
    for pred in PREDICTORS:
        if pred in df.columns:
            vals = pd.to_numeric(df[pred], errors='coerce').dropna()
            if len(vals) > 0:
                plt.figure(figsize=(6,4))
                plt.hist(vals, bins=30, edgecolor='black', alpha=0.7)
                plt.xlabel(pred)
                plt.ylabel('Count')
                plt.title(f'Distribution of {pred} (N={len(vals)})')
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, f"Hist_{pred}.png"), dpi=200)
                plt.close()

    # Save summary
    if rows:
        out_csv = os.path.join(args.outdir, "summary_stats.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved classification summary: {out_csv}")
    else:
        print("No classification results computed.")


if __name__ == '__main__':
    main()
