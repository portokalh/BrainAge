#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute AUCs (with 95% CIs), effect sizes, and plots for BAG_AB & cBAG_AB
against selected outcomes, and compute correlations with cognitive measures
(Trails, MMSE, CDR) with scatter plots.

- Binary outcomes: AUC with 95% CI (bootstrap), OR per +1 SD with 95% CI,
  Cohen's d and Cliff's delta, ROC curves, and group distribution plots.
- Multiclass outcome (APOE4_Genotype): one-vs-rest logistic regression macro AUC.
- Continuous outcomes (Trails / MMSE / CDR): Spearman and Pearson r with 95% CIs
  (bootstrap), and scatter plots with a best-fit line.

Saves to the outdir:
  * summary_stats.csv (classification stats)
  * summary_correlations.csv (continuous correlates)
  * roc_curves/*.png
  * group_plots/*.png
  * scatter_plots/*.png
  * processing_log.txt

Usage (paths are pre-filled to Alexandra's folders, but can be overridden):
    python habs_auc_effects.py \
      --csv "/Users/alex/AlexBadea_MyGrants/BAG_R01_100325/metadata/HABS_metadata_with_PredictedAgeAB_BAG_short.csv" \
      --outdir "/Users/alex/AlexBadea_MyGrants/BAG_R01_100325/metadata/HABS_AB_plots"

Author: ChatGPT
"""
import argparse
import os
import sys
import warnings
import re
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# -----------------------
# Defaults (include your paths)
# -----------------------
DEFAULT_CSV = "/Users/alex/AlexBadea_MyGrants/BAG_R01_100325/metadata/HABS_metadata_with_PredictedAgeAB_BAG_short.csv"
DEFAULT_OUTDIR = "/Users/alex/AlexBadea_MyGrants/BAG_R01_100325/code/results_HABS/"

PREDICTORS = ["BAG_AB", "cBAG_AB"]
OUTCOMES = [
    "IMH_Diabetes",
    "CDX_Diabetes",
    "CDX_Cog",
    "IMH_Alzheimers",
    "APOE4_Carriage",
    "APOE4_Genotype",
    "APOE4_Positivity",
]

# Candidate patterns to auto-detect continuous cognitive columns
# (Feel free to edit to your exact column names.)
CONTINUOUS_PATTERNS = {
    "Trails": [r"(?i)trail", r"(?i)tmta", r"(?i)tmtb", r"(?i)trails?_?a", r"(?i)trails?_?b"],
    "MMSE": [r"(?i)mmse"],
    "CDR": [r"(?i)cdr", r"(?i)cdrsb", r"(?i)cdr_sum", r"(?i)cdrglobal", r"(?i)cdglobal"],
}

N_BOOT = 2000
RANDOM_SEED = 42

# -----------------------
# Utils
# -----------------------

def describe_uniques(name: str, series: pd.Series, log_path: str, prefix: str = "global"):
    try:
        counts = series.value_counts(dropna=False)
        as_dict = {str(k): int(v) for k, v in counts.items()}
        log(f"[UNIQ-{prefix}] {name}: {as_dict}", log_path)
    except Exception as e:
        log(f"[UNIQ-{prefix}] {name}: <failed to summarize> ({e})", log_path)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log(msg: str, log_path: str):
    with open(log_path, "a") as f:
        f.write(msg.rstrip() + "\n")
    print(msg)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
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


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
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


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    mean = np.nanmean(x)
    return (x - mean) / sd


def binary_auc_with_ci(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
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
            continue
    return float(auc_base), bootstrap_ci(boot_vals)


def binary_or_per_sd_with_ci(y: np.ndarray, x_std: np.ndarray, n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
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
            continue
    return or_base, bootstrap_ci(boot_vals)


def multiclass_macro_auc_with_ci(y_raw: np.ndarray, x: np.ndarray, n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    classes = np.unique(y)

    def macro_auc(y_int: np.ndarray, x_feat: np.ndarray) -> float:
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
            continue
    return base_auc, bootstrap_ci(boot_vals)


def plot_roc_binary(y: np.ndarray, score: np.ndarray, title: str, out_png: str):
    fpr, tpr, _ = roc_curve(y, score)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_groups_box_violin(values: np.ndarray, groups: np.ndarray, title: str, out_png: str):
    vals = pd.Series(values).astype(float)
    cats = pd.Series(groups).astype(str)
    m = ~vals.isna() & ~cats.isna()
    vals = vals[m]
    cats = cats[m]
    labels = cats.unique()
    data = [vals[cats == c].values for c in labels]

    plt.figure()
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.boxplot(data, positions=np.arange(1, len(labels) + 1))
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=15)
    plt.ylabel("Value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def spearman_corr_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    from scipy.stats import spearmanr
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, (np.nan, np.nan)
    r_base, _ = spearmanr(x, y)
    rng = np.random.RandomState(seed)
    boot_vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        xb = x[idx]
        yb = y[idx]
        rb, _ = spearmanr(xb, yb)
        if not np.isnan(rb):
            boot_vals.append(rb)
    return float(r_base), bootstrap_ci(boot_vals)


def pearson_corr_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    from scipy.stats import pearsonr
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, (np.nan, np.nan)
    r_base, _ = pearsonr(x, y)
    rng = np.random.RandomState(seed)
    boot_vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        xb = x[idx]
        yb = y[idx]
        rb, _ = pearsonr(xb, yb)
        if not np.isnan(rb):
            boot_vals.append(rb)
    return float(r_base), bootstrap_ci(boot_vals)


def scatter_with_fit(x: np.ndarray, y: np.ndarray, title: str, xlab: str, ylab: str, out_png: str):
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    if len(x) >= 2:
        try:
            b1, b0 = np.polyfit(x, y, 1)
            xline = np.linspace(np.min(x), np.max(x), 100)
            yline = b1 * xline + b0
            plt.plot(xline, yline)
        except Exception:
            pass
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def find_col(df: pd.DataFrame, target: str) -> str:
    # case-insensitive exact match with minor normalization
    cols = {c.lower().replace(" ", "_") : c for c in df.columns}
    key = target.lower().replace(" ", "_")
    return cols.get(key, "")


def find_continuous_cols(df: pd.DataFrame, patterns: Dict[str, List[str]]) -> Dict[str, List[str]]:
    found = {k: [] for k in patterns.keys()}
    for block, pats in patterns.items():
        for pat in pats:
            rx = re.compile(pat)
            for c in df.columns:
                if rx.search(str(c)) and pd.api.types.is_numeric_dtype(df[c]):
                    if c not in found[block]:
                        found[block].append(c)
    return found

# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to CSV metadata")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory for results")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n_boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    roc_dir = os.path.join(args.outdir, "roc_curves")
    grp_dir = os.path.join(args.outdir, "group_plots")
    sca_dir = os.path.join(args.outdir, "scatter_plots")
    ensure_dir(roc_dir)
    ensure_dir(grp_dir)
    ensure_dir(sca_dir)
    log_path = os.path.join(args.outdir, "processing_log.txt")
    if os.path.exists(log_path):
        os.remove(log_path)

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"ERROR: could not read CSV: {e}")
        sys.exit(1)

    # --- CLASSIFICATION STATS ---
    rows: List[Dict] = []
    for pred in PREDICTORS:
        pred_col = find_col(df, pred)
        # Log global predictor availability and NaNs
        if pred_col != "":
            x_all_tmp = pd.to_numeric(df[pred_col], errors="coerce")
            log(f"[UNIQ-global] {pred}: non-NaN={int((~x_all_tmp.isna()).sum())}, NaN={int(x_all_tmp.isna().sum())}", log_path)
        if pred_col == "":
            log(f"[WARN] Predictor '{pred}' not found in CSV; skipping.", log_path)
            continue
        x_all = df[pred_col].astype(float).values

        for outcome in OUTCOMES:
            out_col = find_col(df, outcome)
            if out_col == "":
                log(f"[WARN] Outcome '{outcome}' not found; skipping.", log_path)
                continue

            y_ser = df[out_col]
            # Log global uniques before any filtering
            describe_uniques(outcome, y_ser, log_path, prefix="global")

            # Ignore rows with NaN in either predictor or outcome
            m = (~pd.isna(y_ser)) & (~pd.isna(x_all))
            x = x_all[m]
            y = y_ser[m]

            # Log subset uniques after NaN drop
            if isinstance(y, pd.Series):
                y_for_counts = y
            else:
                y_for_counts = pd.Series(y)
            describe_uniques(f"{outcome} | subset({pred})", y_for_counts, log_path, prefix="subset")

            if len(y) < 3:
                log(f"[WARN] Too few non-NaN rows for '{pred}' vs '{outcome}'; skipping.", log_path)
                continue

            # If outcome collapses to a single level in the subset, skip with explicit note
            if pd.api.types.is_numeric_dtype(y):
                uniq_tmp = np.unique(y.values)
                if len(uniq_tmp) < 2:
                    log(f"[INFO] After NaN filtering with predictor '{pred}', outcome '{outcome}' has a single level; skipping.", log_path)
                    continue
            else:
                cats_tmp = pd.Categorical(pd.Series(y).astype(str))
                if len([c for c in cats_tmp.categories if (cats_tmp == c).sum() > 0]) < 2:
                    log(f"[INFO] After NaN filtering with predictor '{pred}', outcome '{outcome}' has a single level; skipping.", log_path)
                    continue

            if pd.api.types.is_numeric_dtype(y):
                uniq = np.unique(y.values)
                if len(uniq) == 2:  # binary numeric
                    y_bin = (y.values == np.max(uniq)).astype(int)
                    try:
                        auc_base, (auc_lo, auc_hi) = binary_auc_with_ci(y_bin, x, args.n_boot, args.seed)
                        x_std = standardize(x)
                        or_base, (or_lo, or_hi) = binary_or_per_sd_with_ci(y_bin, x_std, args.n_boot, args.seed)
                        g1 = x[y_bin == 0]
                        g2 = x[y_bin == 1]
                        d = cohens_d(g2, g1)
                        cd = cliffs_delta(g2, g1)
                        roc_png = os.path.join(roc_dir, f"ROC_{pred}_{outcome}.png")
                        plot_roc_binary(y_bin, x, f"ROC: {pred} vs {outcome}", roc_png)
                        grp_png = os.path.join(grp_dir, f"Groups_{pred}_{outcome}.png")
                        plot_groups_box_violin(x, y.values, f"{pred} by {outcome}", grp_png)
                        rows.append({
                            "predictor": pred,
                            "outcome": outcome,
                            "n": int(len(y_bin)),
                            "positive_class": int(np.max(uniq)),
                            "auc": auc_base,
                            "auc_ci_lo": auc_lo,
                            "auc_ci_hi": auc_hi,
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
                        continue
                else:
                    log(f"[INFO] Outcome '{outcome}' is numeric with {len(uniq)} unique values; skipping AUC/effect sizes.", log_path)
                    continue
            else:
                cats = pd.Categorical(y.astype(str))
                if len(cats.categories) == 2:
                    # positive = majority class for stability
                    counts = [(cats.codes == i).sum() for i in range(2)]
                    pos_code = int(np.argmax(counts))
                    y_bin = (cats.codes == pos_code).astype(int)
                    try:
                        auc_base, (auc_lo, auc_hi) = binary_auc_with_ci(y_bin, x, args.n_boot, args.seed)
                        x_std = standardize(x)
                        or_base, (or_lo, or_hi) = binary_or_per_sd_with_ci(y_bin, x_std, args.n_boot, args.seed)
                        g1 = x[y_bin == 0]
                        g2 = x[y_bin == 1]
                        d = cohens_d(g2, g1)
                        cd = cliffs_delta(g2, g1)
                        roc_png = os.path.join(roc_dir, f"ROC_{pred}_{outcome}.png")
                        plot_roc_binary(y_bin, x, f"ROC: {pred} vs {outcome}", roc_png)
                        grp_png = os.path.join(grp_dir, f"Groups_{pred}_{outcome}.png")
                        plot_groups_box_violin(x, cats.astype(str).values, f"{pred} by {outcome}", grp_png)
                        rows.append({
                            "predictor": pred,
                            "outcome": outcome,
                            "n": int(len(y_bin)),
                            "positive_class": str(cats.categories[pos_code]),
                            "auc": auc_base,
                            "auc_ci_lo": auc_lo,
                            "auc_ci_hi": auc_hi,
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
                        continue
                else:
                    # Multiclass (e.g., APOE4_Genotype)
                    try:
                        base_auc, (lo, hi) = multiclass_macro_auc_with_ci(y.values, x, args.n_boot, args.seed)
                        grp_png = os.path.join(grp_dir, f"Groups_{pred}_{outcome}.png")
                        plot_groups_box_violin(x, cats.astype(str).values, f"{pred} by {outcome}", grp_png)
                        rows.append({
                            "predictor": pred,
                            "outcome": outcome,
                            "n": int(len(y)),
                            "positive_class": "n/a (multiclass)",
                            "auc": base_auc,
                            "auc_ci_lo": lo,
                            "auc_ci_hi": hi,
                            "or_per_+1SD": np.nan,
                            "or_ci_lo": np.nan,
                            "or_ci_hi": np.nan,
                            "cohens_d": np.nan,
                            "cliffs_delta": np.nan,
                            "plot_roc": "n/a (multiclass)",
                            "plot_groups": grp_png,
                        })
                        log(f"[INFO] Multiclass '{outcome}': macro AUC computed via OVR logistic regression.", log_path)
                    except Exception as e:
                        log(f"[WARN] Failed multiclass analysis for {pred} vs {outcome}: {e}", log_path)
                        continue

    if rows:
        out_csv = os.path.join(args.outdir, "summary_stats.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved classification summary: {out_csv}")
    else:
        print("No classification results computed.")

    # --- CONTINUOUS CORRELATES (Trails/MMSE/CDR) ---
    found_cols = find_continuous_cols(df, CONTINUOUS_PATTERNS)
    corr_rows: List[Dict] = []

    for block, cols in found_cols.items():
        if not cols:
            log(f"[INFO] No columns matched for {block}.", log_path)
            continue
        for pred in PREDICTORS:
            pred_col = find_col(df, pred)
            if pred_col == "":
                continue
            x_all = df[pred_col].astype(float).values
            for col in cols:
                y_all = pd.to_numeric(df[col], errors="coerce").values
                # Correlations
                sp_r, (sp_lo, sp_hi) = spearman_corr_with_ci(x_all, y_all, args.n_boot, args.seed)
                pe_r, (pe_lo, pe_hi) = pearson_corr_with_ci(x_all, y_all, args.n_boot, args.seed)
                # Plot
                sp_png = os.path.join(sca_dir, f"Scatter_{pred}_{block}_{re.sub('[^A-Za-z0-9]+','_', col)}.png")
                scatter_with_fit(x_all, y_all, f"{pred} vs {col}", pred, col, sp_png)
                # Save row
                n_eff = int((~np.isnan(x_all) & ~np.isnan(y_all)).sum())
                corr_rows.append({
                    "predictor": pred,
                    "block": block,
                    "column": col,
                    "n": n_eff,
                    "spearman_r": sp_r,
                    "spearman_ci_lo": sp_lo,
                    "spearman_ci_hi": sp_hi,
                    "pearson_r": pe_r,
                    "pearson_ci_lo": pe_lo,
                    "pearson_ci_hi": pe_hi,
                    "plot_scatter": sp_png,
                })

    if corr_rows:
        out_csv2 = os.path.join(args.outdir, "summary_correlations.csv")
        pd.DataFrame(corr_rows).to_csv(out_csv2, index=False)
        print(f"Saved correlation summary: {out_csv2}")
    else:
        print("No correlation results computed.")


if __name__ == "__main__":
    main()
