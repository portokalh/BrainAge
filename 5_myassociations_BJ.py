#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Associations for HABS brain-age metrics (AB stack)

INPUT:
  ENRICHED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_with_PredictedAgeAB_BAG.csv"

WHAT IT DOES:
  - T-tests + violin plots (with p-value on plot) for:
      * IMH_Diabetes (binary)
      * IMH_Alzheimers (binary)
  - One-way ANOVA + violin plots for:
      * CDX_Cog (excludes code 9)
  - Pearson correlations + scatter (with regression, R/R²/p) for:
      * BW_HBA1c
      * CDR_Sum and CDR_Global

OUTPUT (in OUT_DIR = ".../results/associations_AB"):
  - HABS_AB_ttest_IMH_Diabetes.csv
  - HABS_AB_ttest_IMH_Alzheimers.csv
  - anova_CDX_Cog_AB.csv
  - corr_HBA1c_AB.csv
  - corr_CDR_AB.csv
  - PNGs: violin_* and scatter_* per metric
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Try seaborn for nicer violins; fallback to matplotlib otherwise
try:
    import seaborn as sns
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

# --------------------- CONFIG ---------------------
ENRICHED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_with_PredictedAgeAB_BAG.csv"
OUT_DIR = os.path.join("/home/bas/Desktop/MyData/harmonization/HABS/results/", "associations_AB")
os.makedirs(OUT_DIR, exist_ok=True)

# Candidate metrics we’ll test (only those present will be used)
WIDE_METRICS = [
    "BAG_AB", "cBAG_AB",
    "PredictedAge_AB", "PredictedAge_corrected_AB",
    "Delta_BAG_AB", "Delta_cBAG_AB",
]

# --------------------- HELPERS ---------------------
def coerce_num(s):
    """Numeric coercion with NaNs for non-numeric entries."""
    return pd.to_numeric(s, errors="coerce")

def find_col(df, candidates):
    """
    Return the first column in df whose name matches any in `candidates`
    (case-insensitive, ignores non-alnum). None if not found.
    """
    norm = {re.sub(r'[^a-z0-9]+', '', c.lower()): c for c in df.columns}
    for c in candidates:
        k = re.sub(r'[^a-z0-9]+', '', c.lower())
        if k in norm:
            return norm[k]
    return None

def clean_binary_flag(x):
    """Map various encodings to {0,1}, NaN if unknown."""
    if pd.isna(x): return np.nan
    sx = str(x).strip().lower()
    yes = {"1","yes","y","true","t","pos","positive","dm","diabetes","type2","type 2","typei","type i","typeii","type ii"}
    no  = {"0","no","n","false","f","neg","negative"}
    if sx in yes: return 1.0
    if sx in no:  return 0.0
    try:
        v = float(sx)
        return 1.0 if v > 0 else 0.0
    except Exception:
        return np.nan

def annotate_text(ax, text, loc="upper right"):
    """Place a small annotation box with text on given axes."""
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"
    x = 0.98 if "right" in loc else 0.02
    y = 0.98 if "upper" in loc else 0.02
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, ec='0.3'))

def violin_ttest(df, value_col, group_col, out_png, title=None):
    """
    Violin plot for binary group + Welch t-test. Writes PNG.
    Returns (t, p, n0, n1).
    """
    d = df[[value_col, group_col]].dropna()
    if d[group_col].nunique() < 2 or len(d) < 3:
        return np.nan, np.nan, int((d[group_col]==0).sum()), int((d[group_col]==1).sum())

    g0 = d.loc[d[group_col]==0, value_col].astype(float).values
    g1 = d.loc[d[group_col]==1, value_col].astype(float).values

    if len(g0) < 2 or len(g1) < 2:
        t, p = np.nan, np.nan
    else:
        t, p = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    if HAVE_SNS:
        sns.violinplot(data=d, x=group_col, y=value_col, inner="box", cut=0, ax=ax)
        sns.stripplot(data=d, x=group_col, y=value_col, color="k", alpha=0.25, ax=ax)
        ax.set_xticklabels(["0","1"])
    else:
        groups = [g[value_col].values for _, g in d.groupby(group_col)]
        plt.violinplot(groups, showmeans=False, showmedians=True)
        plt.xticks([1,2], ["0","1"])

    ax.set_title(title if title else f"{value_col} by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)

    # p-value note
    annotate_text(ax, f"Welch t-test: p = {p:.3g}" if pd.notna(p) else "Welch t-test: n/a", loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return t, p, len(g0), len(g1)

def violin_anova(df, value_col, group_col, out_png, title_prefix=""):
    """
    Violin plot across categorical groups + one-way ANOVA.
    Returns (F, p, k_groups).
    """
    d = df[[value_col, group_col]].dropna()
    if d[group_col].nunique() < 2 or len(d) < 3:
        return np.nan, np.nan, int(d[group_col].nunique())

    # Collect arrays per group
    grouped = [g[value_col].astype(float).values for _, g in d.groupby(group_col)]
    if any(len(g) < 2 for g in grouped):
        F, p = np.nan, np.nan
    else:
        F, p = stats.f_oneway(*grouped)

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    if HAVE_SNS:
        sns.violinplot(data=d, x=group_col, y=value_col, inner="box", cut=0, ax=ax)
        sns.stripplot(data=d, x=group_col, y=value_col, color="k", alpha=0.2, ax=ax)
    else:
        # Fallback: simple violins via matplotlib (group order by sorted labels)
        levels = sorted(d[group_col].unique())
        groups = [d.loc[d[group_col]==lv, value_col].values for lv in levels]
        plt.violinplot(groups, showmeans=False, showmedians=True)
        plt.xticks(range(1, len(levels)+1), [str(lv) for lv in levels])

    ttl = f"{title_prefix}{value_col} by {group_col}"
    ax.set_title(ttl)
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)

    annotate_text(ax, f"One-way ANOVA: p = {p:.3g}" if pd.notna(p) else "One-way ANOVA: n/a", loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return F, p, d[group_col].nunique()

def scatter_with_fit(x, y, out_png, xlabel, ylabel, title=None):
    """Scatter with OLS line + R/R²/p in title."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 3:
        return False, np.nan, np.nan, np.nan
    r, p = stats.pearsonr(d["x"], d["y"])
    m, b = np.polyfit(d["x"], d["y"], 1)
    xs = np.array([d["x"].min(), d["x"].max()])
    ys = m*xs + b

    plt.figure(figsize=(6,5))
    plt.scatter(d["x"], d["y"], alpha=0.6, edgecolors="k")
    plt.plot(xs, ys, linewidth=2, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title else f"{ylabel} vs {xlabel}\nR={r:.3f}, R²={r**2:.3f}, p={p:.3g}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return True, r, r**2, p

# --------------------- MAIN ---------------------
def main():
    df = pd.read_csv(ENRICHED_CSV)

    # Pick the metrics that actually exist in the file
    METRICS = [m for m in WIDE_METRICS if m in df.columns]
    if not METRICS:
        raise ValueError(f"No expected metrics found in {ENRICHED_CSV}. Looked for: {WIDE_METRICS}")

    # Useful columns we may need
    subj_root_col = find_col(df, ["SubjectRoot", "Subject_Root"])
    if subj_root_col is None:
        # Make a simple fallback from Subject or Subject_ID if present
        sid_col = find_col(df, ["Subject_ID","Subject","MRI_Exam_fixed","DWI"])
        if sid_col:
            sr = df[sid_col].astype(str).str.replace(r"\.0$","",regex=True)
            df["SubjectRoot"] = sr.str.replace(r"([_\-]?y[0-9]+)$","",regex=True)
            subj_root_col = "SubjectRoot"

    # ----------------- T-TESTS: IMH_Diabetes -----------------
    diab_col = find_col(df, ["IMH_Diabetes"])
    ttest_rows = []
    if diab_col:
        df["IMH_Diabetes_bin"] = df[diab_col].apply(clean_binary_flag)
        for metric in METRICS:
            # Deduplicate delta metrics per SubjectRoot
            if metric.startswith("Delta_") and subj_root_col:
                sub = df[[subj_root_col, metric, "IMH_Diabetes_bin"]].dropna(subset=[metric]).copy()
                sub = sub.groupby(subj_root_col, as_index=False).first()
                plot_df = sub.rename(columns={"IMH_Diabetes_bin":"GroupBin"})
            else:
                plot_df = df[[metric, "IMH_Diabetes_bin"]].rename(columns={"IMH_Diabetes_bin":"GroupBin"})
            plot_df = plot_df.dropna(subset=[metric, "GroupBin"])

            png = os.path.join(OUT_DIR, f"violin_IMH_Diabetes_{metric}.png")
            t, p, n0, n1 = violin_ttest(plot_df, metric, "GroupBin", png, title=f"{metric} by IMH_Diabetes (0/1)")
            ttest_rows.append({
                "outcome": metric, "group": "IMH_Diabetes",
                "n_total": len(plot_df), "n_0": n0, "n_1": n1,
                "test": "Welch t-test", "t": t, "p": p
            })
        pd.DataFrame(ttest_rows).to_csv(os.path.join(OUT_DIR, "HABS_AB_ttest_IMH_Diabetes.csv"), index=False)

    # ----------------- T-TESTS: IMH_Alzheimers -----------------
    alz_col = find_col(df, ["IMH_Alzheimers", "IMH_Alzheimer", "IMH_Alzheimer_s"])
    ttest_alz_rows = []
    if alz_col:
        df["IMH_Alzheimers_bin"] = df[alz_col].apply(clean_binary_flag)
        for metric in METRICS:
            if metric.startswith("Delta_") and subj_root_col:
                sub = df[[subj_root_col, metric, "IMH_Alzheimers_bin"]].dropna(subset=[metric]).copy()
                sub = sub.groupby(subj_root_col, as_index=False).first()
                plot_df = sub.rename(columns={"IMH_Alzheimers_bin":"GroupBin"})
            else:
                plot_df = df[[metric, "IMH_Alzheimers_bin"]].rename(columns={"IMH_Alzheimers_bin":"GroupBin"})
            plot_df = plot_df.dropna(subset=[metric, "GroupBin"])

            png = os.path.join(OUT_DIR, f"violin_IMH_Alzheimers_{metric}.png")
            t, p, n0, n1 = violin_ttest(plot_df, metric, "GroupBin", png, title=f"{metric} by IMH_Alzheimers (0/1)")
            ttest_alz_rows.append({
                "outcome": metric, "group": "IMH_Alzheimers",
                "n_total": len(plot_df), "n_0": n0, "n_1": n1,
                "test": "Welch t-test", "t": t, "p": p
            })
        pd.DataFrame(ttest_alz_rows).to_csv(os.path.join(OUT_DIR, "HABS_AB_ttest_IMH_Alzheimers.csv"), index=False)

    # ----------------- ANOVA: CDX_Cog (exclude code 9) -----------------
    cog_col = find_col(df, ["CDX_Cog","cdx_cog","CDX_COG"])
    if cog_col:
        df[cog_col] = coerce_num(df[cog_col])
        # drop code 9 (unknown / not assigned)
        d_cog = df[(~df[cog_col].isna()) & (df[cog_col] != 9)].copy()
        # round and cast to int categories
        d_cog[cog_col] = d_cog[cog_col].round(0).astype(int)

        anova_rows = []
        for metric in METRICS:
            # Deduplicate delta metrics per SubjectRoot
            if metric.startswith("Delta_") and subj_root_col:
                sub = d_cog[[subj_root_col, metric, cog_col]].dropna(subset=[metric]).copy()
                sub = sub.groupby(subj_root_col, as_index=False).first()
            else:
                sub = d_cog[[metric, cog_col]].dropna()

            png = os.path.join(OUT_DIR, f"violin_{cog_col}_{metric}.png")
            F, p, k = violin_anova(sub, metric, cog_col, png, title_prefix="")
            anova_rows.append({"outcome": metric, "group": cog_col, "test": "ANOVA", "F": F, "p": p, "levels": k})

        pd.DataFrame(anova_rows).to_csv(os.path.join(OUT_DIR, "anova_CDX_Cog_AB.csv"), index=False)

    # ----------------- CORR: BW_HBA1c -----------------
    hba1c_col = find_col(df, ["BW_HBA1c","BW_HbA1c","HBA1c","HbA1c"])
    corr_rows = []
    if hba1c_col:
        x = coerce_num(df[hba1c_col])
        for metric in [m for m in METRICS if not m.startswith("Delta_")]:  # deltas not ideal for HBA1c cross-sectionally
            if metric not in df.columns:
                continue
            y = coerce_num(df[metric])
            ok = (~x.isna()) & (~y.isna())
            if ok.sum() < 3:
                corr_rows.append({"metric": metric, "x": hba1c_col, "n": int(ok.sum()), "r": np.nan, "R2": np.nan, "p": np.nan})
                continue
            r, p = stats.pearsonr(x[ok], y[ok])
            corr_rows.append({"metric": metric, "x": hba1c_col, "n": int(ok.sum()), "r": float(r), "R2": float(r**2), "p": float(p)})
            png = os.path.join(OUT_DIR, f"scatter_{hba1c_col}_{metric}.png")
            scatter_with_fit(x[ok].astype(float), y[ok].astype(float), png, xlabel=hba1c_col, ylabel=metric)
        pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "corr_HBA1c_AB.csv"), index=False)

    # ----------------- CORR: CDR_Sum / CDR_Global -----------------
    cdr_cols = [find_col(df, ["CDR_Sum"]), find_col(df, ["CDR_Global"])]
    cdr_cols = [c for c in cdr_cols if c]
    cdr_rows = []
    for cdr_col in cdr_cols:
        x = coerce_num(df[cdr_col])
        for metric in [m for m in METRICS if not m.startswith("Delta_")]:
            if metric not in df.columns:
                continue
            y = coerce_num(df[metric])
            ok = (~x.isna()) & (~y.isna())
            if ok.sum() < 3:
                cdr_rows.append({"metric": metric, "cdr": cdr_col, "n": int(ok.sum()), "r": np.nan, "R2": np.nan, "p": np.nan})
                continue
            r, p = stats.pearsonr(x[ok], y[ok])
            cdr_rows.append({"metric": metric, "cdr": cdr_col, "n": int(ok.sum()), "r": float(r), "R2": float(r**2), "p": float(p)})
            png = os.path.join(OUT_DIR, f"scatter_{cdr_col}_{metric}.png")
            scatter_with_fit(x[ok].astype(float), y[ok].astype(float), png, xlabel=cdr_col, ylabel=metric,
                             title=f"{metric} vs {cdr_col}")
    if cdr_rows:
        pd.DataFrame(cdr_rows).to_csv(os.path.join(OUT_DIR, "corr_CDR_AB.csv"), index=False)

    print(f"[OK] Outputs saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
