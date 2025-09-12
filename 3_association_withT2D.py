#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run associations on AB brain-age metrics:
- Welch t-tests vs IMH_Diabetes with violin plots (p-values annotated)
- Pearson correlations vs BW_HBA1c with R^2 and p (annotated)
- Save stats as CSVs and figures as PNGs

INPUT (enriched file from previous step):
  /home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_AB_BJ_with_PredictedAgeAB_BAG.csv

TESTED COLUMNS
  For t-tests (and violins):       ["BAG_AB","cBAG_AB","PredictedAge_AB","PredictedAge_corrected_AB","Delta_BAG_AB","Delta_cBAG_AB"]
  For correlations (and scatters): ["BAG_AB","cBAG_AB","PredictedAge_AB","PredictedAge_corrected_AB"]

OUTPUT (same folder, in 'associations_AB'):
  - HABS_AB_ttest_IMH_Diabetes.csv
  - HABS_AB_corr_HBA1c.csv
  - PNGs: violin_IMH_Diabetes_<metric>.png, scatter_HBA1c_<metric>.png
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Try seaborn for nicer violins; fall back to matplotlib if missing
try:
    import seaborn as sns
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

# --------------------- CONFIG ---------------------
ENRICHED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_AB_BJ_with_PredictedAgeAB_BAG.csv"

OUT_DIR = os.path.join("/home/bas/Desktop/MyData/harmonization/HABS/results/", "associations_AB")
os.makedirs(OUT_DIR, exist_ok=True)

TT_METRICS   = ["BAG_AB","cBAG_AB","PredictedAge_AB","PredictedAge_corrected_AB","Delta_BAG_AB","Delta_cBAG_AB"]
CORR_METRICS = ["BAG_AB","cBAG_AB","PredictedAge_AB","PredictedAge_corrected_AB"]

TT_CSV   = os.path.join(OUT_DIR, "HABS_AB_ttest_IMH_Diabetes.csv")
CORR_CSV = os.path.join(OUT_DIR, "HABS_AB_corr_HBA1c.csv")

# --------------------- HELPERS ---------------------
def clean_diabetes_flag(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    yes = {"1","yes","y","true","t","pos","positive","dm","diabetes","type 2","type2","type i","type ii"}
    no  = {"0","no","n","false","f","neg","negative"}
    if s in yes: return 1.0
    if s in no:  return 0.0
    try:
        v = float(s)
        return 1.0 if v > 0 else 0.0
    except Exception:
        return np.nan

def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def format_p(p):
    if pd.isna(p): return "p = n/a"
    if p < 1e-4:   return "p < 1e-4"
    return f"p = {p:.3g}"

def violin_plot_with_p(df, value_col, group_col, out_png, title, t_val, p_val, n0, n1):
    """Violin plot with Welch t/p annotation and n per group in xticks."""
    d = df[[value_col, group_col]].dropna()
    if d[group_col].nunique() < 2 or len(d) < 3:
        return False

    # Prepare xtick labels with n
    levels = sorted(d[group_col].unique())
    counts = [int((d[group_col] == g).sum()) for g in levels]
    xticklabels = [f"{int(g)} (n={c})" for g, c in zip(levels, counts)]

    plt.figure(figsize=(6, 5))
    if HAVE_SNS:
        sns.violinplot(data=d, x=group_col, y=value_col, inner="box", cut=0)
        sns.stripplot(data=d, x=group_col, y=value_col, color="k", alpha=0.25)
        ax = plt.gca()
        ax.set_xticklabels(xticklabels)
    else:
        groups = [d.loc[d[group_col]==g, value_col].values for g in levels]
        plt.violinplot(groups, showmeans=False, showmedians=True)
        plt.xticks([1, 2], xticklabels)
        ax = plt.gca()

    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)

    # Annotation box (lower-right)
    txt = f"Welch t = {t_val:.2f}\n{format_p(p_val)}\n" \
          f"n0 = {n0}, n1 = {n1}"
    ax.text(0.98, 0.02, txt,
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return True

def scatter_with_fit_and_p(x, y, out_png, xlabel, ylabel, title=None):
    """Scatter with regression line and annotation box for r, R², p, n."""
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
    plt.title(title if title else f"{ylabel} vs {xlabel}")
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    txt = f"r = {r:.3f}\nR² = {r**2:.3f}\n{format_p(p)}\nn = {len(d)}"
    ax.text(0.98, 0.02, txt,
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return True, r, r**2, p

# --------------------- MAIN ---------------------
def main():
    df = pd.read_csv(ENRICHED_CSV)

    # Checks
    for col in ["IMH_Diabetes", "BW_HBA1c"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {ENRICHED_CSV}")

    # Prepare group variable (binary 0/1)
    df["IMH_Diabetes_bin"] = df["IMH_Diabetes"].apply(clean_diabetes_flag)
    hba1c = to_numeric(df["BW_HBA1c"])

    # If SubjectRoot missing (needed for delta uniqueness), create a simple fallback
    if "SubjectRoot" not in df.columns:
        if "Subject_ID" in df.columns:
            sr = df["Subject_ID"].astype(str).str.replace(r"\.0$","",regex=True)
            df["SubjectRoot"] = sr.str.replace(r"([_\-]?y[02])$","",regex=True)
        else:
            df["SubjectRoot"] = np.arange(len(df))

    # ---------- Welch t-tests and violins ----------
    ttest_rows = []
    for metric in TT_METRICS:
        if metric not in df.columns:
            continue

        # For delta metrics, deduplicate per SubjectRoot to avoid double counting
        if metric.startswith("Delta_"):
            sub = df[["SubjectRoot", metric, "IMH_Diabetes_bin"]].copy()
            sub = sub.dropna(subset=[metric]).groupby("SubjectRoot", as_index=False).first()
        else:
            sub = df[[metric, "IMH_Diabetes_bin"]].copy()

        sub = sub.dropna(subset=[metric, "IMH_Diabetes_bin"])
        if sub["IMH_Diabetes_bin"].nunique() < 2 or len(sub) < 3:
            ttest_rows.append({
                "metric": metric, "n_total": len(sub),
                "n_diab0": int((sub["IMH_Diabetes_bin"]==0).sum()),
                "n_diab1": int((sub["IMH_Diabetes_bin"]==1).sum()),
                "mean0": np.nan, "mean1": np.nan, "std0": np.nan, "std1": np.nan,
                "t": np.nan, "p": np.nan, "note": "insufficient groups/samples"
            })
            # still attempt a violin for sanity (will likely skip)
            png = os.path.join(OUT_DIR, f"violin_IMH_Diabetes_{metric}.png")
            violin_plot_with_p(sub.rename(columns={"IMH_Diabetes_bin":"IMH_Diabetes"}),
                               metric, "IMH_Diabetes", png,
                               title=f"{metric} by IMH_Diabetes (0/1)",
                               t_val=np.nan, p_val=np.nan, n0=0, n1=0)
            continue

        g0 = sub.loc[sub["IMH_Diabetes_bin"]==0, metric].astype(float).values
        g1 = sub.loc[sub["IMH_Diabetes_bin"]==1, metric].astype(float).values
        if len(g0) < 2 or len(g1) < 2:
            t, p = np.nan, np.nan
            note = "too few per group"
        else:
            t, p = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
            note = ""

        mean0 = float(np.nanmean(g0)) if len(g0) else np.nan
        mean1 = float(np.nanmean(g1)) if len(g1) else np.nan
        std0  = float(np.nanstd(g0, ddof=1)) if len(g0) > 1 else np.nan
        std1  = float(np.nanstd(g1, ddof=1)) if len(g1) > 1 else np.nan

        ttest_rows.append({
            "metric": metric, "n_total": len(sub),
            "n_diab0": len(g0), "n_diab1": len(g1),
            "mean0": mean0, "mean1": mean1,
            "std0": std0, "std1": std1,
            "t": t, "p": p, "note": note
        })

        # Violin plot with p annotation
        viol_df = sub.rename(columns={"IMH_Diabetes_bin":"IMH_Diabetes"})
        png = os.path.join(OUT_DIR, f"violin_IMH_Diabetes_{metric}.png")
        violin_plot_with_p(
            viol_df, value_col=metric, group_col="IMH_Diabetes", out_png=png,
            title=f"{metric} by IMH_Diabetes (0/1)",
            t_val=(t if not np.isnan(t) else 0.0), p_val=p, n0=len(g0), n1=len(g1)
        )

    ttdf = pd.DataFrame(ttest_rows)
    ttdf.to_csv(TT_CSV, index=False)

    # ---------- Correlations vs HBA1c (R^2, p) ----------
    corr_rows = []
    for metric in CORR_METRICS:
        if metric not in df.columns:
            continue
        y = to_numeric(df[metric])
        ok = (~hba1c.isna()) & (~y.isna())
        if ok.sum() < 3:
            corr_rows.append({"metric": metric, "n": int(ok.sum()), "r": np.nan, "R2": np.nan, "p": np.nan})
            # still make an empty-ish plot with notice?
            continue

        r, p = stats.pearsonr(hba1c[ok], y[ok])
        corr_rows.append({"metric": metric, "n": int(ok.sum()), "r": r, "R2": r**2, "p": p})

        # Scatter with regression + annotation
        png = os.path.join(OUT_DIR, f"scatter_HBA1c_{metric}.png")
        scatter_with_fit_and_p(hba1c[ok], y[ok], png, xlabel="BW_HBA1c", ylabel=metric)

    cdf = pd.DataFrame(corr_rows)
    cdf.to_csv(CORR_CSV, index=False)

    print(f"[OK] Saved t-tests: {TT_CSV}  (rows={len(ttdf)})")
    print(f"[OK] Saved correlations: {CORR_CSV}  (rows={len(cdf)})")
    print(f"[OK] PNGs written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
