#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Associations for HABS brain-age metrics with:
  • ROC curves + 95% bootstrap AUC CIs and ROC bands
  • Accuracy, sensitivity, specificity, PPV, NPV (at Youden's J threshold)
  • Violin plots with Welch t-test p-values + effect sizes (Hedges g) + CIs
  • Scatter plots vs HBA1c with β (slope) + 95% CI, R², p
  • Corrected CDX_Cog coding: 0=Normal, 1=MCI, 2=AD (9=Unknown→NaN)
  • Added CDX_Diabetes endpoint
  • Date-suffixed output directory (e.g., associations_AB_YYYYMMDD)
  • RESTRICTED metrics: BAG_AB, cBAG_AB, Delta_BAG_AB

Outputs are written under a dated folder next to the input CSV.
"""

import os
import datetime
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy.stats import t as tdist

# ===================== CONFIG =====================
ENRICHED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_AB_enriched_v2.csv"

# Restrict analysis to these metrics only
TT_METRICS   = ["BAG_AB","cBAG_AB","Delta_BAG_AB","Delta_cBAG_AB"]
CORR_METRICS = ["BAG_AB","cBAG_AB","Delta_BAG_AB","Delta_cBAG_AB"]


# HBA1c thresholds to turn into binary endpoints for ROC/AUC
HBA1C_CUTOFFS = [5.7, 6.5]

# Bootstrap settings
N_BOOTSTRAPS = 2000
CI_LEVEL = 0.95
RNG_SEED = 42

# ===================== UTILS =====================
def ensure_outdir_under(path, sub="associations_AB"):
    base = os.path.dirname(os.path.abspath(path))
    today = datetime.date.today().strftime("%Y%m%d")
    out = os.path.join(base, f"{sub}_{today}")
    os.makedirs(out, exist_ok=True)
    return out

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def clean_binary(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    yes = {"1","yes","y","true","t","pos","positive","dm","diabetes"}
    no  = {"0","no","n","false","f","neg","negative"}
    if s in yes: return 1.0
    if s in no:  return 0.0
    try:
        v = float(s)
        return 1.0 if v > 0 else 0.0
    except Exception:
        return np.nan

def dedup_for_delta(df, metric, group_col="SubjectRoot", label_col=None):
    """For delta metrics, keep a single row per SubjectRoot (first non-null)."""
    cols = [c for c in [group_col, metric, label_col] if c is not None]
    if group_col in df.columns:
        sub = df[cols].dropna(subset=[metric]).copy()
        return sub.groupby(group_col, as_index=False).first()
    else:
        return df[[c for c in [metric, label_col] if c is not None]].dropna()

# ---------- Stats helpers ----------
def bootstrap_auc_ci(y_true, scores, n_boot=N_BOOTSTRAPS, ci=CI_LEVEL, seed=RNG_SEED):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    ok = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(y) < 3 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    base_auc = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yb, sb))
        except Exception:
            pass
    if not aucs:
        return base_auc, np.nan, np.nan
    alpha = 1 - ci
    lo = float(np.percentile(aucs, 100*(alpha/2)))
    hi = float(np.percentile(aucs, 100*(1 - alpha/2)))
    return float(base_auc), lo, hi

def bootstrap_mean_diff_ci(a, b, n_boot=N_BOOTSTRAPS, ci=CI_LEVEL, seed=RNG_SEED):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.nanmean(bb) - np.nanmean(aa))
    diffs = np.asarray(diffs)
    alpha = 1 - ci
    lo = float(np.percentile(diffs, 100*(alpha/2)))
    hi = float(np.percentile(diffs, 100*(1 - alpha/2)))
    return float(np.nanmean(b) - np.nanmean(a)), lo, hi

def cohen_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    na, nb = len(a), len(b)
    ma, mb = np.mean(a), np.mean(b)
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na-1)*sa + (nb-1)*sb) / (na+nb-2))
    d = (mb - ma) / sp
    J = 1 - (3/(4*(na+nb)-9))  # Hedges correction
    g = d * J
    return float(d), float(g)

def bootstrap_d_ci(a, b, n_boot=N_BOOTSTRAPS, ci=CI_LEVEL, seed=RNG_SEED):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan), (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ds, gs = [], []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        d, g = cohen_d(aa, bb)
        ds.append(d); gs.append(g)
    alpha = 1 - ci
    d_lo, d_hi = float(np.percentile(ds, 100*(alpha/2))), float(np.percentile(ds, 100*(1-alpha/2)))
    g_lo, g_hi = float(np.percentile(gs, 100*(alpha/2))), float(np.percentile(gs, 100*(1-alpha/2)))
    return (d_lo, d_hi), (g_lo, g_hi)

def linreg_with_ci(x, y, ci=CI_LEVEL):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    res = stats.linregress(x, y)
    beta, intercept, r, p, se = res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr
    dfree = len(x) - 2
    tcrit = tdist.ppf(0.5 + ci/2, dfree)
    beta_lo, beta_hi = beta - tcrit*se, beta + tcrit*se
    return float(beta), float(beta_lo), float(beta_hi), float(r**2), float(p)

def bootstrap_roc_band(y_true, scores, n_boot=N_BOOTSTRAPS, seed=RNG_SEED, grid_size=101):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    ok = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(y) < 3 or len(np.unique(y)) < 2:
        return None, None, None
    rng = np.random.default_rng(seed)
    fpr_grid = np.linspace(0, 1, grid_size)
    tprs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(yb, sb)
        uniq, uidx = np.unique(fpr_b, return_index=True)
        fpr_b = uniq; tpr_b = tpr_b[uidx]
        try:
            tprs.append(np.interp(fpr_grid, fpr_b, tpr_b))
        except Exception:
            pass
    if not tprs:
        return None, None, None
    tprs = np.vstack(tprs)
    lo = np.percentile(tprs, 2.5, axis=0)
    hi = np.percentile(tprs, 97.5, axis=0)
    return fpr_grid, lo, hi

def compute_operating_point(y_true, scores):
    """Compute Youden's J optimal threshold and return acc, sens, spec, thr, ppv, npv."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    fpr, tpr, thr = roc_curve(y, s)
    J = tpr - fpr
    k = int(np.nanargmax(J))
    t_opt = thr[k]
    sens = tpr[k]
    spec = 1 - fpr[k]
    y_pred = (s >= t_opt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return acc, sens, spec, t_opt, ppv, npv

# ---------- Plot helpers ----------
def violin_with_p(values, groups, out_png, title):
    d = pd.DataFrame({"val": values, "grp": groups}).dropna()
    if d["grp"].nunique() < 2 or len(d) < 3:
        return False, np.nan
    g0 = d.loc[d["grp"]==0, "val"].astype(float).values
    g1 = d.loc[d["grp"]==1, "val"].astype(float).values
    p = np.nan; diff = np.nan; d_eff = np.nan; g_eff = np.nan
    diff_lo = diff_hi = d_lo = d_hi = g_lo = g_hi = np.nan
    if len(g0) >= 2 and len(g1) >= 2:
        _, p = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
        diff, diff_lo, diff_hi = bootstrap_mean_diff_ci(g0, g1)
        d_eff, g_eff = cohen_d(g0, g1)
        (d_lo, d_hi), (g_lo, g_hi) = bootstrap_d_ci(g0, g1)
    plt.figure(figsize=(6,5))
    if HAVE_SNS:
        sns.violinplot(data=d, x="grp", y="val", inner="box", cut=0)
        sns.stripplot(data=d, x="grp", y="val", color="k", alpha=0.25)
    else:
        vals_by = [g["val"].values for _, g in d.groupby("grp")]
        plt.violinplot(vals_by, showmeans=False, showmedians=True)
        plt.xticks([1,2], sorted(d["grp"].unique()))
    plt.title(title)
    plt.xlabel("Group (0/1)"); plt.ylabel("Value")
    try:
        txt = f"p={p:.3g}\nΔmean={diff:.3g} [{diff_lo:.3g},{diff_hi:.3g}]\nHedges g={g_eff:.2f} [{g_lo:.2f},{g_hi:.2f}]"
        plt.text(0.02, 0.98, txt, ha="left", va="top", transform=plt.gca().transAxes)
    except Exception:
        pass
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
    return True, p

def scatter_with_fit(x, y, out_png, xlabel, ylabel, title=None):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 3:
        return False, np.nan, np.nan, np.nan
    beta, beta_lo, beta_hi, r2, p = linreg_with_ci(d["x"], d["y"]) 
    m, b = np.polyfit(d["x"], d["y"], 1)
    xs = np.array([d["x"].min(), d["x"].max()])
    ys = m*xs + b
    plt.figure(figsize=(6,5))
    plt.scatter(d["x"], d["y"], alpha=0.6, edgecolors="k")
    plt.plot(xs, ys, linewidth=2, alpha=0.7)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    ttl = title if title else f"{ylabel} vs {xlabel}\nβ={beta:.3g} [{beta_lo:.3g},{beta_hi:.3g}] • R²={r2:.3f} • p={p:.3g}"
    plt.title(ttl)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
    return True, r2, p, beta

def roc_with_bands(scores, ybin, title, out_png):
    d = pd.DataFrame({"s": scores, "y": ybin}).dropna()
    d["y"] = d["y"].astype(int)
    if len(d) < 3 or d["y"].nunique() < 2:
        return False, np.nan, None, None
    fpr, tpr, _ = roc_curve(d["y"], d["s"])
    auc_base, auc_lo, auc_hi = bootstrap_auc_ci(d["y"].values, d["s"].values)
    grid, band_lo, band_hi = bootstrap_roc_band(d["y"].values, d["s"].values)
    acc, sens, spec, thr, ppv, npv = compute_operating_point(d["y"].values, d["s"].values)
    plt.figure(figsize=(6,6))
    if grid is not None:
        plt.fill_between(grid, band_lo, band_hi, alpha=0.15, label="95% ROC band")
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC {auc_base:.3f} [{auc_lo:.3f},{auc_hi:.3f}])\nYJ thr={thr:.3g} • Acc={acc:.3g} • Sens={sens:.3g} • Spec={spec:.3g}\nPPV={ppv:.3g} • NPV={npv:.3g}")
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
    return True, float(auc_base), fpr, tpr

# ===================== MAIN =====================

def main():
    df = pd.read_csv(ENRICHED_CSV)
    OUT_DIR = ensure_outdir_under(ENRICHED_CSV, "associations_AB")

    # ---------- Prepare endpoints ----------
    df["IMH_Diabetes_bin"]   = df["IMH_Diabetes"].apply(clean_binary)   if "IMH_Diabetes"   in df.columns else np.nan
    df["IMH_Alzheimers_bin"] = df["IMH_Alzheimers"].apply(clean_binary) if "IMH_Alzheimers" in df.columns else np.nan
    df["CDX_Diabetes_bin"]   = df["CDX_Diabetes"].apply(clean_binary)   if "CDX_Diabetes"   in df.columns else np.nan

    # CDX_Cog recode: 0=Normal, 1=MCI, 2=AD; 9=Unknown -> NaN
    if "CDX_Cog" in df.columns:
        df["CDX_Cog_num"] = to_numeric(df["CDX_Cog"])
        df.loc[df["CDX_Cog_num"] == 9, "CDX_Cog_num"] = np.nan
        df["CDX_Cog_AnyImpair"] = np.where(df["CDX_Cog_num"].isin([1,2]), 1.0,
                                     np.where(df["CDX_Cog_num"] == 0, 0.0, np.nan))
        df["CDX_Cog_MCI"] = np.where(df["CDX_Cog_num"] == 1, 1.0,
                                np.where(df["CDX_Cog_num"] == 0, 0.0, np.nan))
        df["CDX_Cog_AD"]  = np.where(df["CDX_Cog_num"] == 2, 1.0,
                                np.where(df["CDX_Cog_num"] == 0, 0.0, np.nan))
    else:
        df["CDX_Cog_AnyImpair"] = np.nan
        df["CDX_Cog_MCI"]       = np.nan
        df["CDX_Cog_AD"]        = np.nan

    # HBA1c numeric
    df["BW_HBA1c_num"] = to_numeric(df["BW_HBA1c"]) if "BW_HBA1c" in df.columns else np.nan

    # ----------------- T-tests + violins (per endpoint) -----------------
    ttest_rows = []

    def run_ttests_for(endpoint_col, endpoint_name):
        for m in TT_METRICS:
            if m not in df.columns:
                continue
            if m.startswith("Delta_"):
                sub = dedup_for_delta(df, m, group_col="SubjectRoot", label_col=endpoint_col)
                vals, grps = sub[m], sub[endpoint_col]
            else:
                vals, grps = df[m], df[endpoint_col]
            sub_clean = pd.DataFrame({"v": vals, "g": grps}).dropna()
            if sub_clean["g"].nunique() < 2 or len(sub_clean) < 3:
                ttest_rows.append({"endpoint": endpoint_name, "metric": m, "n": int(len(sub_clean)),
                                   "mean0": np.nan, "mean1": np.nan, "diff": np.nan,
                                   "diff_L": np.nan, "diff_U": np.nan,
                                   "cohen_d": np.nan, "d_L": np.nan, "d_U": np.nan,
                                   "hedges_g": np.nan, "g_L": np.nan, "g_U": np.nan,
                                   "t": np.nan, "p": np.nan})
                continue
            g0 = sub_clean.loc[sub_clean["g"]==0, "v"].astype(float).values
            g1 = sub_clean.loc[sub_clean["g"]==1, "v"].astype(float).values
            if len(g0) >= 2 and len(g1) >= 2:
                t, p = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
                diff, diff_lo, diff_hi = bootstrap_mean_diff_ci(g0, g1)
                d_eff, g_eff = cohen_d(g0, g1)
                (d_lo, d_hi), (g_lo, g_hi) = bootstrap_d_ci(g0, g1)
            else:
                t=p=diff=diff_lo=diff_hi=d_eff=g_eff=d_lo=d_hi=g_lo=g_hi=np.nan
            ttest_rows.append({"endpoint": endpoint_name, "metric": m, "n": int(len(sub_clean)),
                               "mean0": float(np.nanmean(g0)) if len(g0) else np.nan,
                               "mean1": float(np.nanmean(g1)) if len(g1) else np.nan,
                               "diff": diff, "diff_L": diff_lo, "diff_U": diff_hi,
                               "cohen_d": d_eff, "d_L": d_lo, "d_U": d_hi,
                               "hedges_g": g_eff, "g_L": g_lo, "g_U": g_hi,
                               "t": t, "p": p})
            violin_with_p(vals, grps, os.path.join(OUT_DIR, f"violin_{endpoint_name}_{m}.png"),
                          title=f"{m} by {endpoint_name} (0/1)")

    # Run for available endpoints
    if pd.notna(df.get("IMH_Diabetes_bin")).any():   run_ttests_for("IMH_Diabetes_bin",   "IMH_Diabetes")
    if pd.notna(df.get("IMH_Alzheimers_bin")).any(): run_ttests_for("IMH_Alzheimers_bin", "IMH_Alzheimers")
    if pd.notna(df.get("CDX_Diabetes_bin")).any():   run_ttests_for("CDX_Diabetes_bin",   "CDX_Diabetes")
    for col,name in [("CDX_Cog_AnyImpair","CDX_Cog_AnyImpair"),("CDX_Cog_MCI","CDX_Cog_MCI"),("CDX_Cog_AD","CDX_Cog_AD")]:
        if pd.notna(df.get(col)).any():
            run_ttests_for(col, name)

    # ----------------- Correlations vs HBA1c -----------------
    corr_rows = []
    hba = df["BW_HBA1c_num"]
    for m in CORR_METRICS:
        if m not in df.columns:
            continue
        y = df[m]
        ok = (~hba.isna()) & (~y.isna())
        if ok.sum() < 3:
            corr_rows.append({"metric": m, "n": int(ok.sum()), "beta": np.nan, "beta_L": np.nan, "beta_U": np.nan, "R2": np.nan, "p": np.nan})
            continue
        beta, beta_lo, beta_hi, r2, p = linreg_with_ci(hba[ok], y[ok])
        corr_rows.append({"metric": m, "n": int(ok.sum()), "beta": beta, "beta_L": beta_lo, "beta_U": beta_hi, "R2": r2, "p": p})
        scatter_with_fit(hba[ok], y[ok], os.path.join(OUT_DIR, f"scatter_HBA1c_{m}.png"),
                         xlabel="BW_HBA1c", ylabel=m)

    # ----------------- AUCs with CIs + ROC overlays -----------------
    def auc_block(endpoint_col, endpoint_name):
        rows = []
        overlay = []
        for m in TT_METRICS:
            if m not in df.columns:
                continue
            if m.startswith("Delta_"):
                aligned = dedup_for_delta(df, m, group_col="SubjectRoot", label_col=endpoint_col)
                scores, labels = aligned[m], aligned[endpoint_col]
            else:
                aligned = df[[m, endpoint_col]].dropna()
                scores, labels = aligned[m], aligned[endpoint_col]
            if len(aligned) < 3 or aligned[endpoint_col].nunique() < 2:
                rows.append({"endpoint": endpoint_name, "metric": m, "n": int(len(aligned)),
                             "pos": np.nan, "neg": np.nan, "AUC": np.nan, "AUC_L": np.nan, "AUC_U": np.nan,
                             "Accuracy_YJ": np.nan, "Sensitivity_YJ": np.nan, "Specificity_YJ": np.nan,
                             "Threshold_YJ": np.nan, "PPV_YJ": np.nan, "NPV_YJ": np.nan})
                continue
            auc, lo, hi = bootstrap_auc_ci(labels.values, scores.values)
            acc, sens, spec, thr, ppv, npv = compute_operating_point(labels.values, scores.values)
            rows.append({"endpoint": endpoint_name, "metric": m, "n": int(len(aligned)),
                         "pos": int((labels==1).sum()), "neg": int((labels==0).sum()),
                         "AUC": auc, "AUC_L": lo, "AUC_U": hi,
                         "Accuracy_YJ": acc, "Sensitivity_YJ": sens, "Specificity_YJ": spec,
                         "Threshold_YJ": thr, "PPV_YJ": ppv, "NPV_YJ": npv})
            ok, auc_base, fpr, tpr = roc_with_bands(scores, labels, 
                title=f"{endpoint_name} — {m}",
                out_png=os.path.join(OUT_DIR, f"roc_{endpoint_name}_{m}.png"))
            if ok:
                overlay.append((fpr, tpr, m, auc, lo, hi))
        if overlay:
            plt.figure(figsize=(6,6))
            for fpr, tpr, m, auc, lo, hi in overlay:
                plt.plot(fpr, tpr, lw=2, label=f"{m} (AUC {auc:.3f} [{lo:.3f},{hi:.3f}])")
            plt.plot([0,1],[0,1],'k--', lw=1)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title(f"ROC — {endpoint_name}")
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"roc_{endpoint_name}_all.png"), dpi=300); plt.close()
        return pd.DataFrame(rows)

    auc_tables = []
    if pd.notna(df.get("IMH_Diabetes_bin")).any():
        t = auc_block("IMH_Diabetes_bin","IMH_Diabetes"); auc_tables.append(("AUC_summary_IMH_Diabetes.csv", t))
    if pd.notna(df.get("IMH_Alzheimers_bin")).any():
        t = auc_block("IMH_Alzheimers_bin","IMH_Alzheimers"); auc_tables.append(("AUC_summary_IMH_Alzheimers.csv", t))
    if pd.notna(df.get("CDX_Diabetes_bin")).any():
        t = auc_block("CDX_Diabetes_bin","CDX_Diabetes"); auc_tables.append(("AUC_summary_CDX_Diabetes.csv", t))

    for col,name,fname in [
        ("CDX_Cog_AnyImpair","CDX_Cog_AnyImpair","AUC_summary_CDX_Cog_AnyImpair.csv"),
        ("CDX_Cog_MCI","CDX_Cog_MCI","AUC_summary_CDX_Cog_MCI.csv"),
        ("CDX_Cog_AD","CDX_Cog_AD","AUC_summary_CDX_Cog_AD.csv"),
    ]:
        if pd.notna(df.get(col)).any():
            t = auc_block(col, name); auc_tables.append((fname, t))

    # HBA1c cutoffs
    hba_auc_rows = []
    for cut in HBA1C_CUTOFFS:
        label = (df["BW_HBA1c_num"] >= cut).astype(float)
        if label.notna().sum() < 3 or label.dropna().nunique() < 2:
            continue
        curves = []
        for m in TT_METRICS:
            if m not in df.columns:
                continue
            if m.startswith("Delta_"):
                aligned = dedup_for_delta(df, m, group_col="SubjectRoot")
                aligned = aligned.join(label.rename("yb"), how="left")
                aligned = aligned.rename(columns={m:"s"}).dropna(subset=["s","yb"])
            else:
                aligned = pd.DataFrame({"s": df[m], "yb": label}).dropna()
            if len(aligned) < 3 or aligned["yb"].nunique() < 2:
                hba_auc_rows.append({"cutoff": cut, "metric": m, "n": int(len(aligned)),
                                     "pos": np.nan, "neg": np.nan, "AUC": np.nan, "AUC_L": np.nan, "AUC_U": np.nan,
                                     "Accuracy_YJ": np.nan, "Sensitivity_YJ": np.nan, "Specificity_YJ": np.nan,
                                     "Threshold_YJ": np.nan, "PPV_YJ": np.nan, "NPV_YJ": np.nan})
                continue
            auc, lo, hi = bootstrap_auc_ci(aligned["yb"].values, aligned["s"].values)
            acc, sens, spec, thr, ppv, npv = compute_operating_point(aligned["yb"].values, aligned["s"].values)
            hba_auc_rows.append({"cutoff": cut, "metric": m, "n": int(len(aligned)),
                                 "pos": int((aligned["yb"]==1).sum()),
                                 "neg": int((aligned["yb"]==0).sum()),
                                 "AUC": auc, "AUC_L": lo, "AUC_U": hi,
                                 "Accuracy_YJ": acc, "Sensitivity_YJ": sens, "Specificity_YJ": spec,
                                 "Threshold_YJ": thr, "PPV_YJ": ppv, "NPV_YJ": npv})
            fpr, tpr, _ = roc_curve(aligned["yb"].astype(int), aligned["s"].astype(float))
            curves.append((fpr, tpr, m, auc, lo, hi))
        if curves:
            plt.figure(figsize=(6,6))
            for fpr, tpr, m, auc, lo, hi in curves:
                plt.plot(fpr, tpr, lw=2, label=f"{m} (AUC {auc:.3f} [{lo:.3f},{hi:.3f}])")
            plt.plot([0,1],[0,1],'k--', lw=1)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title(f"ROC — HBA1c ≥ {cut}")
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"roc_HBA1c_ge_{cut}_all.png"), dpi=300); plt.close()

    # ---------- SAVE TABLES ----------
    pd.DataFrame(ttest_rows).to_csv(os.path.join(OUT_DIR, "TTest_summary.csv"), index=False)
    pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "Corr_HBA1c_summary.csv"), index=False)

    combined = []
    for fname, t in auc_tables:
        if not t.empty:
            t.to_csv(os.path.join(OUT_DIR, fname), index=False)
            combined.append(t)
    if hba_auc_rows:
        pd.DataFrame(hba_auc_rows).to_csv(os.path.join(OUT_DIR, "AUC_summary_HBA1cCutoffs.csv"), index=False)
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(os.path.join(OUT_DIR, "AUC_summary_ALL.csv"), index=False)

    print(f"[OK] Wrote outputs to: {OUT_DIR}")

if __name__ == "__main__":
    main()
