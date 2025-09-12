#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append Brain-Age metrics to metadata and run simple associations.

- Fixes dtype mismatch on merge keys by normalizing IDs to strings.
- Adds: Predicted_Age_AB, Real_Age_AB, Real_Age_corrected_AB, BAG_AB, cBAG_AB,
        Delta_BAG_AB, Delta_cBAG_AB, Risk_Diabetes_AB, BW_HBA1c_num
- Writes association summary CSV (t-tests and Pearson r).

Bas – 2025-09-12
"""

# ============================== IMPORTS ==============================
import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats

# ============================ CONFIGURATION ============================
PRED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/results/cv_predictions.csv"
META_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_AB_BJ.csv"
META_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_AB_BJ_short.csv"
WRITE_BACKUP = True  # set False to skip writing a .bak copy
ASSOC_OUT = os.path.join(os.path.dirname(META_CSV), "HABS_AB_assoc_summary.csv")

# ============================ HELPERS ============================
def _first_match(cands, cols):
    for c in cands:
        if c in cols:
            return c
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def standardize_ids(df):
    """Create Subject_ID from any of several possible ID columns, if needed."""
    subj_cands = ["Subject_ID","subject_id","MRI_Exam_fixed","Subject","subject","RID","ID","runno","Runno","runNo"]
    col = _first_match(subj_cands, list(df.columns))
    if col and "Subject_ID" not in df.columns:
        df["Subject_ID"] = df[col]
    return df

def _as_key_series(s: pd.Series) -> pd.Series:
    """Normalize ID-like keys to clean strings for safe merging."""
    s = s.astype(str)
    s = s.str.strip()
    # remove trailing ".0" from IDs that were read as floats
    s = s.str.replace(r"\.0$", "", regex=True)
    return s

def clean_diabetes_flag(x):
    """Map various encodings to binary {0,1}, NaN if unknown."""
    if pd.isna(x):
        return np.nan
    t = str(x).strip().lower()
    if t in {"1","yes","y","true","t","pos","positive"}: return 1
    if t in {"0","no","n","false","f","neg","negative"}: return 0
    if t in {"dm","diabetes","type 2","type2","type i","type ii"}: return 1
    try:
        v = float(t)
        return 1 if v > 0 else 0
    except Exception:
        return np.nan

def coerce_numeric(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series([np.nan]*len(s))

# ============================ LOAD FILES ============================
meta = pd.read_csv(META_CSV)
pred = pd.read_csv(PRED_CSV)

# Create Subject_ID if needed
meta = standardize_ids(meta)
pred  = standardize_ids(pred)

# Immediately normalize IDs (prevents int64/object merge errors)
meta["Subject_ID"] = _as_key_series(meta["Subject_ID"])
pred["Subject_ID"] = _as_key_series(pred["Subject_ID"])



# =================== STANDARDIZE AGE / PRED COLUMNS IN PREDICTIONS ===================
true_cands = ["Real_Age","real_age","y_true","true_age","Age","age","ChronologicalAge","Chron_Age","Target","target","Label","label"]
pred_cands = ["Predicted_Age","predicted_age","y_pred","pred_age","prediction","predicted",
              "Pred","Prediction","BrainAge","brain_age_pred","PredictedAge"]

tc = _first_match(true_cands,  list(pred.columns))
pc = _first_match(pred_cands,  list(pred.columns))
if tc and "Real_Age" not in pred.columns:        pred["Real_Age"] = pred[tc]
if pc and "Predicted_Age" not in pred.columns:   pred["Predicted_Age"] = pred[pc]

for c in ["Subject_ID","Real_Age","Predicted_Age"]:
    if c not in pred.columns:
        raise ValueError(f"Predictions missing required column: {c}")

# ============================ KEEP y0/y2; +2y TO y2 ============================
mask_y_any = pred["Subject_ID"].str.contains(r"(y0|y2)", case=False, na=False)
pred = pred.loc[mask_y_any].copy()

pred["Timepoint"] = np.where(pred["Subject_ID"].str.contains("y2", case=False, na=False), "y2", "y0")
pred["SubjectRoot"] = pred["Subject_ID"].str.replace(r"([_\-]?y[02])$", "", regex=True)

# +2 years for y2 rows
pred.loc[pred["Timepoint"].eq("y2"), "Real_Age"] = pred.loc[pred["Timepoint"].eq("y2"), "Real_Age"] + 2.0

# ============================ COLLAPSE REPEATS PER Subject_ID ============================
per_id = pred.groupby("Subject_ID", as_index=False).agg({
    "SubjectRoot": "first",
    "Timepoint": "first",
    "Real_Age": "first",        # adjusted
    "Predicted_Age": "mean"     # mean across repeats/CV
})

# ============================ BAG + BIAS CORRECTION ============================
per_id["BAG"] = per_id["Predicted_Age"] - per_id["Real_Age"]

X = per_id["Real_Age"].values.reshape(-1,1)
Y = per_id["Predicted_Age"].values
reg = LinearRegression().fit(X, Y)
expected = reg.predict(X)                   # expected predicted age given Real_Age
per_id["cBAG"] = Y - expected               # residuals
per_id["Real_Age_corrected"] = expected     # save expected as a reference ("corrected" real-age proxy)

print(f"[Info] Bias model: Predicted = {reg.coef_[0]:.4f} * Real + {reg.intercept_:.4f}")

# ============================ DELTAS PER SubjectRoot (y2 - y0) ============================
roots_y0 = set(per_id.loc[per_id["Timepoint"]=="y0","SubjectRoot"].unique())
roots_y2 = set(per_id.loc[per_id["Timepoint"]=="y2","SubjectRoot"].unique())
roots_both = roots_y0 & roots_y2
print(f"[Counts] y0 roots: {len(roots_y0)} | y2 roots: {len(roots_y2)} | paired: {len(roots_both)}")

y0 = per_id[per_id["Timepoint"]=="y0"].set_index("SubjectRoot")
y2 = per_id[per_id["Timepoint"]=="y2"].set_index("SubjectRoot")
aligned = y0.join(y2, how="inner", lsuffix="_y0", rsuffix="_y2")

aligned["deltaBAG"]  = aligned["BAG_y2"]  - aligned["BAG_y0"]
aligned["deltaCBAG"] = aligned["cBAG_y2"] - aligned["cBAG_y0"]
deltas = aligned[["deltaBAG","deltaCBAG"]].reset_index()

# Normalize SubjectRoot before merge
deltas["SubjectRoot"] = _as_key_series(deltas["SubjectRoot"])

# ============================ APPEND TO METADATA (_AB SUFFIX) ============================
# 1) Timepoint-level merge on Subject_ID
per_id_for_merge = per_id[["Subject_ID","Predicted_Age","Real_Age","Real_Age_corrected","BAG","cBAG"]].copy()
per_id_for_merge.rename(columns={
    "Predicted_Age":        "Predicted_Age_AB",
    "Real_Age":             "Real_Age_AB",
    "Real_Age_corrected":   "Real_Age_corrected_AB",
    "BAG":                  "BAG_AB",
    "cBAG":                 "cBAG_AB",
}, inplace=True)

# ensure both sides are normalized strings
per_id_for_merge["Subject_ID"] = _as_key_series(per_id_for_merge["Subject_ID"])
meta["Subject_ID"] = _as_key_series(meta["Subject_ID"])

# MERGE 1
meta_merged = meta.merge(per_id_for_merge, on="Subject_ID", how="left")

# 2) Delta merge on SubjectRoot (same delta copied to y0 & y2 rows)
meta_merged["SubjectRoot"] = _as_key_series(
    meta_merged["Subject_ID"].astype(str).str.replace(r"([_\-]?y[02])$", "", regex=True)
)

deltas_for_merge = deltas.rename(columns={
    "deltaBAG":"Delta_BAG_AB",
    "deltaCBAG":"Delta_cBAG_AB"
}).copy()
# (already normalized above)

# MERGE 2
meta_merged = meta_merged.merge(deltas_for_merge, on="SubjectRoot", how="left")

# ============================ SAVE BACK (append in place) ============================
if WRITE_BACKUP and os.path.exists(META_CSV):
    bak_path = META_CSV + ".bak"
    meta.to_csv(bak_path, index=False)
    print(f"[Backup] Wrote: {bak_path}")

meta_merged.to_csv(META_CSV, index=False)
print(f"[OK] Appended AB columns and saved: {META_CSV}")

# ============================ ASSOCIATIONS ============================
# Risk_Diabetes_AB = IMH_Diabetes OR CDX_Diabetes
imh = meta_merged["IMH_Diabetes"] if "IMH_Diabetes" in meta_merged.columns else pd.Series(np.nan, index=meta_merged.index)
cdx = meta_merged["CDX_Diabetes"] if "CDX_Diabetes" in meta_merged.columns else pd.Series(np.nan, index=meta_merged.index)
imh_bin = pd.Series([clean_diabetes_flag(v) for v in imh], index=meta_merged.index, dtype="float")
cdx_bin = pd.Series([clean_diabetes_flag(v) for v in cdx], index=meta_merged.index, dtype="float")

risk_series = imh_bin.copy()
risk_series = risk_series.where(~risk_series.isna(), cdx_bin)  # fill with CDX when IMH missing
mask_both = (~imh_bin.isna()) & (~cdx_bin.isna())
risk_series.loc[mask_both] = ((imh_bin.loc[mask_both] > 0) | (cdx_bin.loc[mask_both] > 0)).astype(int)
meta_merged["Risk_Diabetes_AB"] = risk_series

# HBA1c numeric
meta_merged["BW_HBA1c_num"] = coerce_numeric(meta_merged["BW_HBA1c"]) if "BW_HBA1c" in meta_merged.columns else np.nan

# ---- simple stats helpers
def assoc_continuous(x, y, label_x, label_y):
    d = pd.DataFrame({"x":x, "y":y}).dropna()
    if len(d) < 3:
        return {"contrast":"corr", "x":label_x, "y":label_y, "n":len(d), "stat":np.nan, "p":np.nan, "note":"<3 valid pairs"}
    r, p = stats.pearsonr(d["x"], d["y"])
    return {"contrast":"corr", "x":label_x, "y":label_y, "n":len(d), "stat":r, "p":p, "note":"pearson r"}

def assoc_binary(y, group, label_y, label_group):
    d = pd.DataFrame({"y":y, "g":group}).dropna()
    if d["g"].nunique() < 2 or len(d) < 3:
        return {"contrast":"t-test", "y":label_y, "group":label_group, "n":len(d), "stat":np.nan, "p":np.nan, "note":"<2 groups or <3 samples"}
    g0 = d.loc[d["g"]==0, "y"].values
    g1 = d.loc[d["g"]==1, "y"].values
    if len(g0) < 2 or len(g1) < 2:
        return {"contrast":"t-test", "y":label_y, "group":label_group, "n":len(d), "stat":np.nan, "p":np.nan, "note":"too few per group"}
    t, p = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
    return {"contrast":"t-test", "y":label_y, "group":label_group, "n":len(d), "stat":t, "p":p, "note":"Welch t-test"}

results = []
for col in ["BAG_AB","cBAG_AB","Delta_BAG_AB","Delta_cBAG_AB"]:
    if col in meta_merged.columns:
        results.append(assoc_binary(meta_merged[col], meta_merged["Risk_Diabetes_AB"], col, "Risk_Diabetes_AB"))
        results.append(assoc_continuous(meta_merged["BW_HBA1c_num"], meta_merged[col], "BW_HBA1c", col))

assoc_df = pd.DataFrame(results)
assoc_df.to_csv(ASSOC_OUT, index=False)
print(f"[OK] Association summary written: {ASSOC_OUT}")

# Optional peek
print("\n=== Association summary (head) ===")
print(assoc_df.head(20))


# ============================ PLOTS ============================
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(os.path.dirname(META_CSV), "HABS_AB_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def scatter_with_trend(x, y, xlabel, ylabel, title, filename):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 2:
        print(f"[Plot skip] {title}: <2 points")
        return
    reg = LinearRegression().fit(d[["x"]].values, d["y"].values)
    xmin, xmax = float(d["x"].min()), float(d["x"].max())
    yline = reg.predict(np.array([[xmin], [xmax]]))
    plt.figure(figsize=(7.5, 6))
    plt.scatter(d["x"], d["y"], alpha=0.85, edgecolors="k")
    plt.axhline(0, ls="--", lw=1)
    plt.plot([xmin, xmax], yline, lw=2, label=f"y = {reg.coef_[0]:.3f}x + {reg.intercept_:.2f}")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend(loc="upper left"); plt.grid(True); plt.tight_layout()
    out = os.path.join(PLOT_DIR, filename)
    plt.savefig(out, dpi=300); plt.close()
    print(f"[Saved] {out}")

def box_by_group(y, g, ylabel, title, filename, labels=("No", "Yes")):
    d = pd.DataFrame({"y": y, "g": g}).dropna()
    if d["g"].nunique() < 2:
        print(f"[Plot skip] {title}: <2 groups")
        return
    groups = [d.loc[d["g"] == 0, "y"].values, d.loc[d["g"] == 1, "y"].values]
    plt.figure(figsize=(7.5, 6))
    plt.boxplot(groups, labels=labels, showmeans=True)
    plt.axhline(0, ls="--", lw=1)
    plt.ylabel(ylabel); plt.title(title); plt.tight_layout()
    out = os.path.join(PLOT_DIR, filename)
    plt.savefig(out, dpi=300); plt.close()
    print(f"[Saved] {out}")

def hist_plot(values, title, filename, xlabel):
    v = pd.Series(values).dropna()
    if len(v) < 2:
        print(f"[Plot skip] {title}: <2 points")
        return
    plt.figure(figsize=(7.5, 6))
    plt.hist(v, bins=30)
    plt.axvline(0, ls="--", lw=1)
    plt.xlabel(xlabel); plt.ylabel("Count"); plt.title(title); plt.tight_layout()
    out = os.path.join(PLOT_DIR, filename)
    plt.savefig(out, dpi=300); plt.close()
    print(f"[Saved] {out}")

# ---------- Timepoint-level (y0 + y2 rows) ----------
if {"Real_Age_AB","BAG_AB"}.issubset(meta_merged.columns):
    scatter_with_trend(meta_merged["Real_Age_AB"], meta_merged["BAG_AB"],
                       "Real Age (years)", "BAG_AB (years)",
                       "BAG_AB vs Real Age", "BAG_AB_vs_age.png")
if {"Real_Age_AB","cBAG_AB"}.issubset(meta_merged.columns):
    scatter_with_trend(meta_merged["Real_Age_AB"], meta_merged["cBAG_AB"],
                       "Real Age (years)", "cBAG_AB (years)",
                       "cBAG_AB vs Real Age", "cBAG_AB_vs_age.png")

# Distributions
if "BAG_AB" in meta_merged.columns:
    hist_plot(meta_merged["BAG_AB"], "Distribution of BAG_AB", "hist_BAG_AB.png", "BAG_AB (years)")
if "cBAG_AB" in meta_merged.columns:
    hist_plot(meta_merged["cBAG_AB"], "Distribution of cBAG_AB", "hist_cBAG_AB.png", "cBAG_AB (years)")

# By diabetes risk (timepoint-level)
if {"BAG_AB","Risk_Diabetes_AB"}.issubset(meta_merged.columns):
    box_by_group(meta_merged["BAG_AB"], meta_merged["Risk_Diabetes_AB"],
                 "BAG_AB (years)", "BAG_AB by Risk_Diabetes", "box_BAG_AB_by_risk.png")
if {"cBAG_AB","Risk_Diabetes_AB"}.issubset(meta_merged.columns):
    box_by_group(meta_merged["cBAG_AB"], meta_merged["Risk_Diabetes_AB"],
                 "cBAG_AB (years)", "cBAG_AB by Risk_Diabetes", "box_cBAG_AB_by_risk.png")

# HBA1c associations (timepoint-level)
if {"BW_HBA1c_num","BAG_AB"}.issubset(meta_merged.columns):
    scatter_with_trend(meta_merged["BW_HBA1c_num"], meta_merged["BAG_AB"],
                       "HBA1c (%)", "BAG_AB (years)",
                       "BAG_AB vs HBA1c", "BAG_AB_vs_HBA1c.png")
if {"BW_HBA1c_num","cBAG_AB"}.issubset(meta_merged.columns):
    scatter_with_trend(meta_merged["BW_HBA1c_num"], meta_merged["cBAG_AB"],
                       "HBA1c (%)", "cBAG_AB (years)",
                       "cBAG_AB vs HBA1c", "cBAG_AB_vs_HBA1c.png")

# ---------- Delta plots (root-level: one row per SubjectRoot) ----------
# Baseline age (y0) per root + deltas
if {"SubjectRoot","Real_Age_AB"}.issubset(meta_merged.columns):
    y0_rows = meta_merged[meta_merged["Subject_ID"].str.contains("y0", case=False, na=False)].copy()
    base_age = y0_rows.groupby("SubjectRoot", as_index=False)["Real_Age_AB"].first()
    hba1c_y0 = y0_rows.groupby("SubjectRoot", as_index=False)["BW_HBA1c_num"].first() if "BW_HBA1c_num" in y0_rows.columns else pd.DataFrame(columns=["SubjectRoot","BW_HBA1c_num"])
    deltas_root = meta_merged.groupby("SubjectRoot", as_index=False)[["Delta_BAG_AB","Delta_cBAG_AB"]].first()
    risk_root = meta_merged.groupby("SubjectRoot", as_index=False)["Risk_Diabetes_AB"].max() if "Risk_Diabetes_AB" in meta_merged.columns else pd.DataFrame(columns=["SubjectRoot","Risk_Diabetes_AB"])

    root_df = base_age.merge(deltas_root, on="SubjectRoot", how="left") \
                      .merge(hba1c_y0, on="SubjectRoot", how="left") \
                      .merge(risk_root, on="SubjectRoot", how="left")

    if {"Real_Age_AB","Delta_BAG_AB"}.issubset(root_df.columns):
        scatter_with_trend(root_df["Real_Age_AB"], root_df["Delta_BAG_AB"],
                           "Baseline Real Age (y0, years)", "ΔBAG_AB (y2 − y0, years)",
                           "ΔBAG_AB vs Baseline Age", "Delta_BAG_AB_vs_baseline_age.png")
        hist_plot(root_df["Delta_BAG_AB"], "Distribution of ΔBAG_AB", "hist_Delta_BAG_AB.png", "ΔBAG_AB (years)")
    if {"Real_Age_AB","Delta_cBAG_AB"}.issubset(root_df.columns):
        scatter_with_trend(root_df["Real_Age_AB"], root_df["Delta_cBAG_AB"],
                           "Baseline Real Age (y0, years)", "ΔcBAG_AB (y2 − y0, years)",
                           "ΔcBAG_AB vs Baseline Age", "Delta_cBAG_AB_vs_baseline_age.png")
        hist_plot(root_df["Delta_cBAG_AB"], "Distribution of ΔcBAG_AB", "hist_Delta_cBAG_AB.png", "ΔcBAG_AB (years)")
    if {"Delta_BAG_AB","Risk_Diabetes_AB"}.issubset(root_df.columns):
        box_by_group(root_df["Delta_BAG_AB"], root_df["Risk_Diabetes_AB"],
                     "ΔBAG_AB (years)", "ΔBAG_AB by Risk_Diabetes", "box_Delta_BAG_AB_by_risk.png")
    if {"Delta_cBAG_AB","Risk_Diabetes_AB"}.issubset(root_df.columns):
        box_by_group(root_df["Delta_cBAG_AB"], root_df["Risk_Diabetes_AB"],
                     "ΔcBAG_AB (years)", "ΔcBAG_AB by Risk_Diabetes", "box_Delta_cBAG_AB_by_risk.png")

print(f"[OK] Plots saved to: {PLOT_DIR}")
