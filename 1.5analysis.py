#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:10:15 2025

@author: bas
"""

# ============================== IMPORTS ==============================
import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ============================ CONFIG ============================
META_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_AB_BJ.csv"
OUT_DIR  = os.path.dirname(META_CSV)
OUT_XSEC = os.path.join(OUT_DIR, "HABS_AB_OLS_cross_sectional.csv")
OUT_DELTA= os.path.join(OUT_DIR, "HABS_AB_OLS_deltas.csv")

# ============================ LOAD =============================
df = pd.read_csv(META_CSV)

# -------- helpers
def norm_sex(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"f","female","woman","w"}: return 1
    if s in {"m","male","man"}: return 0
    try:
        v = float(s); 
        if v in (0.0,1.0): return v
    except: pass
    return np.nan

def edu_years(row):
    # try numeric "ID_Education" first
    if "ID_Education" in row and pd.notna(row["ID_Education"]):
        try:
            return float(row["ID_Education"])
        except: pass
    # fallback: degree → years (very rough)
    deg = None
    for c in ["ID_Education_Degree", "Education_Degree", "Highest_Degree"]:
        if c in row and pd.notna(row[c]):
            deg = str(row[c]).strip().lower()
            break
    if deg:
        # simple map; tweak to your coding if needed
        if any(k in deg for k in ["<hs","less than high", "no high"]): return 10
        if "high" in deg: return 12
        if "ged" in deg: return 12
        if "some college" in deg or "assoc" in deg: return 14
        if "bachelor" in deg or "ba" in deg or "bs" in deg: return 16
        if "master" in deg or "ma" in deg or "ms" in deg: return 18
        if "phd" in deg or "md" in deg or "do" in deg or "jd" in deg: return 20
    return np.nan

def first_nonnull(x):
    return x.dropna().iloc[0] if x.notna().any() else np.nan

def ols_table(df_in, y_col, x_cols, model_name):
    d = df_in[[y_col] + x_cols].dropna().copy()
    if d.shape[0] < len(x_cols) + 3:
        return pd.DataFrame([{
            "model": model_name, "term": "(insufficient data)", "beta": np.nan, "se": np.nan,
            "t": np.nan, "p": np.nan, "n": d.shape[0], "r2": np.nan, "adj_r2": np.nan
        }])
    X = sm.add_constant(d[x_cols])
    y = d[y_col]
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    out = []
    for term, est, se, tval, pval in zip(fit.params.index, fit.params.values, fit.bse.values, fit.tvalues.values, fit.pvalues.values):
        out.append({
            "model": model_name, "term": term, "beta": est, "se": se, "t": tval, "p": pval,
            "n": int(fit.nobs), "r2": fit.rsquared, "adj_r2": fit.rsquared_adj
        })
    return pd.DataFrame(out)

# ======================= PREP COVARIATES (timepoint-level) =======================
# Sex
df["Sex_Female"] = df["Sex"].apply(norm_sex) if "Sex" in df.columns else np.nan
# Education (years)
df["EduYears"] = df.apply(edu_years, axis=1)

# Ensure numeric HBA1c
if "BW_HBA1c_num" not in df.columns:
    df["BW_HBA1c_num"] = pd.to_numeric(df.get("BW_HBA1c", np.nan), errors="coerce")

# Ensure diabetes risk (from prior step)
if "Risk_Diabetes_AB" not in df.columns:
    # quick fallback: try IMH_Diabetes or CDX_Diabetes (1/0)
    for src in ["IMH_Diabetes","CDX_Diabetes"]:
        if src in df.columns:
            df["Risk_Diabetes_AB"] = pd.to_numeric(df[src], errors="coerce")
            break

# ======================= CROSS-SECTIONAL OLS =======================
# Use timepoint-level rows (y0 and y2 both appear), outcomes need to exist
xsec_models = []
covs = ["Real_Age_AB", "Sex_Female", "EduYears"]  # adjust for age-at-scan (already +2y for y2), sex, education
# Add risk and HBA1c as predictors
pred_sets = [
    (["Risk_Diabetes_AB"] + covs,            "XSEC: BAG ~ Risk + covars",        "BAG_AB"),
    (["BW_HBA1c_num"] + covs,                "XSEC: BAG ~ HBA1c + covars",       "BAG_AB"),
    (["Risk_Diabetes_AB","BW_HBA1c_num"] + covs, "XSEC: BAG ~ Risk+HBA1c + covars", "BAG_AB"),
    (["Risk_Diabetes_AB"] + covs,            "XSEC: cBAG ~ Risk + covars",       "cBAG_AB"),
    (["BW_HBA1c_num"] + covs,                "XSEC: cBAG ~ HBA1c + covars",      "cBAG_AB"),
    (["Risk_Diabetes_AB","BW_HBA1c_num"] + covs, "XSEC: cBAG ~ Risk+HBA1c + covars","cBAG_AB"),
]

# drop huge outliers if needed (optional)
df_xsec = df.copy()

for xs, name, ycol in pred_sets:
    # keep only rows where outcome exists
    if ycol not in df_xsec.columns:
        continue
    # ensure existence of predictors
    use_cols = [c for c in xs if c in df_xsec.columns]
    if len(use_cols) != len(xs):
        # skip if key predictors missing
        if not set(["Risk_Diabetes_AB","BW_HBA1c_num"]).intersection(xs):
            continue
    res = ols_table(df_xsec, ycol, use_cols, name + f" -> {ycol}")
    xsec_models.append(res)

xsec_out = pd.concat(xsec_models, ignore_index=True) if xsec_models else pd.DataFrame()
if not xsec_out.empty:
    xsec_out.to_csv(OUT_XSEC, index=False)
    print(f"✅ Cross-sectional OLS saved: {OUT_XSEC}")
else:
    print("⚠️ Cross-sectional outcomes not found; skipped.")

# ======================= DELTA OLS (root-level) =======================
# Build a root-level table so each subject contributes once
df["SubjectRoot"] = df["Subject_ID"].astype(str).str.replace(r"([_\-]?y[02])$", "", regex=True)
df["is_y0"] = df["Subject_ID"].astype(str).str.contains("y0", case=False, na=False)

# For each root, collect: delta outcomes (already same on both rows), baseline age, sex, edu, risk, baseline HBA1c
grp = df.groupby("SubjectRoot", dropna=False)

delta_df = pd.DataFrame({
    "Delta_BAG_AB": grp["Delta_BAG_AB"].apply(first_nonnull) if "Delta_BAG_AB" in df.columns else np.nan,
    "Delta_cBAG_AB": grp["Delta_cBAG_AB"].apply(first_nonnull) if "Delta_cBAG_AB" in df.columns else np.nan,
    "Sex_Female": grp["Sex_Female"].apply(first_nonnull),
    "EduYears": grp["EduYears"].apply(first_nonnull),
    # Baseline age (prefer y0 row)
    "Baseline_Real_Age_AB": grp.apply(lambda g: first_nonnull(g.loc[g["is_y0"], "Real_Age_AB"]) if "Real_Age_AB" in g else np.nan),
    # Diabetes risk (OR across timepoints; if only one present, use it)
    "Risk_Diabetes_root": grp["Risk_Diabetes_AB"].max(min_count=1) if "Risk_Diabetes_AB" in df.columns else np.nan,
    # Baseline HBA1c
    "BW_HBA1c_y0": grp.apply(lambda g: first_nonnull(g.loc[g["is_y0"], "BW_HBA1c_num"]) if "BW_HBA1c_num" in g else np.nan),
}).reset_index()

# OLS on deltas with covariates
delta_models = []
delta_sets = [
    (["Risk_Diabetes_root","Baseline_Real_Age_AB","Sex_Female","EduYears"], "DELTA: ΔBAG ~ Risk + covars", "Delta_BAG_AB"),
    (["BW_HBA1c_y0","Baseline_Real_Age_AB","Sex_Female","EduYears"],        "DELTA: ΔBAG ~ HBA1c_y0 + covars", "Delta_BAG_AB"),
    (["Risk_Diabetes_root","BW_HBA1c_y0","Baseline_Real_Age_AB","Sex_Female","EduYears"],
        "DELTA: ΔBAG ~ Risk+HBA1c_y0 + covars", "Delta_BAG_AB"),

    (["Risk_Diabetes_root","Baseline_Real_Age_AB","Sex_Female","EduYears"], "DELTA: ΔcBAG ~ Risk + covars", "Delta_cBAG_AB"),
    (["BW_HBA1c_y0","Baseline_Real_Age_AB","Sex_Female","EduYears"],        "DELTA: ΔcBAG ~ HBA1c_y0 + covars", "Delta_cBAG_AB"),
    (["Risk_Diabetes_root","BW_HBA1c_y0","Baseline_Real_Age_AB","Sex_Female","EduYears"],
        "DELTA: ΔcBAG ~ Risk+HBA1c_y0 + covars", "Delta_cBAG_AB"),
]

for xs, name, ycol in delta_sets:
    if ycol not in delta_df.columns:
        continue
    use_cols = [c for c in xs if c in delta_df.columns]
    res = ols_table(delta_df, ycol, use_cols, name + f" -> {ycol}")
    delta_models.append(res)

delta_out = pd.concat(delta_models, ignore_index=True) if delta_models else pd.DataFrame()
if not delta_out.empty:
    delta_out.to_csv(OUT_DELTA, index=False)
    print(f"✅ Delta OLS saved: {OUT_DELTA}")
else:
    print("⚠️ Delta outcomes not found; skipped.")

# ======================= Quick textual summary =======================
def quick_sig(dfcoeff, alpha=0.05):
    if dfcoeff.empty: 
        return "No models run."
    sig = dfcoeff[(dfcoeff["term"]!="const") & (dfcoeff["p"]<alpha)].copy()
    if sig.empty:
        return "No predictors reached p<0.05."
    keep = sig[["model","term","beta","se","t","p","n","r2","adj_r2"]].sort_values("p")
    return keep.to_string(index=False)

print("\n===== Significant predictors (p<0.05), cross-sectional =====")
print(quick_sig(xsec_out))

print("\n===== Significant predictors (p<0.05), deltas =====")
print(quick_sig(delta_out))
