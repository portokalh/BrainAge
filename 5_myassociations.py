#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean + re-enrich AB brain-age columns and run associations.

INPUT
  ENRICHED_IN:
    /home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_with_PredictedAgeAB_BAG.csv

OUTPUTS
  ENRICHED_OUT (cleaned/standardized/new bias correction & deltas):
    /home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_AB_enriched_v2.csv

  Associations (CSV + PNG) in:
    /home/bas/Desktop/MyData/harmonization/HABS/results/associations_AB_v2/
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# --------------------- CONFIG ---------------------
ENRICHED_IN  = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_with_PredictedAgeAB_BAG.csv"
META_DIR     = os.path.dirname(ENRICHED_IN)
ENRICHED_OUT = os.path.join(META_DIR, "HABS_metadata_AB_enriched_v2.csv")

RES_DIR      = "/home/bas/Desktop/MyData/harmonization/HABS/results"
OUT_DIR      = os.path.join(RES_DIR, "associations_AB_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# Metrics we want standardized in the final enriched file
CANON_COLS = [
    "PredictedAge_AB",
    "PredictedAge_corrected_AB",
    "BAG_AB",
    "cBAG_AB",
    "Delta_BAG_AB",
    "Delta_cBAG_AB",
]

# --------------------- UTILS ---------------------
def pick_first_nonnull(df, candidates, new_name):
    """
    From a list of candidate column names (including _x/_y variants),
    pick (create) a single column 'new_name' taking the first candidate that exists.
    Preference order: unsuffixed -> *_x -> *_y
    """
    ordered = []
    # unsuffixed first
    ordered.extend([c for c in candidates if c in df.columns and not c.endswith(("_x","_y"))])
    # then _x, then _y
    ordered.extend([c for c in candidates if c in df.columns and c.endswith("_x")])
    ordered.extend([c for c in candidates if c in df.columns and c.endswith("_y")])

    if not ordered:
        # create empty column
        df[new_name] = np.nan
        return df

    src = ordered[0]
    df[new_name] = df[src]
    return df

def coerce_num(s):
    return pd.to_numeric(s, errors="coerce")

def clean_binary_flag(x):
    if pd.isna(x): return np.nan
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

def ensure_id_cols(df):
    """Ensure MRI_Exam_fixed and SubjectRoot exist."""
    if "MRI_Exam_fixed" not in df.columns:
        if "DWI" in df.columns:
            df["MRI_Exam_fixed"] = df["DWI"].astype(str)
        elif "Subject_ID" in df.columns:
            df["MRI_Exam_fixed"] = df["Subject_ID"].astype(str)
        else:
            raise KeyError("Need MRI_Exam_fixed / DWI / Subject_ID to identify subjects.")
    df["MRI_Exam_fixed"] = df["MRI_Exam_fixed"].astype(str)
    # SubjectRoot = strip trailing _y0/_y2
    df["SubjectRoot"] = df["MRI_Exam_fixed"].str.replace(r"([_\-]?y[0-9]+)$", "", regex=True)
    return df

def bias_correct(df):
    """
    Recompute bias-corrected brain age using Age:
      Fit PredictedAge_AB ~ a + b * Age  (drop NA)
      PredictedAge_corrected_AB = PredictedAge_AB - (a + b*Age) + Age
      cBAG_AB = PredictedAge_corrected_AB - Age = PredictedAge_AB - (a + b*Age)
    """
    if "PredictedAge_AB" not in df.columns:
        df["PredictedAge_AB"] = np.nan
    if "Age" not in df.columns:
        raise KeyError("Column 'Age' not found; needed for bias correction.")

    age = coerce_num(df["Age"])
    y   = coerce_num(df["PredictedAge_AB"])
    ok  = (~age.isna()) & (~y.isna())

    if ok.sum() < 3:
        # Not enough data to fit; keep existing if any
        if "PredictedAge_corrected_AB" not in df.columns:
            df["PredictedAge_corrected_AB"] = np.nan
        if "cBAG_AB" not in df.columns:
            df["cBAG_AB"] = np.nan
        return df

    reg = LinearRegression().fit(age[ok].values.reshape(-1,1), y[ok].values)
    a, b = float(reg.intercept_), float(reg.coef_[0])

    expected = a + b * age
    df["PredictedAge_corrected_AB"] = y - expected + age
    df["cBAG_AB"] = df["PredictedAge_corrected_AB"] - age
    # Plain BAG too (if missing)
    if "BAG_AB" not in df.columns:
        df["BAG_AB"] = y - age
    return df

def compute_deltas(df):
    """Compute Delta_BAG_AB and Delta_cBAG_AB per SubjectRoot (y2 - y0)."""
    # identify timepoint from MRI_Exam_fixed
    tp = df["MRI_Exam_fixed"].str.extract(r"(y\d+)$", expand=False).str.lower()
    df["_Timepoint"] = tp.fillna("y0")  # default if none
    # reduce per SubjectRoot,timepoint (first non-null)
    keep = ["SubjectRoot", "_Timepoint", "BAG_AB", "cBAG_AB"]
    g = df.dropna(subset=["SubjectRoot"]).groupby(["SubjectRoot", "_Timepoint"], as_index=False).agg(
        BAG_AB = ("BAG_AB", "mean"),
        cBAG_AB = ("cBAG_AB","mean")
    )
    y0 = g[g["_Timepoint"]=="y0"].set_index("SubjectRoot")
    y2 = g[g["_Timepoint"]=="y2"].set_index("SubjectRoot")
    aligned = y0.join(y2, how="inner", lsuffix="_y0", rsuffix="_y2")
    deltas = pd.DataFrame({
        "SubjectRoot": aligned.index,
        "Delta_BAG_AB":  aligned["BAG_AB_y2"]  - aligned["BAG_AB_y0"],
        "Delta_cBAG_AB": aligned["cBAG_AB_y2"] - aligned["cBAG_AB_y0"],
    }).reset_index(drop=True)

    # map back to all rows by SubjectRoot
    df = df.merge(deltas, on="SubjectRoot", how="left")
    df.drop(columns=["_Timepoint"], errors="ignore", inplace=True)
    return df

# --------- plotting helpers (with p-values on plots) ----------
try:
    import seaborn as sns
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

def violin_ttest(df, value_col, group_col, out_png, title_prefix=""):
    d = df[[value_col, group_col]].dropna()
    # Keep only 0/1 groups
    d[group_col] = d[group_col].astype(float)
    d = d[d[group_col].isin([0.0, 1.0])]
    if d[group_col].nunique() < 2 or len(d) < 3:
        return np.nan, np.nan

    g0 = d.loc[d[group_col]==0.0, value_col].astype(float).values
    g1 = d.loc[d[group_col]==1.0, value_col].astype(float).values
    if len(g0) < 2 or len(g1) < 2:
        t, p = np.nan, np.nan
    else:
        t, p = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")

    plt.figure(figsize=(6,5))
    if HAVE_SNS:
        sns.violinplot(data=d, x=group_col, y=value_col, inner="box", cut=0)
        sns.stripplot(data=d, x=group_col, y=value_col, color="k", alpha=0.25)
        plt.xticks([0,1], ["0","1"])
    else:
        groups = [g0, g1]
        plt.violinplot(groups, showmeans=False, showmedians=True)
        plt.xticks([1,2], ["0","1"])
    ttl = f"{title_prefix}{value_col} by {group_col}\nWelch t={t:.3f}, p={p:.3g}"
    plt.title(ttl)
    plt.xlabel(group_col); plt.ylabel(value_col)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
    return float(t) if not np.isnan(t) else np.nan, float(p) if not np.isnan(p) else np.nan

def violin_anova(df, value_col, group_col, out_png, title_prefix=""):
    d = df[[value_col, group_col]].dropna()
    if d[group_col].nunique() < 2 or len(d) < 3:
        return np.nan, np.nan, np.nan
    # Prepare groups
    groups = [g[value_col].astype(float).values for _, g in d.groupby(group_col)]
    # Require at least 2 obs per group
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    F, p = stats.f_oneway(*groups)
    # eta-squared approx (effect size): SS_between/SS_total from ANOVA isn’t trivial here;
    # we’ll just report F,p and group counts.
    plt.figure(figsize=(7,5))
    if HAVE_SNS:
        sns.violinplot(data=d, x=group_col, y=value_col, inner="box", cut=0)
        sns.stripplot(data=d, x=group_col, y=value_col, color="k", alpha=0.25)
    else:
        # simple fallback: boxplot
        d.boxplot(column=value_col, by=group_col)
        plt.suptitle("")
    ttl = f"{title_prefix}{value_col} by {group_col}\nANOVA F={F:.3f}, p={p:.3g}"
    plt.title(ttl)
    plt.xlabel(group_col); plt.ylabel(value_col)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
    return float(F), float(p), int(d[group_col].nunique())

def scatter_corr(x, y, xlabel, ylabel, out_png, title_prefix=""):
    d = pd.DataFrame({"x":x, "y":y}).dropna()
    if len(d) < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(d["x"], d["y"])
    m, b = np.polyfit(d["x"], d["y"], 1)
    xs = np.array([d["x"].min(), d["x"].max()])
    ys = m*xs + b
    plt.figure(figsize=(6,5))
    plt.scatter(d["x"], d["y"], alpha=0.6, edgecolors="k")
    plt.plot(xs, ys, linewidth=2, alpha=0.8)
    ttl = f"{title_prefix}{ylabel} vs {xlabel}\nR={r:.3f}, R²={r**2:.3f}, p={p:.3g}"
    plt.title(ttl); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
    return float(r), float(p)

# --------------------- MAIN ---------------------
def main():
    # ---------- Load ----------
    df = pd.read_csv(ENRICHED_IN)
    df = ensure_id_cols(df)

    # ---------- Normalize/choose AB columns (resolve _x/_y) ----------
    # Build candidate lists per canonical name
    cand_map = {
        "PredictedAge_AB": [
            "PredictedAge_AB","Predicted_Age_AB",
            "PredictedAge_AB_x","Predicted_Age_AB_x",
            "PredictedAge_AB_y","Predicted_Age_AB_y"
        ],
        "BAG_AB": [
            "BAG_AB","BAG_AB_x","BAG_AB_y"
        ],
        "cBAG_AB": [
            "cBAG_AB","cBAG_AB_x","cBAG_AB_y"
        ],
        "PredictedAge_corrected_AB": [
            "PredictedAge_corrected_AB",
            "Predicted_Age_corrected_AB",
            "PredictedAge_corrected_AB_x","Predicted_Age_corrected_AB_x",
            "PredictedAge_corrected_AB_y","Predicted_Age_corrected_AB_y"
        ],
        "Delta_BAG_AB": [
            "Delta_BAG_AB","Delta_BAG_AB_x","Delta_BAG_AB_y"
        ],
        "Delta_cBAG_AB": [
            "Delta_cBAG_AB","Delta_cBAG_AB_x","Delta_cBAG_AB_y"
        ],
    }

    for new_name, cands in cand_map.items():
        df = pick_first_nonnull(df, cands, new_name)

    # Drop any leftover duplicates (_x/_y) for the same bases
    base_pat = re.compile(r'^(?:' + "|".join([re.escape(k) for k in cand_map.keys()]) + r')_[xy]$')
    drop_dups = [c for c in df.columns if base_pat.match(c)]
    if drop_dups:
        df.drop(columns=drop_dups, inplace=True, errors="ignore")

    # ---------- Bias-correct using Age (recompute robustly) ----------
    if "Age" not in df.columns:
        raise KeyError("Column 'Age' is required but not found.")
    df["Age"] = coerce_num(df["Age"])
    df["PredictedAge_AB"] = coerce_num(df["PredictedAge_AB"])

    df = bias_correct(df)

    # ---------- Recompute deltas by SubjectRoot ----------
    df = compute_deltas(df)

    # ---------- Save fresh enriched file ----------
    df.to_csv(ENRICHED_OUT, index=False)
    print(f"[OK] Enriched file written: {ENRICHED_OUT}")

    # ---------- Associations ----------
    # Prepare variables
    # IMH_Diabetes
    diab_col = "IMH_Diabetes" if "IMH_Diabetes" in df.columns else None
    # IMH_Alzheimers (note: name without apostrophe in your metadata)
    alz_col  = "IMH_Alzheimers" if "IMH_Alzheimers" in df.columns else None
    # CDX_Cog (categorical)
    cog_col  = "CDX_Cog" if "CDX_Cog" in df.columns else None
    # HBA1c
    hba1c_col = "BW_HBA1c" if "BW_HBA1c" in df.columns else None

    # Metrics to test
    TT_METRICS = ["BAG_AB","cBAG_AB","PredictedAge_AB","PredictedAge_corrected_AB","Delta_BAG_AB","Delta_cBAG_AB"]
    CORR_METRICS = ["BAG_AB","cBAG_AB","PredictedAge_AB","PredictedAge_corrected_AB"]

    # Binary flags clean
    if diab_col:
        df["IMH_Diabetes_bin"] = df[diab_col].apply(clean_binary_flag)
    if alz_col:
        df["IMH_Alzheimers_bin"] = df[alz_col].apply(clean_binary_flag)

    # HBA1c numeric
    if hba1c_col:
        df["BW_HBA1c_num"] = coerce_num(df[hba1c_col])

    # ---- T-tests + violins: IMH_Diabetes, IMH_Alzheimers ----
    ttest_rows = []
    for metric in TT_METRICS:
        if metric not in df.columns: 
            continue
        # IMH_Diabetes
        if diab_col:
            t, p = violin_ttest(df, metric, "IMH_Diabetes_bin",
                                os.path.join(OUT_DIR, f"violin_IMH_Diabetes_{metric}.png"),
                                title_prefix="")
            # For delta metrics, deduplicate per SubjectRoot to avoid double counting
            # (We already averaged when computing deltas; current dataframe carries one delta per root repeated across visits)
            ttest_rows.append({"outcome":metric, "group":"IMH_Diabetes", "test":"Welch t", "t":t, "p":p})
        # IMH_Alzheimers
        if alz_col:
            t, p = violin_ttest(df, metric, "IMH_Alzheimers_bin",
                                os.path.join(OUT_DIR, f"violin_IMH_Alzheimers_{metric}.png"),
                                title_prefix="")
            ttest_rows.append({"outcome":metric, "group":"IMH_Alzheimers", "test":"Welch t", "t":t, "p":p})

    if ttest_rows:
        pd.DataFrame(ttest_rows).to_csv(os.path.join(OUT_DIR, "ttests_IMH_Diabetes_Alzheimers_AB.csv"), index=False)

    # ---- ANOVA + violin: CDX_Cog ----
    if cog_col and cog_col in df.columns:
        df[cog_col] = coerce_num(df[cog_col]).round(0)  # ensure categorical ints 0,1,2,3
        anova_rows = []
        for metric in TT_METRICS:
            if metric not in df.columns: 
                continue
            F, p, k = violin_anova(df, metric, cog_col,
                                   os.path.join(OUT_DIR, f"violin_{cog_col}_{metric}.png"),
                                   title_prefix="")
            anova_rows.append({"outcome":metric, "group":cog_col, "test":"ANOVA", "F":F, "p":p, "levels":k})
        pd.DataFrame(anova_rows).to_csv(os.path.join(OUT_DIR, "anova_CDX_Cog_AB.csv"), index=False)

    # ---- Correlations: BW_HBA1c ----
    if hba1c_col:
        corr_rows = []
        for metric in CORR_METRICS:
            if metric not in df.columns: 
                continue
            r, p = scatter_corr(df["BW_HBA1c_num"], coerce_num(df[metric]),
                                xlabel="BW_HBA1c", ylabel=metric,
                                out_png=os.path.join(OUT_DIR, f"scatter_HBA1c_{metric}.png"),
                                title_prefix="")
            corr_rows.append({"metric":metric, "n": int(pd.DataFrame({"x":df["BW_HBA1c_num"], "y":df[metric]}).dropna().shape[0]),
                              "r": r, "R2": (r*r if not np.isnan(r) else np.nan), "p": p})
        pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "correlations_HBA1c_AB.csv"), index=False)

    print(f"[OK] Association CSV/PNGs saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()
