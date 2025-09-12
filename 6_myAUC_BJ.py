#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 16:12:28 2025

@author: bas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROC AUC of IMH_Diabetes for brain-age metrics.

Input (enriched file with AB cols):
  ENRICHED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_with_PredictedAgeAB_BAG.csv"

Outputs (to .../results/associations_AB):
  - auc_IMH_Diabetes_AB.csv
  - roc_IMH_Diabetes_all.png
  - roc_IMH_Diabetes_<metric>.png (per metric)
"""

import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

# ---------- CONFIG ----------
ENRICHED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_with_PredictedAgeAB_BAG.csv"
OUT_DIR = os.path.join("/home/bas/Desktop/MyData/harmonization/HABS/results/", "associations_AB")
os.makedirs(OUT_DIR, exist_ok=True)

# Metrics to evaluate (will auto-skip if missing in file)
CANDIDATE_METRICS = [
    "BAG_AB", "cBAG_AB",
    "PredictedAge_AB", "PredictedAge_corrected_AB",
    "Delta_BAG_AB", "Delta_cBAG_AB",
]

BOOTSTRAP_N = 1000  # 1000 is a good default; you can lower to 200 to speed up

# ---------- HELPERS ----------
def coerce_num(s): return pd.to_numeric(s, errors="coerce")

def clean_diabetes_flag(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    yes = {"1","yes","y","true","t","pos","positive","dm","diabetes","type2","type 2","typei","type i","typeii","type ii"}
    no  = {"0","no","n","false","f","neg","negative"}
    if s in yes: return 1.0
    if s in no:  return 0.0
    try:
        v = float(s); return 1.0 if v > 0 else 0.0
    except: return np.nan

def find_col(df, candidates):
    norm = {re.sub(r'[^a-z0-9]+','', c.lower()): c for c in df.columns}
    for c in candidates:
        k = re.sub(r'[^a-z0-9]+','', c.lower())
        if k in norm: return norm[k]
    return None

def prepare_metric_vectors(df, metric, diab_bin_col, subj_root_col=None):
    """Return aligned y_true (0/1) and scores for a metric. Dedup delta metrics by SubjectRoot."""
    if metric.startswith("Delta_") and subj_root_col and subj_root_col in df.columns:
        sub = df[[subj_root_col, metric, diab_bin_col]].copy()
        sub[metric] = coerce_num(sub[metric])
        sub[diab_bin_col] = sub[diab_bin_col].apply(clean_diabetes_flag)
        sub = sub.dropna(subset=[metric, diab_bin_col]).groupby(subj_root_col, as_index=False).first()
    else:
        sub = df[[metric, diab_bin_col]].copy()
        sub[metric] = coerce_num(sub[metric])
        sub[diab_bin_col] = sub[diab_bin_col].apply(clean_diabetes_flag)
        sub = sub.dropna(subset=[metric, diab_bin_col])

    # Need both classes for AUC
    if sub[diab_bin_col].nunique() < 2:
        return None, None
    return sub[diab_bin_col].astype(int).values, sub[metric].astype(float).values

def auc_ci_bootstrap(y, s, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y); aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        # must have both classes in bootstrap sample
        if len(np.unique(yb)) < 2: continue
        aucs.append(roc_auc_score(yb, sb))
    if not aucs:
        return np.nan, np.nan
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)

def youden_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), float(tpr[k]), float(1 - fpr[k]), fpr, tpr

def plot_roc(fpr, tpr, label, out_png):
    plt.figure(figsize=(5.2, 5.2))
    plt.plot(fpr, tpr, lw=2, label=label)
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlim([0,1]); plt.ylim([0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: IMH_Diabetes")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ---------- MAIN ----------
def main():
    df = pd.read_csv(ENRICHED_CSV)

    # Locate IMH_Diabetes and SubjectRoot (for delta de-dup)
    diab_col = find_col(df, ["IMH_Diabetes"])
    if diab_col is None:
        raise ValueError("IMH_Diabetes column not found in input CSV.")
    subj_root_col = find_col(df, ["SubjectRoot","Subject_Root"])
    if subj_root_col is None:
        sid_col = find_col(df, ["Subject_ID","Subject","MRI_Exam_fixed","DWI"])
        if sid_col:
            sr = df[sid_col].astype(str).str.replace(r"\.0$","",regex=True)
            df["SubjectRoot"] = sr.str.replace(r"([_\-]?y[0-9]+)$","",regex=True)
            subj_root_col = "SubjectRoot"

    # Pick only metrics present
    metrics = [m for m in CANDIDATE_METRICS if m in df.columns]
    if not metrics:
        raise ValueError(f"No expected metrics present. Looked for: {CANDIDATE_METRICS}")

    # Compute AUCs + CIs + optimal threshold; make individual ROC plots
    rows = []
    overlay = []

    for m in metrics:
        y, s = prepare_metric_vectors(df, m, diab_col, subj_root_col=subj_root_col)
        if y is None:
            rows.append({"metric": m, "n": 0, "n0": 0, "n1": 0,
                         "AUC": np.nan, "CI95_lo": np.nan, "CI95_hi": np.nan,
                         "thr_Youden": np.nan, "sens": np.nan, "spec": np.nan})
            continue

        n = len(y); n1 = int((y==1).sum()); n0 = n - n1
        try:
            auc_val = roc_auc_score(y, s)
        except Exception:
            auc_val = np.nan

        lo, hi = auc_ci_bootstrap(y, s, n_boot=BOOTSTRAP_N)

        thr, sens, spec, fpr, tpr = youden_threshold(y, s)
        label = f"{m}: AUC={auc_val:.3f} [{lo:.3f},{hi:.3f}]"
        out_png = os.path.join(OUT_DIR, f"roc_IMH_Diabetes_{m}.png")
        plot_roc(fpr, tpr, label, out_png)

        overlay.append((fpr, tpr, label))
        rows.append({
            "metric": m, "n": n, "n0": n0, "n1": n1,
            "AUC": float(auc_val), "CI95_lo": lo, "CI95_hi": hi,
            "thr_Youden": thr, "sens": sens, "spec": spec
        })

    # Save summary CSV
    out_csv = os.path.join(OUT_DIR, "auc_IMH_Diabetes_AB.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Combined overlay ROC
    if overlay:
        plt.figure(figsize=(6,6))
        for fpr, tpr, lab in overlay:
            plt.plot(fpr, tpr, lw=2, label=lab)
        plt.plot([0,1],[0,1],'k--', lw=1)
        plt.xlim([0,1]); plt.ylim([0,1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC (IMH_Diabetes) — all metrics")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "roc_IMH_Diabetes_all.png"), dpi=300)
        plt.close()

    print(f"[OK] AUC summary → {out_csv}")
    print(f"[OK] ROC PNGs → {OUT_DIR}")

if __name__ == "__main__":
    main()
