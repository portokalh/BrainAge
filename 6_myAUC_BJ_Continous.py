#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 16:18:12 2025

@author: bas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Associations with continuous BW_HBA1c:
- Pearson & Spearman correlations, linear regression (R², p)
- Concordance index (c-index)
- Optional ROC AUC using clinical cutoffs: 5.7 (prediabetes), 6.5 (diabetes)
- Scatter plots with regression lines and p annotations
- ROC curves for the two cutoffs

Input:
  ENRICHED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_with_PredictedAgeAB_BAG.csv"

Outputs (to results/associations_AB):
  - assoc_HBA1c_AB.csv
  - assoc_HBA1c_AUC_AB.csv
  - scatter_HBA1c_<metric>.png
  - roc_HBA1c_cut<cutoff>_<metric>.png and roc_HBA1c_all_cut<cutoff>.png
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------- CONFIG ----------------
ENRICHED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_with_PredictedAgeAB_BAG.csv"
OUT_DIR = os.path.join("/home/bas/Desktop/MyData/harmonization/HABS/results/", "associations_AB")
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = [
    "BAG_AB", "cBAG_AB",
    "PredictedAge_AB", "PredictedAge_corrected_AB",
    "Delta_BAG_AB", "Delta_cBAG_AB",
]

HBA1C_COL = "BW_HBA1c"
CUTS = [5.7, 6.5]   # prediabetes, diabetes

# -------------- HELPERS --------------
def coerce_num(s): return pd.to_numeric(s, errors="coerce")

def find_col(df, candidates):
    norm = {re.sub(r'[^a-z0-9]+','', c.lower()): c for c in df.columns}
    for c in candidates:
        k = re.sub(r'[^a-z0-9]+','', c.lower())
        if k in norm: return norm[k]
    return None

def c_index(y_continuous, score):
    """Harrell's C for two continuous vectors: P(concordant) among all comparable pairs."""
    y = np.asarray(y_continuous, float)
    s = np.asarray(score, float)
    n = len(y)
    if n < 3: return np.nan
    conc = 0; ties = 0; total = 0
    # O(n^2) is fine for a few hundred
    for i in range(n):
        for j in range(i+1, n):
            if np.isnan(y[i]) or np.isnan(y[j]) or np.isnan(s[i]) or np.isnan(s[j]): 
                continue
            if y[i] == y[j]:
                # same outcome -> skip (not comparable)
                continue
            total += 1
            # concordant if higher y corresponds to higher score
            if (y[i] > y[j] and s[i] > s[j]) or (y[i] < y[j] and s[i] < s[j]):
                conc += 1
            elif s[i] == s[j]:
                ties += 1
    if total == 0: return np.nan
    return (conc + 0.5 * ties) / total

def scatter_with_fit(x, y, out_png, xlabel, ylabel, title=None):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 3: 
        return False
    r, p = stats.pearsonr(d["x"], d["y"])
    m, b = np.polyfit(d["x"], d["y"], 1)
    xs = np.array([d["x"].min(), d["x"].max()])
    ys = m*xs + b
    plt.figure(figsize=(6,5))
    plt.scatter(d["x"], d["y"], alpha=0.6, edgecolors="k")
    plt.plot(xs, ys, linewidth=2, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ttl = title if title else f"{ylabel} vs {xlabel}"
    plt.title(f"{ttl}\nPearson r={r:.3f}, R²={r**2:.3f}, p={p:.3g}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return True

def prepare_series(df, metric, hba, subj_root_col):
    """Return aligned vectors for metric vs HBA1c.
       If metric is a delta, deduplicate by SubjectRoot (first non-null)."""
    if metric.startswith("Delta_") and subj_root_col and subj_root_col in df.columns:
        sub = df[[subj_root_col, metric, HBA1C_COL]].copy()
        sub[metric] = coerce_num(sub[metric])
        sub[HBA1C_COL] = coerce_num(sub[HBA1C_COL])
        sub = sub.dropna(subset=[metric, HBA1C_COL]).groupby(subj_root_col, as_index=False).first()
        return sub[HBA1C_COL].values, sub[metric].values
    else:
        x = coerce_num(hba)
        y = coerce_num(df[metric])
        ok = (~x.isna()) & (~y.isna())
        return x[ok].values, y[ok].values

# -------------- MAIN --------------
def main():
    df = pd.read_csv(ENRICHED_CSV)

    # SubjectRoot (for delta de-dup)
    subj_root_col = find_col(df, ["SubjectRoot","Subject_Root"])
    if subj_root_col is None:
        sid = find_col(df, ["Subject_ID","MRI_Exam_fixed","DWI","Subject"])
        if sid:
            sr = df[sid].astype(str).str.replace(r"\.0$","",regex=True)
            df["SubjectRoot"] = sr.str.replace(r"([_\-]?y[0-9]+)$","",regex=True)
            subj_root_col = "SubjectRoot"

    # HBA1c numeric
    if HBA1C_COL not in df.columns:
        raise ValueError(f"Column {HBA1C_COL} not found in {ENRICHED_CSV}")
    df[HBA1C_COL] = coerce_num(df[HBA1C_COL])

    # ---------- Correlations / Regression / c-index ----------
    rows = []
    for m in METRICS:
        if m not in df.columns:
            continue
        x, y = prepare_series(df, m, df[HBA1C_COL], subj_root_col)
        if len(x) < 3:
            rows.append({"metric": m, "n": len(x),
                         "pearson_r": np.nan, "pearson_p": np.nan, "R2": np.nan,
                         "spearman_rho": np.nan, "spearman_p": np.nan,
                         "c_index": np.nan})
            continue

        # Pearson / Spearman
        pr, pp = stats.pearsonr(x, y)
        srho, sp = stats.spearmanr(x, y, nan_policy="omit")

        # R² from simple linear fit
        R2 = pr**2

        # c-index (monotonic ranking)
        cidx = c_index(x, y)

        rows.append({"metric": m, "n": int(len(x)),
                     "pearson_r": float(pr), "pearson_p": float(pp), "R2": float(R2),
                     "spearman_rho": float(srho), "spearman_p": float(sp),
                     "c_index": float(cidx)})

        # Scatter with fit + p
        out_png = os.path.join(OUT_DIR, f"scatter_HBA1c_{m}.png")
        scatter_with_fit(x, y, out_png, xlabel="BW_HBA1c", ylabel=m)

    out_csv = os.path.join(OUT_DIR, "assoc_HBA1c_AB.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Saved correlations & c-index → {out_csv}")

    # ---------- Optional: ROC AUC via clinical cutoffs ----------
    auc_rows = []
    for cut in CUTS:
        y_bin_full = (df[HBA1C_COL] >= cut).astype(float)  # 1 if at/above cutoff
        overlay = []
        for m in METRICS:
            if m not in df.columns: 
                continue
            # Prepare aligned vectors (respect delta de-dup)
            if m.startswith("Delta_") and subj_root_col and subj_root_col in df.columns:
                tmp = df[[subj_root_col, m, HBA1C_COL]].copy()
                tmp[m] = coerce_num(tmp[m])
                tmp = tmp.dropna(subset=[m, HBA1C_COL]).groupby(subj_root_col, as_index=False).first()
                yb = (tmp[HBA1C_COL] >= cut).astype(int).values
                s  = tmp[m].astype(float).values
            else:
                s = coerce_num(df[m])
                yb = y_bin_full.copy()
                ok = (~s.isna()) & (~yb.isna())
                yb = yb[ok].astype(int).values
                s  = s[ok].astype(float).values

            if len(np.unique(yb)) < 2 or len(yb) < 3:
                auc_rows.append({"cutoff": cut, "metric": m, "n": len(yb),
                                 "pos": int((yb==1).sum()), "neg": int((yb==0).sum()),
                                 "AUC": np.nan, "thr_Youden": np.nan, "sens": np.nan, "spec": np.nan})
                continue

            auc = roc_auc_score(yb, s)
            fpr, tpr, thr = roc_curve(yb, s)
            youden = tpr - fpr
            k = int(np.argmax(youden))
            thr_star = float(thr[k]); sens = float(tpr[k]); spec = float(1 - fpr[k])

            auc_rows.append({"cutoff": cut, "metric": m, "n": len(yb),
                             "pos": int((yb==1).sum()), "neg": int((yb==0).sum()),
                             "AUC": float(auc), "thr_Youden": thr_star,
                             "sens": sens, "spec": spec})

            # per metric ROC
            png = os.path.join(OUT_DIR, f"roc_HBA1c_cut{str(cut).replace('.','_')}_{m}.png")
            plt.figure(figsize=(5.2,5.2))
            plt.plot(fpr, tpr, lw=2, label=f"{m} (AUC={auc:.3f})")
            plt.plot([0,1],[0,1],'k--', lw=1)
            plt.xlabel("FPR"); plt.ylabel("TPR")
            plt.title(f"ROC vs HBA1c ≥ {cut}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(png, dpi=300)
            plt.close()

            overlay.append((fpr, tpr, m, auc))

        # overlay for this cutoff
        if overlay:
            plt.figure(figsize=(6,6))
            for fpr, tpr, m, auc in overlay:
                plt.plot(fpr, tpr, lw=2, label=f"{m} (AUC={auc:.3f})")
            plt.plot([0,1],[0,1],'k--', lw=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC — HBA1c ≥ {cut}")
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"roc_HBA1c_all_cut{str(cut).replace('.','_')}.png"), dpi=300)
            plt.close()

    auc_csv = os.path.join(OUT_DIR, "assoc_HBA1c_AUC_AB.csv")
    pd.DataFrame(auc_rows).to_csv(auc_csv, index=False)
    print(f"[OK] Saved cutoff-based AUCs → {auc_csv}")
    print(f"[OK] Figures → {OUT_DIR}")

if __name__ == "__main__":
    main()
