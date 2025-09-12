#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROC AUC vs CDX_Cog (0..3) for brain-age metrics.

Input:
  ENRICHED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_with_PredictedAgeAB_BAG.csv"

Outputs to results/associations_AB:
  - auc_CDX_Cog_AB.csv
  - roc_CDX_Cog_all_<contrast>.png (overlay per contrast)
  - roc_CDX_Cog_<contrast>_<metric>.png (per metric)
"""

import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# ---------- CONFIG ----------
ENRICHED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_with_PredictedAgeAB_BAG.csv"
OUT_DIR = os.path.join("/home/bas/Desktop/MyData/harmonization/HABS/results/", "associations_AB")
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATE_METRICS = [
    "BAG_AB", "cBAG_AB",
    "PredictedAge_AB", "PredictedAge_corrected_AB",
    "Delta_BAG_AB", "Delta_cBAG_AB",
]

BOOTSTRAP_N = 1000

# ---------- HELPERS ----------
def coerce_num(s): return pd.to_numeric(s, errors="coerce")

def find_col(df, candidates):
    norm = {re.sub(r'[^a-z0-9]+','', c.lower()): c for c in df.columns}
    for c in candidates:
        k = re.sub(r'[^a-z0-9]+','', c.lower())
        if k in norm: return norm[k]
    return None

def clean_cdx_cog(x):
    """Return int in {0,1,2,3} or NaN; drop 9/other codes."""
    v = coerce_num(pd.Series([x])).iloc[0]
    if np.isnan(v): return np.nan
    v = int(v)
    return v if v in {0,1,2,3} else np.nan

def prepare_metric_vectors(df, metric, y_bin, subj_root_col=None):
    """Return y (0/1) and score arrays; de-dup delta metrics per SubjectRoot."""
    if metric.startswith("Delta_") and subj_root_col and subj_root_col in df.columns:
        sub = df[[subj_root_col, metric]].copy()
        sub[metric] = coerce_num(sub[metric])
        sub = sub.dropna(subset=[metric]).groupby(subj_root_col, as_index=False).first()
        sub["y"] = y_bin.loc[sub[subj_root_col].values].values if isinstance(y_bin, pd.Series) else np.nan
        sub = sub.dropna(subset=["y"])
        y = sub["y"].astype(int).values
        s = sub[metric].astype(float).values
        return (None, None) if len(np.unique(y)) < 2 else (y, s)
    else:
        sub = df[[metric]].copy()
        sub[metric] = coerce_num(sub[metric])
        sub["y"] = y_bin.values if isinstance(y_bin, pd.Series) else y_bin
        sub = sub.dropna(subset=[metric, "y"])
        y = sub["y"].astype(int).values
        s = sub[metric].astype(float).values
        return (None, None) if len(np.unique(y)) < 2 else (y, s)

def auc_ci_bootstrap(y, s, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y); aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:  # need both classes
            continue
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

def plot_roc(fpr, tpr, label, title, out_png):
    plt.figure(figsize=(5.2, 5.2))
    plt.plot(fpr, tpr, lw=2, label=label)
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlim([0,1]); plt.ylim([0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ---------- MAIN ----------
def main():
    df = pd.read_csv(ENRICHED_CSV)

    # Find columns
    cdx_col = find_col(df, ["CDX_Cog","cdx_cog","CDX_COG"])
    if cdx_col is None:
        raise ValueError("CDX_Cog column not found.")
    subj_root_col = find_col(df, ["SubjectRoot","Subject_Root"])
    if subj_root_col is None:
        sid_col = find_col(df, ["Subject_ID","Subject","MRI_Exam_fixed","DWI"])
        if sid_col:
            sr = df[sid_col].astype(str).str.replace(r"\.0$","",regex=True)
            df["SubjectRoot"] = sr.str.replace(r"([_\-]?y[0-9]+)$","",regex=True)
            subj_root_col = "SubjectRoot"

    # Clean CDX values
    df["_CDX"] = df[cdx_col].apply(clean_cdx_cog)

    # Define binarizations (label, lambda mapping from {0..3} to 0/1)
    CONTRASTS = [
        ("CDX_ge1_vs_0",   lambda v: 1 if v>=1 else 0, "Any concern (≥1) vs CN (0)"),
        ("CDX_ge2_vs_lt2", lambda v: 1 if v>=2 else 0, "Impaired (≥2) vs CN/SMC (0–1)"),
        ("CDX_eq3_vs_rest",lambda v: 1 if v==3 else 0, "AD (3) vs others (0–2)"),
    ]

    metrics = [m for m in CANDIDATE_METRICS if m in df.columns]
    if not metrics:
        raise ValueError(f"No expected metrics present. Looked for: {CANDIDATE_METRICS}")

    all_rows = []
    for ckey, binfun, cdesc in CONTRASTS:
        # Build binary series for available rows
        mask = ~df["_CDX"].isna()
        y_bin = df.loc[mask, "_CDX"].map(binfun)
        y_bin = y_bin.astype(float)  # keep as series aligned to df[mask]

        overlay_cur = []
        for m in metrics:
            y, s = prepare_metric_vectors(df.loc[mask].assign(y_bin=y_bin), m,
                                          y_bin=y_bin, subj_root_col=subj_root_col)
            if y is None:
                all_rows.append({"contrast": ckey, "contrast_desc": cdesc, "metric": m,
                                 "n": 0, "n0": 0, "n1": 0,
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
            out_png = os.path.join(OUT_DIR, f"roc_CDX_Cog_{ckey}_{m}.png")
            plot_roc(fpr, tpr, label, f"ROC: {cdesc}", out_png)
            overlay_cur.append((fpr, tpr, label))

            all_rows.append({
                "contrast": ckey, "contrast_desc": cdesc, "metric": m,
                "n": n, "n0": n0, "n1": n1,
                "AUC": float(auc_val), "CI95_lo": lo, "CI95_hi": hi,
                "thr_Youden": thr, "sens": sens, "spec": spec
            })

        # Overlay for this contrast
        if overlay_cur:
            plt.figure(figsize=(6,6))
            for fpr, tpr, lab in overlay_cur:
                plt.plot(fpr, tpr, lw=2, label=lab)
            plt.plot([0,1],[0,1],'k--', lw=1)
            plt.xlim([0,1]); plt.ylim([0,1])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC — {cdesc}")
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"roc_CDX_Cog_all_{ckey}.png"), dpi=300)
            plt.close()

    # Save summary
    out_csv = os.path.join(OUT_DIR, "auc_CDX_Cog_AB.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"[OK] CDX_Cog AUC summary → {out_csv}")
    print(f"[OK] ROC PNGs → {OUT_DIR}")

if __name__ == "__main__":
    main()
