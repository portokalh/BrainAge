#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge metadata with predictions, compute Brain-Age metrics (using Age directly),
and save outputs.

Adds columns:
  - PredictedAge_AB
  - BAG_AB                         (PredictedAge_AB - Age)
  - cBAG_AB                        (residuals of Predicted ~ Age)
  - PredictedAge_corrected_AB      (bias-corrected brain age = Age + cBAG_AB)
  - Delta_BAG_AB                   (y2 − y0 per SubjectRoot)
  - Delta_cBAG_AB                  (y2 − y0 per SubjectRoot)

Outputs (same directory as META_CSV):
  <basename>_with_PredictedAgeAB_BAG.csv
  <basename>_with_PredictedAgeAB_BAG_short.csv   (columns with ≥50% non-missing kept)
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ===================== CONFIG =====================
META_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata_AB_BJ.csv"
PRED_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/results/cv_predictions.csv"

MAKE_SHORT   = True       # save a short CSV keeping cols with ≥ SHORT_THRESH non-missing
SHORT_THRESH = 0.50

# ===================== HELPERS =====================
def _first_match(cands, cols):
    for c in cands:
        if c in cols:
            return c
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _as_key_series(s: pd.Series) -> pd.Series:
    """Normalize ID/path-like keys for merging."""
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)   # drop trailing .0 from float-y IDs
    s = s.str.replace(r"\s+", " ", regex=True)   # collapse whitespace
    return s

def _basename_noext(s: pd.Series) -> pd.Series:
    """Get basename and remove common medical image extensions (case-insensitive)."""
    t = _as_key_series(s)
    t = t.str.replace(r".*[\\/]", "", regex=True)  # basename
    t = t.str.replace(r"(?i)\.(nii(\.gz)?|mgh|mgz|nrrd|img|hdr|dcm|dicom|zip|tar\.gz)$", "", regex=True)
    return t

def _choose_pred_col(pred_df: pd.DataFrame) -> str:
    """Pick a predicted-age column from predictions."""
    cands = [
        "Predicted_Age","PredictedAge",
        "06_BrainAgeR_PredictedAge","06_DeepBrainNet_PredictedAge",
        "BrainAge","brain_age_pred","y_pred","prediction","Prediction","Pred"
    ]
    col = _first_match(cands, list(pred_df.columns))
    if not col:
        raise ValueError(f"No predicted-age column found. Tried: {cands}")
    return col

def _infer_timepoint(row) -> str:
    """Return 'y2' if row suggests it, else 'y0'."""
    for c in ("Subject_ID","DWI"):
        if c in row and pd.notna(row[c]):
            txt = str(row[c]).lower()
            if "y2" in txt: return "y2"
            if "y0" in txt: return "y0"
    return "y0"

# ===================== MAIN =====================
def main():
    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"Metadata file not found: {META_CSV}")
    if not os.path.exists(PRED_CSV):
        raise FileNotFoundError(f"Predictions file not found: {PRED_CSV}")

    meta = pd.read_csv(META_CSV)
    pred = pd.read_csv(PRED_CSV)

    # ---- bring PredictedAge_AB by matching DWI <-> MRI_Exam_fixed ----
    if "DWI" not in meta.columns:
        raise ValueError("Metadata must contain a 'DWI' column for matching.")

    mri_exam_col = _first_match(
        ["MRI_Exam_fixed","MRI_Exam","MRIExam","ExamID","exam_id","Subject_ID","runno"],
        list(pred.columns)
    )
    if not mri_exam_col:
        raise ValueError("Predictions must contain an exam identifier like 'MRI_Exam_fixed'.")

    pred_age_col = _choose_pred_col(pred)

    meta["__key_exact"] = _as_key_series(meta["DWI"])
    pred["__key_exact"] = _as_key_series(pred[mri_exam_col])

    # Average predictions per exam (handles CV/repeats)
    pred_exact = pred.groupby("__key_exact", as_index=False)[pred_age_col].mean()

    merged = meta.merge(
        pred_exact.rename(columns={pred_age_col: "__PredictedAge_tmp"}),
        on="__key_exact", how="left"
    )
    exact_hits = merged["__PredictedAge_tmp"].notna().sum()

    # Fallback: basename match if few exact hits
    strategy = "exact"
    if exact_hits < max(5, 0.2 * len(meta)):  # heuristic
        merged["__key_base"] = _basename_noext(merged["DWI"])
        pred["__key_base"]  = _basename_noext(pred[mri_exam_col])
        pred_base = pred.groupby("__key_base", as_index=False)[pred_age_col].mean()

        merged_base = merged.merge(
            pred_base.rename(columns={pred_age_col: "__PredictedAge_tmp_base"}),
            on="__key_base", how="left"
        )
        merged["__PredictedAge_tmp"] = merged["__PredictedAge_tmp"].where(
            merged["__PredictedAge_tmp"].notna(),
            merged_base["__PredictedAge_tmp_base"]
        )
        base_only = merged["__PredictedAge_tmp"].notna().sum() - exact_hits
        strategy = f"exact+basename (exact={exact_hits}, base-only adds={base_only})"

    merged["PredictedAge_AB"] = merged["__PredictedAge_tmp"]
    merged.drop(columns=[c for c in merged.columns if c.startswith("__key_") or c.startswith("__PredictedAge_tmp")],
                inplace=True, errors="ignore")

    # ---- Ensure Age exists and numeric ----
    if "Age" not in merged.columns:
        raise ValueError("No 'Age' column found in metadata.")
    age_num = pd.to_numeric(merged["Age"], errors="coerce")

    # SubjectRoot + Timepoint (for deltas)
    if "Subject_ID" in merged.columns:
        sid = merged["Subject_ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        merged["SubjectRoot"] = sid.str.replace(r"([_\-]?y[02])$", "", regex=True)
    else:
        merged["SubjectRoot"] = _basename_noext(merged["DWI"]).str.replace(r"([_\-]?y[02])$", "", regex=True)
    merged["Timepoint_AB"] = merged.apply(_infer_timepoint, axis=1)

    # ---- BAG_AB (using Age directly) ----
    merged["BAG_AB"] = merged["PredictedAge_AB"] - age_num

    # ---- Bias-corrected cBAG_AB + bias-corrected brain age (using Age) ----
    fit_df = pd.DataFrame({
        "Pred": pd.to_numeric(merged["PredictedAge_AB"], errors="coerce"),
        "Age":  age_num
    }).dropna()

    if len(fit_df) >= 3 and fit_df["Age"].nunique() >= 2:
        reg = LinearRegression().fit(
            fit_df[["Age"]].to_numpy(),
            fit_df["Pred"].to_numpy()
        )
        mask = age_num.notna() & merged["PredictedAge_AB"].notna()
        expected = np.full(len(merged), np.nan, dtype=float)
        expected[mask.to_numpy()] = reg.predict(
            pd.DataFrame({"Age": age_num[mask]}).to_numpy()
        )

        # Age-independent gap (residuals)
        merged["cBAG_AB"] = merged["PredictedAge_AB"] - expected
        # >>> Bias-corrected brain age <<<
        merged["PredictedAge_corrected_AB"] = age_num + merged["cBAG_AB"]

        print(f"[Bias model] Predicted = {reg.coef_[0]:.4f} * Age + {reg.intercept_:.4f} (n={len(fit_df)})")
    else:
        merged["cBAG_AB"] = np.nan
        merged["PredictedAge_corrected_AB"] = np.nan
        print("[Bias model] Not enough usable rows (need ≥3 and ≥2 unique ages).")

    # ---- Deltas per SubjectRoot: y2 - y0 (for gaps) ----
    tmp = merged[["SubjectRoot","Timepoint_AB","BAG_AB","cBAG_AB"]].copy()
    y0 = tmp[tmp["Timepoint_AB"]=="y0"].groupby("SubjectRoot", as_index=True)[["BAG_AB","cBAG_AB"]].mean()
    y2 = tmp[tmp["Timepoint_AB"]=="y2"].groupby("SubjectRoot", as_index=True)[["BAG_AB","cBAG_AB"]].mean()
    aligned = y0.join(y2, how="inner", lsuffix="_y0", rsuffix="_y2")

    aligned["Delta_BAG_AB"]  = aligned["BAG_AB_y2"]  - aligned["BAG_AB_y0"]
    aligned["Delta_cBAG_AB"] = aligned["cBAG_AB_y2"] - aligned["cBAG_AB_y0"]
    deltas = aligned[["Delta_BAG_AB","Delta_cBAG_AB"]].reset_index()

    # Merge deltas back to all rows of that SubjectRoot
    merged = merged.merge(deltas, on="SubjectRoot", how="left")

    # ---- SAVE full ----
    base_dir = os.path.dirname(META_CSV)
    base_name, ext = os.path.splitext(os.path.basename(META_CSV))
    out_full = os.path.join(base_dir, f"{base_name}_with_PredictedAgeAB_BAG{ext}")
    merged.to_csv(out_full, index=False)

    print(f"[OK] Wrote full CSV: {out_full}")
    print(f"Matched PredictedAge_AB rows: {merged['PredictedAge_AB'].notna().sum()}/{len(merged)}")
    print(f"Paired subjects (y0 & y2): {deltas.shape[0]}")

    # ---- SAVE short (≥50% filled) ----
    if MAKE_SHORT:
        eval_df = merged.replace(r'^\s*$', np.nan, regex=True)
        fill_ratio = eval_df.notna().mean(axis=0)

        must_keep = {
            "DWI","SubjectRoot","Timepoint_AB","Age",
            "PredictedAge_AB","BAG_AB","cBAG_AB","PredictedAge_corrected_AB",
            "Delta_BAG_AB","Delta_cBAG_AB"
        }
        must_keep |= {c for c in ["Subject_ID"] if c in merged.columns}

        keep_cols = sorted(set(fill_ratio[fill_ratio >= SHORT_THRESH].index).union(must_keep))
        keep_cols = [c for c in keep_cols if c in merged.columns]
        short_df = merged[keep_cols].copy()

        out_short = os.path.join(base_dir, f"{base_name}_with_PredictedAgeAB_BAG_short{ext}")
        short_df.to_csv(out_short, index=False)
        print(f"[OK] Wrote short CSV (≥{int(SHORT_THRESH*100)}% filled): {out_short}")
        print(f"Full shape:  {merged.shape}  |  Short shape: {short_df.shape}")

if __name__ == "__main__":
    main()
