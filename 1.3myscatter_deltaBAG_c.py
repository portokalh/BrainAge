# ============================== IMPORTS ==============================
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ============================ CONFIGURATION ============================
# If you want the uploaded file from this chat, set:
# CSV_PATH = "/mnt/data/cv_predictions.csv"
CSV_PATH = "/home/bas/Desktop/MyData/harmonization/HABS/results/cv_predictions.csv"
OUTPUT_DIR = "HABS_BAG_and_corrected_with_deltas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================== LOAD CSV ==============================
df = pd.read_csv(CSV_PATH)

# =================== STANDARDIZE/RENAME COLUMNS (IN MEMORY) ===================
def _first_match(cands, cols):
    for c in cands:
        if c in cols:
            return c
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

true_cands = [
    "Real_Age","real_age","y_true","true_age","Age","age",
    "ChronologicalAge","Chron_Age","Target","target","Label","label"
]
pred_cands = [
    # If you have a corrected-age column and want to prefer it, put it first, e.g.:
    # "06_BrainAgeR_PredictedAge",
    "Predicted_Age","predicted_age","y_pred","pred_age","prediction",
    "predicted","Pred","Prediction","BrainAge","brain_age_pred","PredictedAge"
]
# Include MRI_Exam_fixed so it becomes Subject_ID
subj_cands = ["Subject_ID","subject_id","MRI_Exam_fixed","Subject","subject","RID","ID","runno","Runno","runNo"]

cols = list(df.columns)
true_col = _first_match(true_cands, cols)
pred_col = _first_match(pred_cands, cols)
subj_col = _first_match(subj_cands, cols)

if true_col and "Real_Age" not in df.columns:
    df["Real_Age"] = df[true_col]
if pred_col and "Predicted_Age" not in df.columns:
    df["Predicted_Age"] = df[pred_col]
if subj_col and "Subject_ID" not in df.columns:
    df["Subject_ID"] = df[subj_col]   # MRI_Exam_fixed → Subject_ID if present

# ============================== FILTER TO y0 / y2 ==============================
mask_y_any = df["Subject_ID"].astype(str).str.contains(r"(y0|y2)", case=False, na=False)
df = df.loc[mask_y_any].copy()
if df.empty:
    raise ValueError("After filtering for Subject_ID containing 'y0' or 'y2', no rows remain.")

# Identify timepoint & SubjectRoot
df["Timepoint"] = np.where(df["Subject_ID"].astype(str).str.contains("y2", case=False, na=False), "y2", "y0")
df["SubjectRoot"] = df["Subject_ID"].astype(str).str.replace(r"([_\-]?y[02])$", "", regex=True)

# Quick counts
roots_y0 = set(df.loc[df["Timepoint"]=="y0","SubjectRoot"].unique())
roots_y2 = set(df.loc[df["Timepoint"]=="y2","SubjectRoot"].unique())
roots_both = roots_y0 & roots_y2
print("==========================================")
print(f"✅ Unique subjects with y0: {len(roots_y0)}")
print(f"✅ Unique subjects with y2: {len(roots_y2)}")
print(f"✅ Unique subjects present at BOTH y0 and y2: {len(roots_both)}")
print("==========================================")

# ============================== AGE ADJUSTMENT (+2y for y2) ==============================
is_y2_row = df["Timepoint"].eq("y2")
df.loc[is_y2_row, "Real_Age"] = df.loc[is_y2_row, "Real_Age"] + 2.0

# ============================== COLLAPSE REPEATS PER Subject_ID ==============================
# One row per Subject_ID (so y0 and y2 each appear once if present)
df_u = df.groupby("Subject_ID").agg({
    "SubjectRoot": "first",
    "Timepoint": "first",
    "Real_Age": "first",          # adjusted real age (y2 +2y already applied)
    "Predicted_Age": "mean"       # mean across repeats/CV if any
}).reset_index()

# ============================== BAG & BIAS CORRECTION ==============================
# Raw BAG per timepoint
df_u["BAG_raw"] = df_u["Predicted_Age"] - df_u["Real_Age"]

# Bias correction: Predicted ~ Real (on ALL y0+y2 entries after adjustment)
x = df_u["Real_Age"].values.reshape(-1,1)
y = df_u["Predicted_Age"].values
reg = LinearRegression().fit(x, y)
expected = reg.predict(x)

# Save "expected predicted age" as Real_Age_corrected proxy
df_u["Real_Age_corrected"] = expected
# Corrected BAG = residuals (Predicted − Expected)
df_u["BAG_corrected"] = y - expected

print(f"Linear correction model: Predicted = {reg.coef_[0]:.4f} * Real + {reg.intercept_:.4f}")

# ============================== SAVE PER-SUBJECT (TIMEPOINT) CSV ==============================
per_subject_csv = os.path.join(OUTPUT_DIR, "per_subject_timepoint.csv")
df_u[[
    "Subject_ID","SubjectRoot","Timepoint",
    "Real_Age","Real_Age_corrected",
    "Predicted_Age","BAG_raw","BAG_corrected"
]].to_csv(per_subject_csv, index=False)
print(f"✅ Saved timepoint-level CSV to: {per_subject_csv}")

# ============================== BUILD DELTAS (y2 − y0) PER ROOT ==============================
# Keep only paired roots
df_paired = df_u[df_u["SubjectRoot"].isin(roots_both)].copy()

# Split y0 and y2 tables
y0 = df_paired[df_paired["Timepoint"]=="y0"].set_index("SubjectRoot")
y2 = df_paired[df_paired["Timepoint"]=="y2"].set_index("SubjectRoot")

# Align on roots (inner join ensures both present)
aligned = y0.join(
    y2,
    how="inner",
    lsuffix="_y0",
    rsuffix="_y2"
)

# Compute deltas: y2 − y0
aligned["Delta_BAG_raw"]             = aligned["BAG_raw_y2"]             - aligned["BAG_raw_y0"]
aligned["Delta_BAG_corrected"]       = aligned["BAG_corrected_y2"]       - aligned["BAG_corrected_y0"]
aligned["Delta_Real_Age_corrected"]  = aligned["Real_Age_corrected_y2"]  - aligned["Real_Age_corrected_y0"]

# Baseline (y0) real age for plotting delta panels
aligned["Real_Age_y0_for_plot"] = aligned["Real_Age_y0"]

# ============================== SAVE DELTAS CSV ==============================
delta_cols = [
    "Subject_ID_y0","Subject_ID_y2",
    "Real_Age_y0","Real_Age_y2",
    "Real_Age_corrected_y0","Real_Age_corrected_y2","Delta_Real_Age_corrected",
    "Predicted_Age_y0","Predicted_Age_y2",
    "BAG_raw_y0","BAG_raw_y2","Delta_BAG_raw",
    "BAG_corrected_y0","BAG_corrected_y2","Delta_BAG_corrected"
]
per_root_delta_csv = os.path.join(OUTPUT_DIR, "per_subjectroot_deltas.csv")
aligned.reset_index()[["SubjectRoot"] + delta_cols].to_csv(per_root_delta_csv, index=False)
print(f"✅ Saved paired deltas (including ΔReal_Age_corrected) to: {per_root_delta_csv}")

# ============================== PLOTTING HELPERS ==============================
def _scatter_with_trend(x, y, xlabel, ylabel, title, out_png):
    x = np.asarray(x)
    y = np.asarray(y)
    reg_ = LinearRegression().fit(x.reshape(-1,1), y)
    slope_, intercept_ = reg_.coef_[0], reg_.intercept_
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, alpha=0.85, edgecolors="k")
    xmin, xmax = float(np.min(x)), float(np.max(x))
    plt.axhline(0.0, linestyle="--")
    plt.plot([xmin, xmax], reg_.predict([[xmin],[xmax]]), linewidth=2,
             label=f"Trend: {slope_:.4f}·x + {intercept_:.2f}")
    mean_y = float(np.mean(y))
    std_y  = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    plt.gca().text(
        0.97, 0.03,
        f"Mean = {mean_y:.2f} ± {std_y:.2f}\n"
        f"n = {len(y)}",
        transform=plt.gca().transAxes,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightgray", edgecolor="black", alpha=0.85)
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✅ Saved plot: {out_png}")

# ============================== PLOTS ==============================
# 1) BAG vs Real Age (all entries)
_scatter_with_trend(
    df_u["Real_Age"].values,
    df_u["BAG_raw"].values,
    xlabel="Real Age (years)",
    ylabel="BAG (Predicted − Real) (years)",
    title="BAG vs Real Age (y0 & y2; y2 age +2y)",
    out_png=os.path.join(OUTPUT_DIR, "BAG_vs_real_age_all.png")
)

# 2) Corrected BAG vs Real Age (all entries)
_scatter_with_trend(
    df_u["Real_Age"].values,
    df_u["BAG_corrected"].values,
    xlabel="Real Age (years)",
    ylabel="Corrected BAG (years)",
    title="Corrected BAG vs Real Age (bias-corrected residuals)",
    out_png=os.path.join(OUTPUT_DIR, "Corrected_BAG_vs_real_age_all.png")
)

# 3) ΔBAG vs baseline Real Age (paired roots only)
_scatter_with_trend(
    aligned["Real_Age_y0_for_plot"].values,
    aligned["Delta_BAG_raw"].values,
    xlabel="Baseline Real Age (y0, years)",
    ylabel="ΔBAG (y2 − y0, years)",
    title="ΔBAG vs Baseline Real Age (paired y0/y2)",
    out_png=os.path.join(OUTPUT_DIR, "Delta_BAG_vs_baseline_real_age.png")
)

# 4) ΔCorrected BAG vs baseline Real Age (paired roots only)
_scatter_with_trend(
    aligned["Real_Age_y0_for_plot"].values,
    aligned["Delta_BAG_corrected"].values,
    xlabel="Baseline Real Age (y0, years)",
    ylabel="ΔCorrected BAG (y2 − y0, years)",
    title="ΔCorrected BAG vs Baseline Real Age (paired y0/y2; bias-corrected)",
    out_png=os.path.join(OUTPUT_DIR, "Delta_Corrected_BAG_vs_baseline_real_age.png")
)

print("🎯 Done: per-timepoint + deltas CSVs saved, and 4 plots written to:", OUTPUT_DIR)
