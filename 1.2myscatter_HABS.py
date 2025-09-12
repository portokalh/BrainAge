# ============================== IMPORTS ==============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ============================ CONFIGURATION ============================
# Use your file path; if you want the uploaded file from this chat, set:
# CSV_PATH = "/mnt/data/cv_predictions.csv"
CSV_PATH = "/home/bas/Desktop/MyData/harmonization/HABS/results/cv_predictions.csv"
OUTPUT_DIR = "/home/bas/Desktop/MyData/harmonization/HABS/results/HABS_original_scatter_1_y0only"
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
    "Real_Age", "real_age", "y_true", "true_age", "Age", "age",
    "ChronologicalAge", "Chron_Age", "Target", "target", "Label", "label"
]
pred_cands = [
    "Predicted_Age", "predicted_age", "y_pred", "pred_age", "prediction",
    "predicted", "Pred", "Prediction", "BrainAge", "brain_age_pred", "PredictedAge"
]
# Include MRI_Exam_fixed so it becomes Subject_ID
subj_cands = [
    "Subject_ID","subject_id","MRI_Exam_fixed","Subject","subject","RID","ID","runno","Runno","runNo"
]
fold_cands = ["Fold","fold","CV_Fold","cv_fold"]
rep_cands  = ["Repeat","repeat","Rep","rep","seed","Seed"]

cols = list(df.columns)
true_col = _first_match(true_cands, cols)
pred_col = _first_match(pred_cands, cols)
subj_col = _first_match(subj_cands, cols)
fold_col = _first_match(fold_cands, cols)
rep_col  = _first_match(rep_cands, cols)

# Create standardized columns in-memory only (no file writes)
if true_col and "Real_Age" not in df.columns:
    df["Real_Age"] = df[true_col]
if pred_col and "Predicted_Age" not in df.columns:
    df["Predicted_Age"] = df[pred_col]
if subj_col and "Subject_ID" not in df.columns:
    df["Subject_ID"] = df[subj_col]   # handles MRI_Exam_fixed → Subject_ID
if fold_col and "Fold" not in df.columns:
    df["Fold"] = df[fold_col]
if rep_col and "Repeat" not in df.columns:
    df["Repeat"] = df[rep_col]

# ============================== FILTER: only IDs with 'y0' ==============================
if "Subject_ID" not in df.columns:
    raise ValueError(
        "No Subject_ID-like column found (e.g., 'MRI_Exam_fixed', 'Subject', 'runno'). "
        "Needed to filter by 'y0'."
    )

mask_y0 = df["Subject_ID"].astype(str).str.contains("y0", case=False, na=False)
df = df.loc[mask_y0].copy()
if df.empty:
    raise ValueError("After filtering for Subject_ID containing 'y0', no rows remain.")

print(f"Rows after 'y0' filter: {len(df)}")

# ========================= SANITY CHECK FOR REQUIRED COLS =========================
for c in ["Real_Age", "Predicted_Age"]:
    if c not in df.columns:
        raise ValueError(f"Required column '{c}' not found after renaming. "
                         f"Available columns: {list(df.columns)}")

# =================================================================================
# 1) SCATTER PLOT — ALL POINTS (one prediction per repetition)
# =================================================================================
x_all = df["Real_Age"].values
y_all = df["Predicted_Age"].values

mae_all  = mean_absolute_error(x_all, y_all)
rmse_all = mean_squared_error(x_all, y_all, squared=False)
r2_all   = r2_score(x_all, y_all)

if {"Fold","Repeat"}.issubset(df.columns):
    grouped = df.groupby(["Fold","Repeat"]).apply(
        lambda g: pd.Series({
            "MAE": mean_absolute_error(g["Real_Age"], g["Predicted_Age"]),
            "RMSE": mean_squared_error(g["Real_Age"], g["Predicted_Age"], squared=False),
            "R2": r2_score(g["Real_Age"], g["Predicted_Age"])
        })
    )
    mae_std, rmse_std, r2_std = grouped["MAE"].std(ddof=1), grouped["RMSE"].std(ddof=1), grouped["R2"].std(ddof=1)
else:
    mae_std = rmse_std = r2_std = np.nan

reg = LinearRegression().fit(x_all.reshape(-1, 1), y_all)
slope, intercept = reg.coef_[0], reg.intercept_

plt.figure(figsize=(8, 6))
plt.scatter(x_all, y_all, alpha=0.6, label="Predictions", edgecolors="k")
min_val, max_val = min(x_all.min(), y_all.min()), max(x_all.max(), y_all.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y = x)")
plt.plot([min_val, max_val], reg.predict([[min_val], [max_val]]), 'b-', alpha=0.6, linewidth=2,
         label=f"Trend: y = {slope:.2f}x + {intercept:.1f}")

def _fmt_std(v): return f" ± {v:.2f}" if not np.isnan(v) else ""
plt.gca().text(0.95, 0.05,
    f"MAE = {mae_all:.2f}{_fmt_std(mae_std)}\n"
    f"RMSE = {rmse_all:.2f}{_fmt_std(rmse_std)}\n"
    f"R² = {r2_all:.2f}{_fmt_std(r2_std)}",
    transform=plt.gca().transAxes, fontsize=12,
    va='bottom', ha='right',
    bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black', alpha=0.8)
)
plt.xlabel("Real Age (years)")
plt.ylabel("Predicted Age (years)")
plt.title("Scatter — All Repetitions (Subject_ID contains 'y0')")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_all_predictions_y0.png"), dpi=300)
plt.close()

# =================================================================================
# 2) SCATTER PLOT — MEAN PREDICTION PER SUBJECT
# =================================================================================
df_mean = df.groupby("Subject_ID").agg({
    "Real_Age": "first",
    "Predicted_Age": "mean"
}).reset_index()

x_mean = df_mean["Real_Age"].values
y_mean = df_mean["Predicted_Age"].values

mae_mean  = mean_absolute_error(x_mean, y_mean)
rmse_mean = mean_squared_error(x_mean, y_mean, squared=False)
r2_mean   = r2_score(x_mean, y_mean)

if {"Fold","Repeat"}.issubset(df.columns):
    metrics_by_split = []
    for (fold, rep), grp in df.groupby(["Fold","Repeat"]):
        g_mean = grp.groupby("Subject_ID").agg({"Real_Age":"first","Predicted_Age":"mean"})
        metrics_by_split.append([
            mean_absolute_error(g_mean["Real_Age"], g_mean["Predicted_Age"]),
            mean_squared_error(g_mean["Real_Age"], g_mean["Predicted_Age"], squared=False),
            r2_score(g_mean["Real_Age"], g_mean["Predicted_Age"])
        ])
    metrics_by_split = np.array(metrics_by_split)
    mae_mean_std, rmse_mean_std, r2_mean_std = (
        metrics_by_split[:,0].std(ddof=1),
        metrics_by_split[:,1].std(ddof=1),
        metrics_by_split[:,2].std(ddof=1),
    )
else:
    mae_mean_std = rmse_mean_std = r2_mean_std = np.nan

reg = LinearRegression().fit(x_mean.reshape(-1, 1), y_mean)
slope, intercept = reg.coef_[0], reg.intercept_

plt.figure(figsize=(8, 6))
plt.scatter(x_mean, y_mean, alpha=0.7, label="Mean per Subject", edgecolors="k")
min_val, max_val = min(x_mean.min(), y_mean.min()), max(x_mean.max(), y_mean.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y = x)")
plt.plot([min_val, max_val], reg.predict([[min_val], [max_val]]), 'b-', linewidth=2, alpha=0.6,
         label=f"Trend: y = {slope:.2f}x + {intercept:.1f}")

def _fmt(v): return f" ± {v:.2f}" if not np.isnan(v) else ""
plt.gca().text(
    0.95, 0.05,
    f"MAE  = {mae_mean:.2f}{_fmt(mae_mean_std)}\n"
    f"RMSE = {rmse_mean:.2f}{_fmt(rmse_mean_std)}\n"
    f"R²   = {r2_mean:.2f}{_fmt(r2_mean_std)}",
    transform=plt.gca().transAxes, fontsize=12,
    ha='right', va='bottom',
    bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black', alpha=0.8)
)
plt.xlabel("Real Age (years)")
plt.ylabel("Mean Predicted Age (years)")
plt.title("Scatter — Mean Prediction per Subject (ID contains 'y0')")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_mean_prediction_per_subject_y0.png"), dpi=300)
plt.close()

# =================================================================================
# 3) SCATTER PLOT — MEAN ± SD PER SUBJECT (error bars)
# =================================================================================
df_stats = df.groupby("Subject_ID").agg({
    "Real_Age": "first",
    "Predicted_Age": ["mean", "std"]
}).reset_index()
df_stats.columns = ["Subject_ID", "Real_Age", "Pred_Mean", "Pred_SD"]

x_err = df_stats["Real_Age"].values
y_err = df_stats["Pred_Mean"].values
y_std = df_stats["Pred_SD"].values

reg = LinearRegression().fit(x_err.reshape(-1, 1), y_err)
slope, intercept = reg.coef_[0], reg.intercept_

plt.figure(figsize=(8, 6))
plt.errorbar(
    x_err, y_err, yerr=y_std,
    fmt='o', ecolor='gray', elinewidth=1, capsize=3,
    markeredgecolor='k', markerfacecolor='tab:blue', alpha=0.8,
    label="Mean ± SD per Subject"
)
min_val, max_val = min(x_err.min(), y_err.min()), max(x_err.max(), y_err.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y = x)")
plt.plot([min_val, max_val], reg.predict([[min_val], [max_val]]), 'b-', linewidth=2, alpha=0.6,
         label=f"Trend: y = {slope:.2f}x + {intercept:.1f}")

plt.gca().text(
    0.95, 0.05,
    f"MAE  = {mean_absolute_error(x_err, y_err):.2f}\n"
    f"RMSE = {mean_squared_error(x_err, y_err, squared=False):.2f}\n"
    f"R²   = {r2_score(x_err, y_err):.2f}",
    transform=plt.gca().transAxes, fontsize=12,
    ha='right', va='bottom',
    bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black', alpha=0.8)
)
plt.xlabel("Real Age (years)")
plt.ylabel("Mean Predicted Age (years)")
plt.title("Scatter — Mean ± SD per Subject (ID contains 'y0')")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_mean_prediction_per_subject_errorbars_y0.png"), dpi=300)
plt.close()

print("✅ Finished plots (filtered to Subject_ID containing 'y0'). Saved in:", OUTPUT_DIR)
