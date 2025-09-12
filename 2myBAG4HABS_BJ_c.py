#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HABS brain-age: preprocess + train + predict (7-fold CV, GATv2)
Fixes:
- Store demographics u as [1,3] per graph so batching yields [B,3]
- Robust bias correction fit on TRAIN (mask NaNs; fallback if <3 pairs)
- Proper test-ID alignment across DataLoader batches
- Drops rows with missing Age before graph build to avoid NaNs in bias fit

Outputs (to OUTDIR_RES):
  - cv_metrics.csv (per-fold: MAE/RMSE/R2, corrected MAE/RMSE/R2)
  - cv_predictions.csv (MRI_Exam_fixed, Age, PredictedAge, PredictedAge_corrected, BAG, cBAG)
Also saves intermediate CSVs in OUTDIR_CSV.
"""

import os, re, random
import numpy as np
import pandas as pd
import torch

# -----------------------------
#            PATHS
# -----------------------------
CONN_DIR = "/home/bas/Desktop/MyData/harmonization/HABS/connectomes/DWI/plain/"

# Use the CSV you showed in your last run
HABS_META_PATH  = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata.csv"
HABS_META_SHEET = 0  # ignored for CSV

FA_PATH  = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_Regional_Stats/HABS_studywide_stats_for_fa.txt"
MD_PATH  = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_Regional_Stats/HABS_studywide_stats_for_md.txt"
VOL_PATH = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_Regional_Stats/HABS_studywide_stats_for_volume.txt"

OUTDIR_CSV = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/"
OUTDIR_RES = "/home/bas/Desktop/MyData/harmonization/HABS/results/"

os.makedirs(OUTDIR_CSV, exist_ok=True)
os.makedirs(OUTDIR_RES, exist_ok=True)

# Toggle if you already ran preprocessing and have .pt files in OUTDIR_RES
SKIP_PREPROCESS = False

# -----------------------------
#         UTILITIES
# -----------------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def read_metadata_any(path: str, sheet=None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
        print(f"[meta] XLSX loaded: shape={df.shape}")
        return df
    # CSV/TSV autodetect
    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True)
        print(f"[meta] CSV autodetect loaded: shape={df.shape}")
        return df
    except Exception:
        for sep in [",", "\t", r"\s+"]:
            try:
                df = pd.read_csv(path, sep=sep, engine="python", comment="#", skip_blank_lines=True)
                print(f"[meta] CSV loaded with sep={sep!r}: shape={df.shape}")
                return df
            except Exception:
                continue
        raise ValueError(f"Could not parse metadata file: {path}")

def load_and_clean_metric_to_mri_exam(
    fpath: str,
    roi_col: str = "ROI",
    subject_regex: str = r"^H\d+",
    drop_first_row: bool = True,
) -> pd.DataFrame:
    """Read studywide stats (tsv), transpose to subjects x ROIs, index as MRI_Exam_fixed (H####_y#)."""
    df = pd.read_csv(fpath, sep="\t")
    if drop_first_row: df = df.iloc[1:, :]
    if roi_col in df.columns:
        df = df[df[roi_col] != "0"].reset_index(drop=True)

    subj_cols = [c for c in df.columns if re.match(subject_regex, str(c))]
    if not subj_cols:
        raise ValueError(f"No subject columns matched regex {subject_regex!r} in {fpath}")

    df_t = df[subj_cols].transpose()
    df_t.columns = [f"ROI_{i+1}" for i in range(df_t.shape[1])]
    df_t.index.name = "subject_raw"
    df_t = df_t.apply(pd.to_numeric, errors="coerce")

    # Build keys EXACTLY like MRI_Exam_fixed: "H####_y#"
    pat = re.compile(r'^(?:H)?(\d{3,6})(?:_y(\d+))?$', re.IGNORECASE)
    cleaned = {}
    for subj in map(str, df_t.index):
        m = pat.match(subj)
        if not m: 
            continue
        num_no_pad = str(int(m.group(1)))  # "04369" -> "4369"
        visit = m.group(2)
        key = f"H{num_no_pad}_y{visit}" if visit is not None else f"H{num_no_pad}"
        if key not in cleaned:
            cleaned[key] = df_t.loc[subj]

    out = pd.DataFrame.from_dict(cleaned, orient="index")
    out.index.name = "MRI_Exam_fixed"
    return out

def normalize_multimodal_nodewise(feature_dict):
    if not feature_dict:
        return {}
    all_feats = torch.stack(list(feature_dict.values()))  # [N,84,3]
    mean = all_feats.mean(dim=0)
    std  = all_feats.std(dim=0) + 1e-8
    return {sid: (feat - mean) / std for sid, feat in feature_dict.items()}

def sex_to_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"m","male","0"}: return 0.0
    if s in {"f","female","1"}: return 1.0
    return np.nan

def first_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================================================================
#                              PREPROCESS
# =============================================================================
if not SKIP_PREPROCESS:
    # ---- Load connectomes
    print("\nCONNECTOMES")
    connectomes = {}
    for fname in os.listdir(CONN_DIR):
        if fname.endswith("_harmonized.csv"):
            sid = fname.replace("_harmonized.csv", "")  # "H4369_y0"
            fpath = os.path.join(CONN_DIR, fname)
            df_conn = pd.read_csv(fpath, header=None)
            connectomes[sid] = df_conn
    connectomes = {k: v for k, v in connectomes.items() if "_whitematter" not in k}
    print("Connectomes:", len(connectomes))

    # ---- Load metadata
    print("\nHABS METADATA")
    df_meta = read_metadata_any(HABS_META_PATH, sheet=HABS_META_SHEET)
    df_meta = df_meta.drop_duplicates().reset_index(drop=True)
    print(f"[meta] columns: {len(df_meta.columns)}")

    # Ensure ID col named MRI_Exam_fixed exists
    if "MRI_Exam_fixed" not in df_meta.columns:
        dwi_col = first_col(df_meta, ["DWI","dwi"])
        if dwi_col:
            df_meta["MRI_Exam_fixed"] = df_meta[dwi_col].astype(str)
        else:
            raise KeyError("Need 'MRI_Exam_fixed' or 'DWI' in metadata.")

    # ---- Risk columns (flexible names)
    cog_col = first_col(df_meta, ["CDX_Cog","cdx_cogx","CDX_COG","cdx_cog"])
    dia_col = first_col(df_meta, ["CDX_Diabetes","cdx_diabetesx","CDX_DIABETES","cdx_diabetes"])

    df_meta["__Cog"] = to_numeric(df_meta[cog_col]) if cog_col else np.nan
    df_meta["__Dia"] = to_numeric(df_meta[dia_col]) if dia_col else np.nan

    dementia_bool = df_meta["__Cog"].isin([2, 3]).fillna(False)           # MCI/AD => risk
    diabetes_bool = (df_meta["__Dia"].fillna(0).astype(int) == 1)         # diabetes => risk

    df_meta["Dementia_Risk"] = dementia_bool.astype(int)
    df_meta["Diabetes_Risk"] = diabetes_bool.astype(int)
    df_meta["Risk"] = ((df_meta["Dementia_Risk"] == 1) |
                       (df_meta["Diabetes_Risk"] == 1)).astype(int)

    cogx_map = {0: "CN", 1: "Other/SMC", 2: "MCI", 3: "Dementia/AD"}
    df_meta["CogDx_Label"] = df_meta["__Cog"].map(cogx_map).fillna("Unknown")

    # ---- Match metadata to available connectomes
    df_matched_connectomes = df_meta[df_meta["MRI_Exam_fixed"].isin(connectomes.keys())].copy()
    print(f"Matched subjects (metadata & connectome): {len(df_matched_connectomes)} / {len(connectomes)}")

    # ---- Healthy filter (Risk == 0)
    before_n = len(df_matched_connectomes)
    df_matched_habs_healthy = df_matched_connectomes[df_matched_connectomes["Risk"] == 0].copy()
    after_n = len(df_matched_habs_healthy)
    print(f"Healthy subjects (Risk==0): {after_n} / {before_n}")

    # ---- Save CSVs
    df_matched_connectomes.to_csv(os.path.join(OUTDIR_CSV, "HABS_with_risk.csv"), index=False)
    df_matched_habs_healthy.to_csv(os.path.join(OUTDIR_CSV, "HABS_healthy_controls.csv"), index=False)
    print("Saved CSVs to:", OUTDIR_CSV)

    # ---- Healthy connectome dict
    matched_connectomes_healthy = {
        row["MRI_Exam_fixed"]: connectomes[row["MRI_Exam_fixed"]]
        for _, row in df_matched_habs_healthy.iterrows()
        if row["MRI_Exam_fixed"] in connectomes
    }
    print("Healthy connectomes:", len(matched_connectomes_healthy))

    # ---- Placeholder biomarkers
    df_metadata_PCA_healthy_withConnectome = df_matched_habs_healthy.copy()
    for col in ["Abeta", "Tau", "GFAP"]:
        df_metadata_PCA_healthy_withConnectome[col] = np.nan

    # ---- Load FA/MD/VOL with MRI_Exam_fixed indices
    df_fa_clean  = load_and_clean_metric_to_mri_exam(FA_PATH)
    df_md_clean  = load_and_clean_metric_to_mri_exam(MD_PATH)
    df_vol_clean = load_and_clean_metric_to_mri_exam(VOL_PATH)
    print("FA/MD/VOL shapes:", df_fa_clean.shape, df_md_clean.shape, df_vol_clean.shape)

    # ---- Build multimodal node features: [84,3] per subject
    multimodal_features_dict = {}
    for sid in df_fa_clean.index:
        if sid in df_md_clean.index and sid in df_vol_clean.index:
            fa  = torch.tensor(df_fa_clean.loc[sid].values,  dtype=torch.float32)
            md  = torch.tensor(df_md_clean.loc[sid].values,  dtype=torch.float32)
            vol = torch.tensor(df_vol_clean.loc[sid].values, dtype=torch.float32)
            stacked = torch.stack([fa, md, vol], dim=1)  # [84, 3]
            multimodal_features_dict[sid] = stacked
    print("Subjects with FA/MD/Vol features:", len(multimodal_features_dict))

    # ---- Normalize node features across subjects (z-score)
    normalized_node_features_dict = normalize_multimodal_nodewise(multimodal_features_dict)

    # ---- Demographics: sex_num, APOE4_carrier, weight=1.0
    sex_col   = first_col(df_metadata_PCA_healthy_withConnectome, ["sex","Sex","SEX"])
    apoe_col  = first_col(df_metadata_PCA_healthy_withConnectome, ["apoe4_genotype","APOE4_Genotype","APOE4"])
    age_col   = first_col(df_metadata_PCA_healthy_withConnectome, ["Age","age"])
    if age_col is None:
        raise KeyError("No 'Age' column found in metadata.")
    df_metadata_PCA_healthy_withConnectome["Age"] = to_numeric(df_metadata_PCA_healthy_withConnectome[age_col])

    df_metadata_PCA_healthy_withConnectome["sex_num"] = (
        df_metadata_PCA_healthy_withConnectome.get(sex_col, np.nan).apply(sex_to_num) if sex_col else np.nan
    )
    df_metadata_PCA_healthy_withConnectome["APOE4_carrier"] = (
        df_metadata_PCA_healthy_withConnectome.get(apoe_col, "").astype(str).str.contains("4").astype(int)
        if apoe_col else 0
    )
    df_metadata_PCA_healthy_withConnectome["weight"] = 1.0

    # Save enriched healthy metadata
    df_metadata_PCA_healthy_withConnectome.to_csv(
        os.path.join(OUTDIR_CSV, "HABS_healthy_metadata_with_placeholders.csv"), index=False
    )

    # ---- Build demographic tensor dict [1,3] per subject (for batching)
    key_col = "MRI_Exam_fixed"
    meta_for_tensors = df_metadata_PCA_healthy_withConnectome.dropna(subset=[key_col])
    subject_to_demographic_tensor = {}
    for _, row in meta_for_tensors.iterrows():
        # Keep as [1,3] so batches => [B,3]
        u_vec = [row.get("sex_num", np.nan), row.get("APOE4_carrier", np.nan), row.get("weight", 1.0)]
        subject_to_demographic_tensor[row[key_col]] = torch.tensor([u_vec], dtype=torch.float32)  # [1,3]

    print("Demographic tensors:", len(subject_to_demographic_tensor))

    # ---- Save tensors
    torch.save(multimodal_features_dict,          os.path.join(OUTDIR_RES, "multimodal_features_dict.pt"))
    torch.save(normalized_node_features_dict,     os.path.join(OUTDIR_RES, "normalized_node_features_dict.pt"))
    torch.save(subject_to_demographic_tensor,     os.path.join(OUTDIR_RES, "subject_to_demographic_tensor.pt"))
    torch.save(matched_connectomes_healthy,       os.path.join(OUTDIR_RES, "connectomes_healthy_dict.pt"))
    torch.save(df_metadata_PCA_healthy_withConnectome.to_dict(orient="list"),
               os.path.join(OUTDIR_RES, "healthy_metadata_dict.pt"))  # compact save
    print("Saved tensors to:", OUTDIR_RES)

# =============================================================================
#                           TRAIN / PREDICT
# =============================================================================
# Load tensors (works whether we just saved them or SKIP_PREPROCESS=True)
multimodal_features_dict      = torch.load(os.path.join(OUTDIR_RES, "multimodal_features_dict.pt"))
normalized_node_features_dict = torch.load(os.path.join(OUTDIR_RES, "normalized_node_features_dict.pt"))
subject_to_demographic_tensor = torch.load(os.path.join(OUTDIR_RES, "subject_to_demographic_tensor.pt"))
matched_connectomes_healthy   = torch.load(os.path.join(OUTDIR_RES, "connectomes_healthy_dict.pt"))
healthy_metadata_dict         = torch.load(os.path.join(OUTDIR_RES, "healthy_metadata_dict.pt"))
df_healthy_meta = pd.DataFrame(healthy_metadata_dict)

# Ensure needed columns
if "MRI_Exam_fixed" not in df_healthy_meta.columns:
    raise KeyError("Healthy metadata missing MRI_Exam_fixed")
age_col = first_col(df_healthy_meta, ["Age","age"])
if age_col != "Age":
    df_healthy_meta["Age"] = to_numeric(df_healthy_meta[age_col])
else:
    df_healthy_meta["Age"] = to_numeric(df_healthy_meta["Age"])

# Drop missing Age (prevents NaNs in bias fit)
missing_age_before = df_healthy_meta["Age"].isna().sum()
if missing_age_before > 0:
    print(f"[cleanup] Missing Age before: {missing_age_before} | Dropped rows with NaN Age")
df_healthy_meta = df_healthy_meta.dropna(subset=["Age"]).reset_index(drop=True)

# Final overlap IDs (must exist everywhere)
ids_feats = set(normalized_node_features_dict.keys())
ids_demo  = set(subject_to_demographic_tensor.keys())
ids_conn  = set(matched_connectomes_healthy.keys())
ids_meta  = set(df_healthy_meta["MRI_Exam_fixed"])

final_ids = sorted(list(ids_feats & ids_demo & ids_conn & ids_meta))
print(f"\nFinal overlap subjects: {len(final_ids)}")

# ----------------- Build PyG graphs -----------------
from torch_geometric.data import Data
def mat_to_edge_index(mat: torch.Tensor, thr: float = 0.0):
    # mat: 2D tensor [84,84]
    mask = (mat > thr)
    idx = mask.nonzero(as_tuple=False)  # [E,2]
    return idx.t().contiguous()         # [2,E]

graphs, id_order = [], []
age_series = df_healthy_meta.set_index("MRI_Exam_fixed")["Age"]
for sid in final_ids:
    conn_np = np.asarray(matched_connectomes_healthy[sid].values, dtype=np.float32)
    edge_index = mat_to_edge_index(torch.from_numpy(conn_np), thr=0.0)

    x = normalized_node_features_dict[sid]  # [84,3]
    u = subject_to_demographic_tensor[sid]  # [1,3]  <-- keep as [1,3], NOT [3]
    y = torch.tensor(age_series.loc[sid], dtype=torch.float32)  # scalar

    data = Data(x=x, edge_index=edge_index, y=y, u=u)
    graphs.append(data)
    id_order.append(sid)

print("Graph objects:", len(graphs))

# ----------------- Define GATv2 model -----------------
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class BrainAgeGAT(nn.Module):
    def __init__(self, in_channels=3, hidden=32, heads=8, demo_dim=3):
        super().__init__()
        self.g1 = GATv2Conv(in_channels, hidden, heads=heads, concat=True)
        self.g2 = GATv2Conv(hidden*heads, hidden, heads=heads, concat=True)
        self.lin1 = nn.Linear(hidden*heads, 64)
        self.lin_out = nn.Linear(64 + demo_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.g1(x, edge_index))
        x = torch.relu(self.g2(x, edge_index))
        x = global_mean_pool(x, batch)        # [B, hidden*heads]
        x = torch.relu(self.lin1(x))
        u = data.u                             # [B, 3] because each graph carried [1,3]
        if u.dim() == 1:
            u = u.view(1, -1)                 # safety
        x = torch.cat([x, u], dim=1)
        return self.lin_out(x).squeeze(1)     # [B]

# ----------------- Train with 7-fold CV -----------------
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

kf = KFold(n_splits=7, shuffle=True, random_state=42)
all_rows, fold_metrics = [], []

for fold, (tr_idx, te_idx) in enumerate(kf.split(graphs), start=1):
    train_loader = DataLoader([graphs[i] for i in tr_idx], batch_size=16, shuffle=True)
    test_loader  = DataLoader([graphs[i] for i in te_idx], batch_size=32, shuffle=False)

    model = BrainAgeGAT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # ---- Train
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)              # [B]
            loss = loss_fn(pred, batch.y)    # batch.y is [B] because we stored scalars
            loss.backward()
            opt.step()

    # ---- Get TRAIN predictions for bias correction
    model.eval()
    ypred_tr, age_tr = [], []
    with torch.no_grad():
        for batch in DataLoader([graphs[i] for i in tr_idx], batch_size=64, shuffle=False):
            batch = batch.to(device)
            pred = model(batch).cpu().numpy().ravel()
            y    = batch.y.cpu().numpy().ravel()
            ypred_tr.extend(pred.tolist())
            age_tr.extend(y.tolist())
    ypred_tr = np.array(ypred_tr, dtype=float)
    age_tr   = np.array(age_tr,   dtype=float)
    mask_tr  = np.isfinite(ypred_tr) & np.isfinite(age_tr)

    if mask_tr.sum() >= 3:
        bias_reg = LinearRegression().fit(age_tr[mask_tr].reshape(-1,1), ypred_tr[mask_tr])
        a, b = float(bias_reg.intercept_), float(bias_reg.coef_[0])
    else:
        a, b = 0.0, 1.0  # fallback: no correction

    # ---- Eval on TEST
    fold_pred, fold_true, fold_sid = [], [], []
    test_ids = [id_order[i] for i in te_idx]
    offset = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred  = model(batch).cpu().numpy().ravel()
            true  = batch.y.cpu().numpy().ravel()
            bsz   = len(pred)
            sids  = test_ids[offset:offset+bsz]
            offset += bsz

            fold_pred.extend(pred.tolist())
            fold_true.extend(true.tolist())
            fold_sid.extend(sids)

            # Bias-correct per item (on test)
            pred_corr = pred - (a + b*true) + true  # ŷ - (a + b*age) + age
            for sid_i, age_i, p_i, pc_i in zip(sids, true, pred, pred_corr):
                all_rows.append({
                    "MRI_Exam_fixed": sid_i,
                    "Age": float(age_i),
                    "PredictedAge": float(p_i),
                    "PredictedAge_corrected": float(pc_i),
                    "BAG": float(p_i - age_i),
                    "cBAG": float(pc_i - age_i)
                })

    # Fold metrics (uncorrected and corrected)
    fold_pred = np.array(fold_pred, dtype=float)
    fold_true = np.array(fold_true, dtype=float)
    mae  = float(mean_absolute_error(fold_true, fold_pred))
    rmse = float(np.sqrt(np.mean((fold_true - fold_pred)**2)))
    r2   = float(r2_score(fold_true, fold_pred))

    pred_corr_full = fold_pred - (a + b*fold_true) + fold_true
    mae_c  = float(mean_absolute_error(fold_true, pred_corr_full))
    rmse_c = float(np.sqrt(np.mean((fold_true - pred_corr_full)**2)))
    r2_c   = float(r2_score(fold_true, pred_corr_full))

    fold_metrics.append({"fold": fold,
                         "MAE": mae, "RMSE": rmse, "R2": r2,
                         "MAE_corr": mae_c, "RMSE_corr": rmse_c, "R2_corr": r2_c})
    print(f"Fold {fold}: MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f} | "
          f"MAE_corr={mae_c:.2f} | RMSE_corr={rmse_c:.2f} | R2_corr={r2_c:.3f}")

# ----------------- Save results -----------------
metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv(os.path.join(OUTDIR_RES, "cv_metrics.csv"), index=False)

pred_df = pd.DataFrame(all_rows)
# in case of duplicates from any unforeseen reason, keep first
pred_df = pred_df.drop_duplicates(subset=["MRI_Exam_fixed"]).reset_index(drop=True)
pred_df.to_csv(os.path.join(OUTDIR_RES, "cv_predictions.csv"), index=False)

print("\n=== Cross-Validation Summary ===")
print(metrics_df)
print("\nSaved to:", OUTDIR_RES)



###save final model trained on all healthy ###

# ===== Train final model on ALL healthy graphs and save a checkpoint =====
from torch_geometric.loader import DataLoader
import json

final_model = BrainAgeGAT().to(device)
final_opt   = torch.optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
final_loss  = nn.MSELoss()

loader_all = DataLoader(graphs, batch_size=16, shuffle=True)

EPOCHS_ALL = 100
for epoch in range(EPOCHS_ALL):
    final_model.train()
    for batch in loader_all:
        batch = batch.to(device)
        final_opt.zero_grad()
        pred = final_model(batch)          # [B]
        loss = final_loss(pred, batch.y)   # batch.y is [B]
        loss.backward()
        final_opt.step()

# Save the trained-on-all model checkpoint
ckpt_path = os.path.join(OUTDIR_RES, "model_trained_on_all_healthy.pt")
torch.save(final_model.state_dict(), ckpt_path)
print(f"[OK] Saved final model: {ckpt_path}")

# (Optional) Fit & save bias-correction params on ALL healthy after training
final_model.eval()
with torch.no_grad():
    yhat_all, age_all = [], []
    for batch in DataLoader(graphs, batch_size=64, shuffle=False):
        batch = batch.to(device)
        yhat_all.extend(final_model(batch).cpu().numpy().ravel().tolist())
        age_all.extend(batch.y.cpu().numpy().ravel().tolist())

yhat_all = np.array(yhat_all, dtype=float)
age_all  = np.array(age_all,  dtype=float)
mask_all = np.isfinite(yhat_all) & np.isfinite(age_all)

if mask_all.sum() >= 3:
    bias_reg_all = LinearRegression().fit(age_all[mask_all].reshape(-1,1), yhat_all[mask_all])
    a_all, b_all = float(bias_reg_all.intercept_), float(bias_reg_all.coef_[0])
else:
    a_all, b_all = 0.0, 1.0

bias_json = os.path.join(OUTDIR_RES, "bias_correction_all_healthy.json")
with open(bias_json, "w") as f:
    json.dump({"a": a_all, "b": b_all}, f, indent=2)
print(f"[OK] Saved bias params on all healthy to: {bias_json}")
