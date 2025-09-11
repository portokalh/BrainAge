#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 18:00:34 2025

@author: bas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HABS brain-age: preprocess + train + predict (7-fold CV, GATv2)
Outputs:
- CSVs (risk + healthy) -> /home/bas/Desktop/MyData/harmonization/HABS/metadata/
- Tensors, models, predictions -> /home/bas/Desktop/MyData/harmonization/HABS/results/
"""

import os, re, random, json
import numpy as np
import pandas as pd
import torch

# -----------------------------
#            PATHS
# -----------------------------
CONN_DIR = "/home/bas/Desktop/MyData/harmonization/HABS/connectomes/DWI/plain/"

# Metadata can be CSV or XLSX
HABS_META_PATH = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata.xlsx"
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

    # Ensure ID col named MRI_Exam_fixed exists
    if "MRI_Exam_fixed" not in df_meta.columns:
        if "DWI" in df_meta.columns:
            df_meta["MRI_Exam_fixed"] = df_meta["DWI"].astype(str)
        else:
            raise KeyError("Need 'MRI_Exam_fixed' or 'DWI' in metadata.")

    # ---- Match metadata to available connectomes
    df_matched_connectomes = df_meta[df_meta["MRI_Exam_fixed"].isin(connectomes.keys())].copy()
    print(f"Matched subjects (metadata & connectome): {len(df_matched_connectomes)} / {len(connectomes)}")

    # ---- Risk columns (HABS rules)
    df_matched_connectomes["cdx_cogx"] = to_numeric(df_matched_connectomes.get("cdx_cogx", np.nan))
    df_matched_connectomes["cdx_diabetesx"] = to_numeric(df_matched_connectomes.get("cdx_diabetesx", np.nan))

    dementia_bool = df_matched_connectomes["cdx_cogx"].isin([2, 3]).fillna(False)           # MCI/AD => risk
    diabetes_bool = (df_matched_connectomes["cdx_diabetesx"].fillna(0).astype(int) == 1)   # diabetes => risk

    df_matched_connectomes["Dementia_Risk"] = dementia_bool.astype(int)
    df_matched_connectomes["Diabetes_Risk"] = diabetes_bool.astype(int)
    df_matched_connectomes["Risk"] = ((df_matched_connectomes["Dementia_Risk"] == 1) |
                                      (df_matched_connectomes["Diabetes_Risk"] == 1)).astype(int)

    cogx_map = {0: "CN", 1: "Other/SMC", 2: "MCI", 3: "Dementia/AD"}
    df_matched_connectomes["CogDx_Label"] = df_matched_connectomes["cdx_cogx"].map(cogx_map).fillna("Unknown")

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
    df_metadata_PCA_healthy_withConnectome["sex_num"] = df_metadata_PCA_healthy_withConnectome.get("sex", np.nan).apply(sex_to_num)
    df_metadata_PCA_healthy_withConnectome["APOE4_carrier"] = (
        df_metadata_PCA_healthy_withConnectome.get("apoe4_genotype", "").astype(str).str.contains("4").astype(int)
    )
    df_metadata_PCA_healthy_withConnectome["weight"] = 1.0

    # Save enriched healthy metadata
    df_metadata_PCA_healthy_withConnectome.to_csv(
        os.path.join(OUTDIR_CSV, "HABS_healthy_metadata_with_placeholders.csv"), index=False
    )

    # ---- Build demographic tensor dict [1,3] per subject (for batching)
    key_col = "MRI_Exam_fixed"
    meta_for_tensors = df_metadata_PCA_healthy_withConnectome.dropna(subset=[key_col, "sex_num"])
    subject_to_demographic_tensor = {
        row[key_col]: torch.tensor([[row["sex_num"], row["APOE4_carrier"], row["weight"]]], dtype=torch.float32)
        for _, row in meta_for_tensors.iterrows()
    }
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
if "age" not in df_healthy_meta.columns:
    raise KeyError("Healthy metadata missing 'age' column")

df_healthy_meta["age"] = to_numeric(df_healthy_meta["age"])

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

graphs = []
age_series = df_healthy_meta.set_index("MRI_Exam_fixed")["age"]
for sid in final_ids:
    # adjacency from connectome (numpy -> torch)
    conn_np = np.asarray(matched_connectomes_healthy[sid].values, dtype=np.float32)
    edge_index = mat_to_edge_index(torch.from_numpy(conn_np), thr=0.0)
    # node features
    x = normalized_node_features_dict[sid]  # [84,3]
    # demographics [1,3] per graph
    u = subject_to_demographic_tensor[sid]  # [1,3]
    # label
    y = torch.tensor([age_series.loc[sid]], dtype=torch.float32)
    graphs.append(Data(x=x, edge_index=edge_index, y=y, u=u))

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
        x = global_mean_pool(x, batch)  # [B, hidden*heads]
        x = torch.relu(self.lin1(x))
        # demographics (cat along feature dim)
        u = data.u  # [B, 3] (because each graph had [1,3] and DataLoader concatenates on dim 0)
        x = torch.cat([x, u], dim=1)
        return self.lin_out(x).squeeze(1)

# ----------------- Train with 7-fold CV -----------------
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

kf = KFold(n_splits=7, shuffle=True, random_state=42)
all_preds, all_truths, all_sids, fold_metrics = [], [], [], []

for fold, (tr_idx, te_idx) in enumerate(kf.split(graphs), start=1):
    train_loader = DataLoader([graphs[i] for i in tr_idx], batch_size=16, shuffle=True)
    test_loader  = DataLoader([graphs[i] for i in te_idx], batch_size=32, shuffle=False)

    model = BrainAgeGAT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # ---- Train
    for epoch in range(100):  # you can tune/patience
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            opt.step()

    # ---- Eval
    model.eval()
    fold_pred, fold_true, fold_sid = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).cpu().numpy()
            true = batch.y.cpu().numpy()
            fold_pred.extend(pred.tolist())
            fold_true.extend(true.tolist())
            # Recover subject ids per graph in this batch
            # We saved them in the same order 'final_ids'
        # te_idx maps to indices in 'graphs' which correspond to 'final_ids'
        fold_sid = [final_ids[i] for i in te_idx]

    mae = float(mean_absolute_error(fold_true, fold_pred))
    rmse = float(np.sqrt(np.mean((np.array(fold_true)-np.array(fold_pred))**2)))
    r2 = float(r2_score(fold_true, fold_pred))
    fold_metrics.append({"fold": fold, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"Fold {fold}: MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f}")

    all_preds.extend(fold_pred)
    all_truths.extend(fold_true)
    all_sids.extend(fold_sid)

# ----------------- Save results -----------------
metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv(os.path.join(OUTDIR_RES, "cv_metrics.csv"), index=False)

pred_df = pd.DataFrame({"MRI_Exam_fixed": all_sids,
                        "Age": np.array(all_truths).ravel(),
                        "PredictedAge": np.array(all_preds).ravel()})
pred_df.to_csv(os.path.join(OUTDIR_RES, "cv_predictions.csv"), index=False)

# Simple summary
print("\n=== Cross-Validation Summary ===")
print(metrics_df)
print("\nSaved to:", OUTDIR_RES)
