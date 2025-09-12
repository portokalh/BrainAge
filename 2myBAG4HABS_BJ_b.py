#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 14:02:03 2025

@author: bas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HABS brain-age: preprocess + train + predict (7-fold CV, GATv2)
Writes:
  - cv_metrics.csv
  - cv_predictions.csv with columns:
      MRI_Exam_fixed, Age, PredictedAge, PredictedAge_corrected, BAG, cBAG, Fold,
      SubjectRoot, Timepoint, Delta_BAG, Delta_cBAG
  - cv_deltas_by_root.csv (one row per SubjectRoot with both y0 & y2)
"""

import os, re, random
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

def pick_col(df, candidates):
    """Return the first matching column (case-insensitive)."""
    if df is None or df.empty: return None
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns: return c
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    return None

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
    """Transpose ROI table to subjects×ROIs; index as MRI_Exam_fixed (H####_y# or H####y#)."""
    df = pd.read_csv(fpath, sep="\t")
    if drop_first_row:
        df = df.iloc[1:, :]
    if roi_col in df.columns:
        roi_vals = pd.to_numeric(df[roi_col], errors="coerce")
        df = df.loc[~(roi_vals == 0)].reset_index(drop=True)

    subj_cols = [c for c in df.columns if re.match(subject_regex, str(c))]
    if not subj_cols:
        raise ValueError(f"No subject columns matched regex {subject_regex!r} in {fpath}")

    df_t = df[subj_cols].transpose()
    df_t.columns = [f"ROI_{i+1}" for i in range(df_t.shape[1])]
    df_t = df_t.apply(pd.to_numeric, errors="coerce")

    # Accept H1234, H1234_y0, H1234y0
    pat = re.compile(r'^(?:H)?(\d{3,6})(?:_?y(\d+))?$', re.IGNORECASE)
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
    """Z-score across subjects per ROI/channel."""
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
    if os.path.isdir(CONN_DIR):
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

    # Ensure ID col named MRI_Exam_fixed
    mri_col = pick_col(df_meta, ["MRI_Exam_fixed", "DWI", "Subject_ID", "runno"])
    if not mri_col:
        raise KeyError("Need an exam ID column like 'MRI_Exam_fixed' or 'DWI' in metadata.")
    df_meta["MRI_Exam_fixed"] = df_meta[mri_col].astype(str).str.strip()

    # ---- Match metadata to available connectomes
    df_matched_connectomes = df_meta[df_meta["MRI_Exam_fixed"].isin(connectomes.keys())].copy()
    print(f"Matched subjects (metadata & connectome): {len(df_matched_connectomes)} / {len(connectomes)}")

    # ---- Risk columns (canonicalize to CDX_Cog; binary diabetes)
    cog_col = pick_col(df_matched_connectomes, ["CDX_Cog", "CDX_Cognitive", "cogdx", "cog_dx", "cogx", "CDX"])
    dia_col = pick_col(df_matched_connectomes, ["CDX_Diabetes", "IMH_Diabetes", "cdx_diabetesx"])

    df_matched_connectomes["CDX_Cog"] = to_numeric(df_matched_connectomes[cog_col]) if cog_col else np.nan

    if dia_col:
        def diab_bin(x):
            if pd.isna(x): return np.nan
            s = str(x).strip().lower()
            if s in {"1","yes","y","true","t","pos","positive"}: return 1
            if s in {"0","no","n","false","f","neg","negative"}: return 0
            try: return int(float(s) > 0)
            except Exception: return np.nan
        df_matched_connectomes["CDX_Diabetes_bin"] = df_matched_connectomes[dia_col].apply(diab_bin)
    else:
        df_matched_connectomes["CDX_Diabetes_bin"] = np.nan

    dementia_bool = df_matched_connectomes["CDX_Cog"].isin([2, 3])
    diabetes_bool = (df_matched_connectomes["CDX_Diabetes_bin"] == 1)

    df_matched_connectomes["Dementia_Risk"] = dementia_bool.fillna(False).astype(int)
    df_matched_connectomes["Diabetes_Risk"] = diabetes_bool.fillna(False).astype(int)
    df_matched_connectomes["Risk"] = ((df_matched_connectomes["Dementia_Risk"] == 1) |
                                      (df_matched_connectomes["Diabetes_Risk"] == 1)).astype(int)

    cogx_map = {0: "CN", 1: "Other/SMC", 2: "MCI", 3: "Dementia/AD"}
    df_matched_connectomes["CogDx_Label"] = df_matched_connectomes["CDX_Cog"].map(cogx_map).fillna("Unknown")

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
    common_ids = set(df_fa_clean.index) & set(df_md_clean.index) & set(df_vol_clean.index)
    for sid in common_ids:
        fa  = torch.tensor(df_fa_clean.loc[sid].values,  dtype=torch.float32)
        md  = torch.tensor(df_md_clean.loc[sid].values,  dtype=torch.float32)
        vol = torch.tensor(df_vol_clean.loc[sid].values, dtype=torch.float32)
        if fa.numel() == md.numel() == vol.numel():
            stacked = torch.stack([fa, md, vol], dim=1)  # [84, 3]
            multimodal_features_dict[sid] = stacked
    print("Subjects with FA/MD/Vol features:", len(multimodal_features_dict))

    # ---- Normalize node features across subjects (z-score)
    normalized_node_features_dict = normalize_multimodal_nodewise(multimodal_features_dict)

    # ---- Demographics: sex_num, APOE4_carrier, weight=1.0
    sex_col  = pick_col(df_metadata_PCA_healthy_withConnectome, ["Sex","sex","Gender"])
    apoe_col = pick_col(df_metadata_PCA_healthy_withConnectome, ["APOE4_Genotype","APOE4","apoe4_genotype"])
    if sex_col:
        df_metadata_PCA_healthy_withConnectome["sex_num"] = df_metadata_PCA_healthy_withConnectome[sex_col].apply(sex_to_num)
    else:
        df_metadata_PCA_healthy_withConnectome["sex_num"] = np.nan
    if apoe_col:
        df_metadata_PCA_healthy_withConnectome["APOE4_carrier"] = (
            df_metadata_PCA_healthy_withConnectome[apoe_col].astype(str).str.contains("4").astype(float)
        )
    else:
        df_metadata_PCA_healthy_withConnectome["APOE4_carrier"] = np.nan
    df_metadata_PCA_healthy_withConnectome["weight"] = 1.0

    # Save enriched healthy metadata
    df_metadata_PCA_healthy_withConnectome.to_csv(
        os.path.join(OUTDIR_CSV, "HABS_healthy_metadata_with_placeholders.csv"), index=False
    )

    # ---- Build demographic tensor dict [1,3] per subject (matches your earlier runs)
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
               os.path.join(OUTDIR_RES, "healthy_metadata_dict.pt"))
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

# Ensure age column (case-insensitive)
age_col = pick_col(df_healthy_meta, ["Age","age"])
if not age_col:
    raise KeyError("Healthy metadata missing 'Age' column")
df_healthy_meta["Age"] = to_numeric(df_healthy_meta[age_col])

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
    mask = (mat > thr)
    idx = mask.nonzero(as_tuple=False)  # [E,2]
    return idx.t().contiguous()         # [2,E]

graphs = []
age_series = df_healthy_meta.set_index("MRI_Exam_fixed")["Age"]
for sid in final_ids:
    conn_np = np.asarray(matched_connectomes_healthy[sid].values, dtype=np.float32)
    edge_index = mat_to_edge_index(torch.from_numpy(conn_np), thr=0.0)
    x = normalized_node_features_dict[sid]      # [84,3]
    u = subject_to_demographic_tensor[sid]      # [1,3]
    y = torch.tensor([age_series.loc[sid]], dtype=torch.float32)
    g = Data(x=x, edge_index=edge_index, y=y, u=u)
    g.sid = sid  # keep id on the Data object for later retrieval
    graphs.append(g)

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
        u = data.u  # concatenated across graphs; with [1,3] per graph becomes [B,3]
        if u.dim() == 3 and u.size(1) == 1:
            u = u.squeeze(1)  # handle [B,1,3] edge case
        x = torch.cat([x, u], dim=1)
        return self.lin_out(x).squeeze(1)  # [B]

# ----------------- Train with 7-fold CV + bias correction -----------------
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

kf = KFold(n_splits=7, shuffle=True, random_state=42)

rows = []           # per-sample outputs (all folds)
fold_metrics = []   # raw metrics on uncorrected predictions (summary)

for fold, (tr_idx, te_idx) in enumerate(kf.split(graphs), start=1):
    train_ds = [graphs[i] for i in tr_idx]
    test_ds  = [graphs[i] for i in te_idx]
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    # for bias fit (train predictions)
    train_eval_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    model = BrainAgeGAT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # ---- Train
    for epoch in range(100):  # tune/patience as needed
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)                      # [B]
            loss = loss_fn(pred, batch.y.view(-1))   # FIX: no squeeze(1)
            loss.backward()
            opt.step()

    # ---- TRAIN predictions for bias model
    model.eval()
    ypred_tr, age_tr = [], []
    with torch.no_grad():
        for batch in train_eval_loader:
            batch = batch.to(device)
            pred = model(batch).cpu().numpy().ravel()
            ages = batch.y.cpu().numpy().ravel()
            ypred_tr.extend(pred.tolist())
            age_tr.extend(ages.tolist())
    ypred_tr = np.asarray(ypred_tr)
    age_tr   = np.asarray(age_tr)

    # Fit linear bias model: PredictedAge ~ Age  (train only)
    bias_reg = LinearRegression().fit(age_tr.reshape(-1,1), ypred_tr)
    a, b = float(bias_reg.intercept_), float(bias_reg.coef_[0])

    # ---- TEST predictions (apply correction) with aligned IDs
    sid_seq = [g.sid for g in test_ds]  # order preserved when shuffle=False
    pos = 0
    ypred_te, age_te, sid_te = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).cpu().numpy().ravel()
            ages = batch.y.cpu().numpy().ravel()

            n = pred.shape[0]
            ypred_te.extend(pred.tolist())
            age_te.extend(ages.tolist())
            sid_te.extend(sid_seq[pos:pos+n])  # align IDs to this batch
            pos += n

    ypred_te = np.asarray(ypred_te)
    age_te   = np.asarray(age_te)

    # Correction per fold:
    expected_te = bias_reg.predict(age_te.reshape(-1,1))  # a + b*Age
    pred_corr   = ypred_te - expected_te + age_te         # corrected predicted age
    bag         = ypred_te - age_te                       # raw BAG
    cbag        = pred_corr - age_te                      # corrected BAG (residual)

    # ---- Raw metrics (optional logging)
    mae = float(mean_absolute_error(age_te, ypred_te))
    rmse = float(np.sqrt(np.mean((age_te - ypred_te)**2)))
    r2 = float(r2_score(age_te, ypred_te))
    fold_metrics.append({"fold": fold, "MAE": mae, "RMSE": rmse, "R2": r2, "a": a, "b": b})
    print(f"Fold {fold}: MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f} | bias a={a:.3f}, b={b:.3f}")

    # ---- Collect per-sample rows
    for sid, age_i, p_raw, p_cor, bag_i, cbag_i in zip(sid_te, age_te, ypred_te, pred_corr, bag, cbag):
        rows.append({
            "MRI_Exam_fixed": sid,
            "Age": float(age_i),
            "PredictedAge": float(p_raw),
            "PredictedAge_corrected": float(p_cor),
            "BAG": float(bag_i),
            "cBAG": float(cbag_i),
            "Fold": fold
        })

# ----------------- Save base predictions -----------------
metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv(os.path.join(OUTDIR_RES, "cv_metrics.csv"), index=False)

pred_df = pd.DataFrame(rows)

# ----------------- Add SubjectRoot / Timepoint -----------------
# Timepoint: infer y0 vs y2 from the exam string (suffix like _y0/_y2 or y0/y2)
pred_df["MRI_Exam_fixed"] = pred_df["MRI_Exam_fixed"].astype(str)
pred_df["Timepoint"] = np.where(
    pred_df["MRI_Exam_fixed"].str.contains(r"[_-]?y2$", case=False, regex=True), "y2", "y0"
)
# SubjectRoot: strip trailing visit tokens
pred_df["SubjectRoot"] = pred_df["MRI_Exam_fixed"].str.replace(r"([_\-]?y[0-9]+)$", "", regex=True)

# ----------------- Compute deltas per root for subjects with both y0 & y2 -----------------
y0 = pred_df[pred_df["Timepoint"]=="y0"].set_index("SubjectRoot")
y2 = pred_df[pred_df["Timepoint"]=="y2"].set_index("SubjectRoot")
aligned = y0.join(y2, how="inner", lsuffix="_y0", rsuffix="_y2")

aligned["Delta_BAG"]   = aligned["BAG_y2"]  - aligned["BAG_y0"]
aligned["Delta_cBAG"]  = aligned["cBAG_y2"] - aligned["cBAG_y0"]

# summary: how many roots have both timepoints
n_both = aligned.shape[0]
print(f"Roots with both y0 & y2: {n_both}")

# compact root-level CSV
aligned[["Delta_BAG","Delta_cBAG"]].reset_index().to_csv(
    os.path.join(OUTDIR_RES, "cv_deltas_by_root.csv"), index=False
)

# merge delta columns back to per-row table (same delta copied to y0/y2 rows of the same SubjectRoot)
pred_df = pred_df.merge(
    aligned[["Delta_BAG","Delta_cBAG"]].reset_index(),
    on="SubjectRoot", how="left"
)

# ----------------- Final save -----------------
pred_df.to_csv(os.path.join(OUTDIR_RES, "cv_predictions.csv"), index=False)

print("\n=== Cross-Validation Summary (uncorrected) ===")
print(metrics_df)
print("\nWrote:")
print(" -", os.path.join(OUTDIR_RES, "cv_metrics.csv"))
print(" -", os.path.join(OUTDIR_RES, "cv_predictions.csv"))
print(" -", os.path.join(OUTDIR_RES, "cv_deltas_by_root.csv"))
