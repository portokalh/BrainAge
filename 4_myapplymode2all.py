#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply trained brain-age model to ALL subjects, compute BAG & bias-corrected BAG,
and append results to metadata.

Outputs:
  - results/predictions_all_subjects_AB.csv
  - metadata/<original name>_with_PredictedAgeAB_BAG.csv

Assumes you already trained and saved:
  results/brain_age_model/model_trained_on_all_healthy.pt

And you have these tensors saved from preprocessing:
  results/multimodal_features_dict.pt
  results/normalized_node_features_dict.pt  (contains ALL subjects; used for scaling)
"""

import os
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# =========================
#            PATHS
# =========================
CONN_DIR      = "/home/bas/Desktop/MyData/harmonization/HABS/connectomes/DWI/plain/"
META_CSV      = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata.csv"  # or your current file
RESULTS_DIR   = "/home/bas/Desktop/MyData/harmonization/HABS/results/"
MODEL_DIR     = os.path.join(RESULTS_DIR, "brain_age_model")
MODEL_PATH    = os.path.join(MODEL_DIR, "model_trained_on_all_healthy.pt")

# Saved tensors (from your earlier preprocessing step)
MULTI_PT      = os.path.join(RESULTS_DIR, "multimodal_features_dict.pt")
NORMAL_PT     = os.path.join(RESULTS_DIR, "normalized_node_features_dict.pt")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# =========================
#        HELPERS
# =========================
def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def sex_to_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"m","male","0"}: return 0.0
    if s in {"f","female","1"}: return 1.0
    return np.nan

def pick_series(df: pd.DataFrame, names, fallback_val=np.nan) -> pd.Series:
    """Return the first existing column as a Series; else Series of fallback."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([fallback_val] * len(df), index=df.index)

def read_metadata_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=0)
    try:
        return pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True)
    except Exception:
        # fallbacks
        for sep in [",", "\t", r"\s+"]:
            try:
                return pd.read_csv(path, sep=sep, engine="python", comment="#", skip_blank_lines=True)
            except Exception:
                pass
        raise

def mat_to_edge_index(mat: torch.Tensor, thr: float = 0.0):
    mask = (mat > thr)
    idx = mask.nonzero(as_tuple=False)  # [E,2]
    return idx.t().contiguous()         # [2,E]

# =========================
#        MODEL DEF
# =========================
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
        u = data.u  # expect [B, 3]
        if u.dim() == 1:
            u = u.unsqueeze(0)
        x = torch.cat([x, u], dim=1)
        return self.lin_out(x).squeeze(1)

# =========================
#           MAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Load metadata
    meta = read_metadata_any(META_CSV)
    meta = meta.drop_duplicates().reset_index(drop=True)
    print(f"[meta] loaded: shape={meta.shape}")

    # Ensure ID column MRI_Exam_fixed
    if "MRI_Exam_fixed" not in meta.columns:
        if "DWI" in meta.columns:
            meta["MRI_Exam_fixed"] = meta["DWI"].astype(str)
        else:
            raise KeyError("Need 'MRI_Exam_fixed' or 'DWI' in metadata.")

    # Age column must exist and be numeric for BAG/cBAG
    if "Age" not in meta.columns:
        raise KeyError("Metadata must contain 'Age' column.")
    meta["Age"] = to_numeric(meta["Age"])

    # Demographics (robust to missing columns; avoids .apply on scalar)
    sex_series  = pick_series(meta, ["sex", "Sex", "ID_Gender", "Gender"])
    apoe_series = pick_series(meta, ["apoe4_genotype", "APOE4_Genotype"], fallback_val="")
    meta["sex_num"]       = sex_series.apply(sex_to_num).astype(float).fillna(0.0)
    meta["APOE4_carrier"] = apoe_series.astype(str).str.contains("4").astype(float).fillna(0.0)
    meta["weight"]        = 1.0

    # ---- Load normalized node features dict (ALL subjects)
    if not os.path.exists(NORMAL_PT):
        raise FileNotFoundError(f"Missing normalized features: {NORMAL_PT}")
    normalized_node_features_dict = torch.load(NORMAL_PT)
    print(f"[features] normalized_node_features_dict: {len(normalized_node_features_dict)} subjects")

    # ---- Load connectomes for ALL subjects from disk
    print("\nCONNECTOMES")
    connectomes = {}
    for fname in os.listdir(CONN_DIR):
        if fname.endswith("_harmonized.csv"):
            sid = fname.replace("_harmonized.csv", "")  # e.g., H4369_y0
            fpath = os.path.join(CONN_DIR, fname)
            df_conn = pd.read_csv(fpath, header=None)
            connectomes[sid] = df_conn
    # simple filter
    connectomes = {k: v for k, v in connectomes.items() if "_whitematter" not in k}
    print("Connectomes:", len(connectomes))

    # ---- Final overlap for inference
    meta_ids = set(meta["MRI_Exam_fixed"].astype(str))
    feat_ids = set(map(str, normalized_node_features_dict.keys()))
    conn_ids = set(map(str, connectomes.keys()))
    final_ids = sorted(list(meta_ids & feat_ids & conn_ids))
    print(f"Subjects available for inference (intersection): {len(final_ids)}")

    if len(final_ids) == 0:
        raise RuntimeError("No subjects available for inference. Check IDs alignment.")

    # ---- Build PyG graphs
    id_to_age   = meta.set_index("MRI_Exam_fixed")["Age"].to_dict()
    id_to_demo  = meta.set_index("MRI_Exam_fixed")[["sex_num","APOE4_carrier","weight"]].to_dict(orient="index")

    graphs = []
    graph_ids = []
    for sid in final_ids:
        # adjacency
        conn_np = np.asarray(connectomes[sid].values, dtype=np.float32)
        edge_index = mat_to_edge_index(torch.from_numpy(conn_np), thr=0.0)
        # node features
        if sid not in normalized_node_features_dict:
            continue
        x = normalized_node_features_dict[sid]  # torch.Tensor [84,3]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # demographics [1,3] for batching -> [B,3]
        demo = id_to_demo.get(sid, {"sex_num":0.0, "APOE4_carrier":0.0, "weight":1.0})
        u = torch.tensor([[float(demo["sex_num"]), float(demo["APOE4_carrier"]), float(demo["weight"])]],
                         dtype=torch.float32)

        # Label is age (if present). We can keep it for convenience, but not required to predict.
        y_val = id_to_age.get(sid, np.nan)
        if np.isnan(y_val):
            y = torch.tensor([0.0], dtype=torch.float32)  # dummy
        else:
            y = torch.tensor([float(y_val)], dtype=torch.float32)

        graphs.append(Data(x=x, edge_index=edge_index, y=y, u=u))
        graph_ids.append(sid)

    print("Graph objects:", len(graphs))

    # ---- Load trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    model = BrainAgeGAT().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---- Predict
    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            yhat = model(batch)        # [B]
            preds.extend(yhat.detach().cpu().numpy().tolist())

    pred_df = pd.DataFrame({
        "MRI_Exam_fixed": graph_ids,
        "PredictedAge_AB": np.array(preds, dtype=float)
    })

    # ---- Compute BAG & bias-corrected BAG
    # Merge predicted ages back to metadata
    out = meta.merge(pred_df, on="MRI_Exam_fixed", how="left")

    # BAG = Predicted - Age
    out["BAG_AB"] = out["PredictedAge_AB"] - out["Age"]

    # Bias correction: regress PredictedAge_AB ~ Age on rows with both present, use residuals as cBAG
    fit_mask = (~out["PredictedAge_AB"].isna()) & (~out["Age"].isna())
    if fit_mask.sum() >= 3:
        reg = LinearRegression().fit(out.loc[fit_mask, ["Age"]].values, out.loc[fit_mask, "PredictedAge_AB"].values)
        expected = reg.predict(out.loc[fit_mask, ["Age"]].values)
        resid = out.loc[fit_mask, "PredictedAge_AB"].values - expected
        out.loc[fit_mask, "cBAG_AB"] = resid
        # corrected predicted age = Age + residual
        out.loc[fit_mask, "PredictedAge_corrected_AB"] = out.loc[fit_mask, "Age"].values + resid
        print(f"[bias] Predicted ~ Age: slope={reg.coef_[0]:.4f}, intercept={reg.intercept_:.4f}")
    else:
        out["cBAG_AB"] = np.nan
        out["PredictedAge_corrected_AB"] = np.nan
        print("[bias] Not enough data to fit bias correction (need >=3 rows with Age & PredictedAge_AB).")

    # ---- Save
    PRED_CSV = os.path.join(RESULTS_DIR, "predictions_all_subjects_AB.csv")
    OUT_CSV  = os.path.splitext(META_CSV)[0] + "_with_PredictedAgeAB_BAG.csv"

    pred_df.to_csv(PRED_CSV, index=False)
    out.to_csv(OUT_CSV, index=False)

    print(f"\n[OK] Saved predictions: {PRED_CSV}")
    print(f"[OK] Saved enriched metadata: {OUT_CSV}")
    # Small preview
    print(out[["MRI_Exam_fixed","Age","PredictedAge_AB","BAG_AB","cBAG_AB","PredictedAge_corrected_AB"]].head(10))

if __name__ == "__main__":
    main()
