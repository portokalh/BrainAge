#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HABS healthy pipeline (Risk == 0)
- Reads metadata (CSV or XLSX)
- Computes Risk (diabetes OR cognitive impairment)
- Saves CSVs of all + healthy
- Loads connectomes and matches
- Loads FA/MD/VOL with MRI_Exam_fixed indices (H####_y#)
- Builds multimodal node features dict [84, 3] per subject, z-scored
- Adds placeholder biomarkers (Abeta/Tau/GFAP = NaN)
- Maps sex to 0/1, APOE4 carrier to 0/1, sets weight = 1.0
"""

import os, re, random
import numpy as np
import pandas as pd
import torch

# =========================
#         PATHS
# =========================
CONN_DIR = "/home/bas/Desktop/MyData/harmonization/HABS/connectomes/DWI/plain/"

# Metadata can be CSV or XLSX; set one path:
HABS_META_PATH = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_metadata.xlsx"
HABS_META_SHEET = 0          # set to a sheet name or index if XLSX (ignored for CSV)

FA_PATH  = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_Regional_Stats/HABS_studywide_stats_for_fa.txt"
MD_PATH  = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_Regional_Stats/HABS_studywide_stats_for_md.txt"
VOL_PATH = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/HABS_Regional_Stats/HABS_studywide_stats_for_volume.txt"


outdir = "/home/bas/Desktop/MyData/harmonization/HABS/metadata/"

df_matched_connectomes.to_csv(os.path.join(outdir, "HABS_with_risk.csv"), index=False)
df_matched_habs_healthy.to_csv(os.path.join(outdir, "HABS_healthy_controls.csv"), index=False)
df_metadata_PCA_healthy_withConnectome.to_csv(os.path.join(outdir, "HABS_healthy_metadata_with_placeholders.csv"), index=False)

print(f"\nSaved CSVs to: {outdir}")

# =========================
#       UTILITIES
# =========================
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def read_metadata_any(path: str, sheet=None) -> pd.DataFrame:
    """Read CSV or XLSX metadata with minimal fuss."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
        print(f"[meta] XLSX loaded: shape={df.shape}")
        return df
    # CSV/TSV/misc: try auto-detect first
    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True)
        print(f"[meta] CSV autodetect loaded: shape={df.shape}")
        return df
    except Exception:
        # Common fallbacks
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
    """
    Reads studywide stats (tsv), selects subject columns, transposes to subjects x ROIs,
    coerces numeric, and indexes by MRI_Exam_fixed-style keys (e.g., 'H4369_y0').
    """
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
    """Z-score across subjects for each ROI/feature channel. Input: sid -> [84,3] tensor."""
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

# =========================
#          RUN
# =========================
seed_everything(42)

# ---- Load connectomes (dict: MRI_Exam_fixed -> 84x84 DataFrame) ----
print("CONNECTOMES\n")
connectomes = {}
for fname in os.listdir(CONN_DIR):
    if fname.endswith("_harmonized.csv"):
        sid = fname.replace("_harmonized.csv", "")  # e.g., "H4369_y0"
        fpath = os.path.join(CONN_DIR, fname)
        df_conn = pd.read_csv(fpath, header=None)
        connectomes[sid] = df_conn

print(f"Total connectome matrices loaded: {len(connectomes)}")
connectomes = {k: v for k, v in connectomes.items() if "_whitematter" not in k}
print(f"Total connectomes after filtering: {len(connectomes)}")
if connectomes:
    example_id = next(iter(connectomes))
    print("Example connectome:", example_id, connectomes[example_id].shape)

# ---- Load metadata (CSV or XLSX) ----
print("\nHABS METADATA\n")
df_meta = read_metadata_any(HABS_META_PATH, sheet=HABS_META_SHEET)
df_meta = df_meta.drop_duplicates().reset_index(drop=True)

# Ensure ID col named MRI_Exam_fixed exists
if "MRI_Exam_fixed" not in df_meta.columns:
    if "DWI" in df_meta.columns:
        df_meta["MRI_Exam_fixed"] = df_meta["DWI"].astype(str)
    else:
        raise KeyError("Need 'MRI_Exam_fixed' or 'DWI' in metadata.")

# ---- Match metadata to available connectomes ----
df_matched_connectomes = df_meta[df_meta["MRI_Exam_fixed"].isin(connectomes.keys())].copy()
print(f"Matched subjects (metadata & connectome): {len(df_matched_connectomes)} / {len(connectomes)}")

# ---- Risk columns (HABS rules) ----
df_matched_connectomes["cdx_cogx"] = to_numeric(df_matched_connectomes.get("cdx_cogx", np.nan))
df_matched_connectomes["cdx_diabetesx"] = to_numeric(df_matched_connectomes.get("cdx_diabetesx", np.nan))

dementia_bool = df_matched_connectomes["cdx_cogx"].isin([2, 3]).fillna(False)              # MCI/AD => risk
diabetes_bool = (df_matched_connectomes["cdx_diabetesx"].fillna(0).astype(int) == 1)      # diabetes => risk

df_matched_connectomes["Dementia_Risk"] = dementia_bool.astype(int)
df_matched_connectomes["Diabetes_Risk"] = diabetes_bool.astype(int)
df_matched_connectomes["Risk"] = ((df_matched_connectomes["Dementia_Risk"] == 1) |
                                  (df_matched_connectomes["Diabetes_Risk"] == 1)).astype(int)

cogx_map = {0: "CN", 1: "Other/SMC", 2: "MCI", 3: "Dementia/AD"}
df_matched_connectomes["CogDx_Label"] = df_matched_connectomes["cdx_cogx"].map(cogx_map).fillna("Unknown")

print("\nRisk counts:")
print(df_matched_connectomes["Risk"].value_counts(dropna=False))

# ---- Filter to healthy (Risk == 0) ----
before_n = len(df_matched_connectomes)
df_matched_habs_healthy = df_matched_connectomes[df_matched_connectomes["Risk"] == 0].copy()
after_n = len(df_matched_habs_healthy)
print(f"\nSubjects before filtering: {before_n}")
print(f"Subjects after keeping Risk == 0: {after_n}")
print(f"Removed: {before_n - after_n}")

# Save CSVs
df_matched_connectomes.to_csv("HABS_with_risk.csv", index=False)
df_matched_habs_healthy.to_csv("HABS_healthy_controls.csv", index=False)
print("Saved: HABS_with_risk.csv, HABS_healthy_controls.csv")

# Removal summary
removed_df = df_matched_connectomes[df_matched_connectomes["Risk"] == 1]
n_diab = removed_df["Diabetes_Risk"].sum()
n_dem  = removed_df["Dementia_Risk"].sum()
n_both = ((removed_df["Diabetes_Risk"] == 1) & (removed_df["Dementia_Risk"] == 1)).sum()
print("Removal summary:", {"total_removed": len(removed_df), "diabetes": int(n_diab), "dementia": int(n_dem), "both": int(n_both)})

# ---- Healthy connectome dict (Risk==0) ----
matched_connectomes_healthy = {
    row["MRI_Exam_fixed"]: connectomes[row["MRI_Exam_fixed"]]
    for _, row in df_matched_habs_healthy.iterrows()
    if row["MRI_Exam_fixed"] in connectomes
}
print(f"Connectomes selected (healthy only): {len(matched_connectomes_healthy)}")

# ---- Placeholder biomarkers on healthy metadata ----
df_metadata_PCA_healthy_withConnectome = df_matched_habs_healthy.copy()
for col in ["Abeta", "Tau", "GFAP"]:
    df_metadata_PCA_healthy_withConnectome[col] = np.nan

# ---- Load FA/MD/VOL with MRI_Exam_fixed indices ----
df_fa_clean  = load_and_clean_metric_to_mri_exam(FA_PATH)
df_md_clean  = load_and_clean_metric_to_mri_exam(MD_PATH)
df_vol_clean = load_and_clean_metric_to_mri_exam(VOL_PATH)
print("\nFA/MD/VOL shapes:", df_fa_clean.shape, df_md_clean.shape, df_vol_clean.shape)

# ---- Build multimodal node features: [84,3] per subject ----
multimodal_features_dict = {}
for sid in df_fa_clean.index:
    if sid in df_md_clean.index and sid in df_vol_clean.index:
        fa  = torch.tensor(df_fa_clean.loc[sid].values,  dtype=torch.float32)
        md  = torch.tensor(df_md_clean.loc[sid].values,  dtype=torch.float32)
        vol = torch.tensor(df_vol_clean.loc[sid].values, dtype=torch.float32)
        stacked = torch.stack([fa, md, vol], dim=1)  # [84, 3]
        multimodal_features_dict[sid] = stacked

print("Subjects with FA/MD/Vol features:", len(multimodal_features_dict))
normalized_node_features_dict = normalize_multimodal_nodewise(multimodal_features_dict)

# ---- Demographics: sex_num, APOE4_carrier, weight=1.0 ----
df_metadata_PCA_healthy_withConnectome["sex_num"] = df_metadata_PCA_healthy_withConnectome.get("sex", np.nan).apply(sex_to_num)
df_metadata_PCA_healthy_withConnectome["APOE4_carrier"] = (
    df_metadata_PCA_healthy_withConnectome.get("apoe4_genotype", "").astype(str).str.contains("4").astype(int)
)
df_metadata_PCA_healthy_withConnectome["weight"] = 1.0

key_col = "MRI_Exam_fixed"
meta_for_tensors = df_metadata_PCA_healthy_withConnectome.dropna(subset=[key_col, "sex_num"])
subject_to_demographic_tensor = {
    row[key_col]: torch.tensor([row["sex_num"], row["APOE4_carrier"], row["weight"]], dtype=torch.float32)
    for _, row in meta_for_tensors.iterrows()
}
print(f"Built subject_to_demographic_tensor for {len(subject_to_demographic_tensor)} subjects.")

# ---- Overlaps (sanity) ----
fa_md_vol_ids = set(multimodal_features_dict.keys())
meta_ids      = set(df_metadata_PCA_healthy_withConnectome["MRI_Exam_fixed"])
conn_ids      = set(matched_connectomes_healthy.keys())

print("\nOverlap sizes:")
print(" FA/MD/Vol ∩ metadata:", len(fa_md_vol_ids & meta_ids))
print(" FA/MD/Vol ∩ connectomes:", len(fa_md_vol_ids & conn_ids))
print(" metadata ∩ connectomes:", len(meta_ids & conn_ids))
final_overlap = fa_md_vol_ids & meta_ids & conn_ids
print(" Subjects with FA/MD/Vol + metadata + connectome:", len(final_overlap))

# Optional save for your downstream steps
df_metadata_PCA_healthy_withConnectome.to_csv("HABS_healthy_metadata_with_placeholders.csv", index=False)
print("\nSaved: HABS_healthy_metadata_with_placeholders.csv")
