#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alex4gaudi_columns_v10_heatmap_deeplay.py

Train GAUDI (deeplay-based) on MD, QSM, and joint connectome graphs,
attach standardized metadata, visualize latent clusters,
perform trait-level statistical tests, and generate
subject×feature heatmaps.

Author: Alexandra Badea
Date: Nov 3, 2025
"""

import os
import numpy as np
import pandas as pd
import torch
import deeplay as dl
from components import GraphEncoder, GraphDecoder
from applications import VariationalGraphAutoEncoder
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader   # ✅ FIXED
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway, kruskal, shapiro
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CONFIGURATION
# ==========================================================
def resolve_base_path():
    base_env = os.path.expandvars("$PAROS")
    candidate1 = os.path.join(base_env, "paros_WORK", "alex", "alex4gaudi", "GAUDI-implementation", "Graph_data")
    candidate2 = os.path.join(base_env, "alex", "alex4gaudi", "GAUDI-implementation", "Graph_data")
    if os.path.exists(candidate1):
        return candidate1
    elif os.path.exists(candidate2):
        return candidate2
    else:
        raise FileNotFoundError(f"Could not find Graph_data directory. Checked:\n{candidate1}\n{candidate2}")

BASE_PATH = resolve_base_path()
print(f"✅ Using base path: {BASE_PATH}")

def find_file(base, pattern):
    for f in os.listdir(base):
        if f.lower() == pattern.lower():
            return os.path.join(base, f)
    raise FileNotFoundError(f"Could not find {pattern} in {base}")

GRAPHS_MD = find_file(BASE_PATH, "graph_data_list_md.pt")
GRAPHS_QSM = find_file(BASE_PATH, "graph_data_list_QSM.pt")
META_PATH = find_file(BASE_PATH, "metadata_with_PCs.xlsx")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 10
EPOCHS = 50
BATCH_SIZE = 8
N_CLUSTERS = 4

sns.set(style="whitegrid", font_scale=1.1)

# ==========================================================
# UTILITIES
# ==========================================================
def zscore_features(graphs):
    for g in graphs:
        mean, std = g.x.mean(0, keepdim=True), g.x.std(0, keepdim=True)
        std[std == 0] = 1
        g.x = (g.x - mean) / std
    return graphs

def rebuild_edges_topk(graphs, k=10):
    for g in graphs:
        x = g.x
        corr = torch.corrcoef(x.T)
        corr.fill_diagonal_(0)
        edge_index, edge_attr = [], []
        for i in range(corr.size(0)):
            topk_idx = torch.topk(corr[i], k=k).indices
            for j in topk_idx:
                edge_index.append([i, j.item()])
                edge_attr.append([corr[i, j].item()])
        g.edge_index = torch.tensor(edge_index, dtype=torch.long).T
        g.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    return graphs

def load_metadata():
    meta = pd.read_excel(META_PATH)
    meta = meta.replace([np.inf, -np.inf], np.nan)
    num_cols = meta.select_dtypes(include=[np.number]).columns
    print(f"Attaching {len(num_cols)} numeric metadata columns.")
    na_counts = meta[num_cols].isna().sum()
    if na_counts.sum() > 0:
        print("\n⚠️ Missing values detected; replacing with column means:")
        print(na_counts[na_counts > 0].to_string())
    for c in num_cols:
        meta[c] = meta[c].fillna(meta[c].mean())
    scaler = StandardScaler()
    meta[num_cols] = scaler.fit_transform(meta[num_cols])
    return meta, num_cols

def attach_metadata(graphs, meta, numeric_cols):
    meta = meta.reset_index(drop=True)
    for i, g in enumerate(graphs):
        for c in numeric_cols:
            g[f"meta_{c}"] = float(meta.loc[i, c])
    return graphs

# ==========================================================
# CLUSTER STATS & VISUALIZATION
# ==========================================================
def cluster_stats_analysis(meta_with_clusters, label, outdir):
    results = []
    numeric_cols = meta_with_clusters.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["Cluster", "SilhouetteScore"]]

    for trait in numeric_cols:
        groups = [
            meta_with_clusters.loc[meta_with_clusters["Cluster"] == c, trait].dropna()
            for c in sorted(meta_with_clusters["Cluster"].unique())
        ]
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            continue
        normal = all(shapiro(g)[1] > 0.05 for g in groups if len(g) < 50)
        try:
            if normal:
                stat, p = f_oneway(*groups)
                test = "ANOVA"
            else:
                stat, p = kruskal(*groups)
                test = "Kruskal–Wallis"
        except Exception:
            continue
        results.append(
            {"Trait": trait, "Test": test, "Stat": stat, "p_value": p, "Significant(p<0.05)": p < 0.05}
        )

    df_stats = pd.DataFrame(results).sort_values("p_value")
    out_csv = os.path.join(outdir, f"{label}_cluster_trait_stats.csv")
    df_stats.to_csv(out_csv, index=False)
    print(f"[{label}] Cluster-trait statistical results → {out_csv}")

    top = df_stats.head(15)
    plt.figure(figsize=(6, 5))
    sns.barplot(data=top, y="Trait", x="p_value", hue="Test", dodge=False)
    plt.xscale("log")
    plt.xlabel("p-value (log scale)")
    plt.title(f"{label} – Top Trait Differences Across Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_cluster_trait_stats.png"))
    plt.close()

def plot_subject_feature_heatmap(meta_with_clusters, label, outdir, top_features=20):
    print(f"[{label}] Building subject–feature heatmap ...")

    numeric_cols = meta_with_clusters.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["Cluster", "SilhouetteScore"]]

    variances = meta_with_clusters[numeric_cols].var().sort_values(ascending=False)
    top_feats = variances.head(top_features).index.tolist()

    df = meta_with_clusters.copy().sort_values("Cluster")
    data = df[top_feats]
    data = (data - data.mean()) / data.std()

    cluster_colors = sns.color_palette("tab10", n_colors=len(df["Cluster"].unique()))
    cluster_lut = {cl: cluster_colors[i] for i, cl in enumerate(sorted(df["Cluster"].unique()))}
    row_colors = df["Cluster"].map(cluster_lut)

    g = sns.clustermap(
        data,
        row_cluster=False,
        col_cluster=True,
        row_colors=row_colors,
        cmap="vlag",
        figsize=(12, max(6, len(df) * 0.15)),
        xticklabels=top_feats,
        yticklabels=False,
    )
    g.fig.suptitle(f"{label} – Subject × Feature Heatmap (Top {top_features})", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_subject_feature_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[{label}] Heatmap saved → {outdir}/{label}_subject_feature_heatmap.png")

# ==========================================================
# EMBEDDING VISUALIZATION
# ==========================================================
def visualize_embeddings(z, label, meta, outdir):
    os.makedirs(outdir, exist_ok=True)
    print(f"[{label}] Running UMAP + KMeans + metadata export ...")

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
    z_umap = reducer.fit_transform(z)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    clusters = km.fit_predict(z)
    sil_score_val = silhouette_score(z, clusters)
    print(f"[{label}] Silhouette Score = {sil_score_val:.3f}")

    np.save(os.path.join(outdir, f"{label}_latent.npy"), z)
    np.save(os.path.join(outdir, f"{label}_UMAP.npy"), z_umap)
    np.save(os.path.join(outdir, f"{label}_clusters.npy"), clusters)

    df_vis = pd.DataFrame({"UMAP1": z_umap[:, 0], "UMAP2": z_umap[:, 1], "Cluster": clusters})
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df_vis, x="UMAP1", y="UMAP2", hue="Cluster", palette="tab10", s=35)
    plt.title(f"{label} – {N_CLUSTERS} Clusters (Sil={sil_score_val:.2f})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_KMeans_UMAP.png"))
    plt.close()

    meta_with_clusters = meta.iloc[:len(z)].copy()
    meta_with_clusters["Cluster"] = clusters
    meta_with_clusters["SilhouetteScore"] = sil_score_val
    subj_out = os.path.join(outdir, f"{label}_subject_clusters.csv")
    meta_with_clusters.to_csv(subj_out, index=False)
    print(f"[{label}] Subject-level clusters saved → {subj_out}")

    cluster_stats_analysis(meta_with_clusters, label, outdir)
    plot_subject_feature_heatmap(meta_with_clusters, label, outdir)
    return meta_with_clusters

# ==========================================================
# TRAINING (DEEPLAY-STYLE)
# ==========================================================
# ==========================================================
# TRAINING (DEEPLAY-STYLE)
# ==========================================================
def train_gaudi(graphs, meta, label="MD", epochs=20):
    print(f"\n[{label}] Training with {len(graphs)} graphs")

    # ✅ Add reconstruction targets before training
    for g in graphs:
        g.y = [g.x, g.edge_attr]

    sample_graph = graphs[0]
    n_features = sample_graph.x.shape[1]
    e_features = sample_graph.edge_attr.shape[1]
    print(f"[{label}] Node features: {n_features}, Edge features: {e_features}")

    encoder = GraphEncoder(
        hidden_features=96,
        num_blocks=3,
        num_clusters=[20, 5, 1],
        thresholds=[1 / 19, 1 / 5, None]
    )

    decoder = GraphDecoder(
        hidden_features=96,
        num_blocks=3,
        output_node_dim=n_features,
        output_edge_dim=e_features
    )

    vgae = VariationalGraphAutoEncoder(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        optimizer=dl.Adam(lr=1e-4),
        alpha=2,
        beta=1e-4,
        gamma=5,
        delta=1
    )

    VGAE = vgae.build()
    print(f"[{label}] Model built. Starting training...")

    # ✅ Graphs now have .y before batching
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
    trainer = dl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(VGAE, loader)
    trainer.history.plot(title=f"Training loss ({label})")

    VGAE.eval()
    embeddings = []
    for g in graphs:
        g_batch = Batch.from_data_list([g])
        out = VGAE(g_batch)
        embeddings.append(out["mu"].detach().numpy())
    embeddings = np.vstack(embeddings)

    outdir = os.path.join(BASE_PATH, f"latent_epochs_{label}")
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"latent_final_{label}.npy"), embeddings)
    print(f"[{label}] Latent embeddings saved → {outdir}/latent_final_{label}.npy")

    return embeddings


# ==========================================================
# MAIN PIPELINE
# ==========================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    graphs_md = torch.load(GRAPHS_MD, map_location="cpu")
    graphs_qsm = torch.load(GRAPHS_QSM, map_location="cpu")
    graphs_md = rebuild_edges_topk(zscore_features(graphs_md), TOPK)
    graphs_qsm = rebuild_edges_topk(zscore_features(graphs_qsm), TOPK)

    meta, numeric_cols = load_metadata()
    graphs_md = attach_metadata(graphs_md, meta, numeric_cols)
    graphs_qsm = attach_metadata(graphs_qsm, meta, numeric_cols)

    # Train GAUDI
    z_md = train_gaudi(graphs_md, meta, label="MD", epochs=EPOCHS)
    z_qsm = train_gaudi(graphs_qsm, meta, label="QSM", epochs=EPOCHS)

    # Visualize per modality
    md_out = os.path.join(BASE_PATH, "latent_epochs_MD")
    qsm_out = os.path.join(BASE_PATH, "latent_epochs_QSM")
    meta_md = visualize_embeddings(z_md, "MD", meta, md_out)
    meta_qsm = visualize_embeddings(z_qsm, "QSM", meta, qsm_out)

    # ---- Joint ----
    z_joint = np.concatenate([z_md, z_qsm], axis=1)
    joint_dir = os.path.join(BASE_PATH, "latent_epochs_joint")
    os.makedirs(joint_dir, exist_ok=True)
    np.save(os.path.join(joint_dir, "latent_final_joint.npy"), z_joint)
    print("[Joint] Latent embedding saved → latent_final_joint.npy")

    meta_joint = visualize_embeddings(z_joint, "Joint", meta, joint_dir)
