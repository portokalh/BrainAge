#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train GAUDI on real connectome graphs (MD + QSM) with Top-K correlation edges
and prefixed metadata fields.
Author: Alexandra Badea
Date: Oct 30, 2025
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import deeplay as dl
from components import GraphEncoder, GraphDecoder
from applications import VariationalGraphAutoEncoder
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Batch


# ==========================================================
# 1. Paths and data loading
# ==========================================================

PAROS = os.environ["PAROS"]
BASE_PATH = os.path.join(
    PAROS,
    "paros_WORK",
    "alex",
    "alex4gaudi",
    "GAUDI-implementation",
    "Graph_data"
)

print(f"Using base path: {BASE_PATH}")

graphs_md_path = os.path.join(BASE_PATH, "graph_data_list_md.pt")
graphs_qsm_path = os.path.join(BASE_PATH, "graph_data_list_QSM.pt")
metadata_path = os.path.join(BASE_PATH, "metadata_with_PCs.xlsx")

print("Loading graph data ...")
graphs_md = torch.load(graphs_md_path, map_location="cpu")
graphs_qsm = torch.load(graphs_qsm_path, map_location="cpu")
print(f"MD: {len(graphs_md)} graphs | QSM: {len(graphs_qsm)} graphs")

print("Loading metadata ...")
meta = pd.read_excel(metadata_path)
print(f"Metadata columns: {list(meta.columns)}")


# ==========================================================
# 2. Rebuild edges using Top-K correlation per graph
# ==========================================================

def rebuild_edges_topk(graph_list, k=10, symmetrize=True, absolute=True):
    """
    Build new edges for each graph by computing correlations between node features.
    Keep the top-k strongest connections per node.
    """
    new_graphs = []
    for g in graph_list:
        x = g.x  # [n_nodes, n_features]
        x_centered = x - x.mean(dim=0, keepdim=True)
        cov = torch.mm(x_centered, x_centered.T)
        std = x_centered.norm(dim=1, keepdim=True)
        corr = cov / (std @ std.T + 1e-8)
        corr.fill_diagonal_(0)
        if absolute:
            corr = corr.abs()

        topk_vals, topk_idx = torch.topk(corr, k=k, dim=1)
        edges, weights = [], []
        for i in range(corr.size(0)):
            for j, w in zip(topk_idx[i], topk_vals[i]):
                edges.append([i, j.item()])
                weights.append([w.item()])

        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.tensor(weights, dtype=torch.float32)

        if symmetrize:
            edge_index_rev = edge_index[[1, 0], :]
            edge_attr_rev = edge_attr.clone()
            edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr_rev], dim=0)

        g.edge_index = edge_index
        g.edge_attr = edge_attr
        new_graphs.append(g)
    return new_graphs


print("Rebuilding edges using Top-K correlations ...")
graphs_md = rebuild_edges_topk(graphs_md, k=10, symmetrize=True)
graphs_qsm = rebuild_edges_topk(graphs_qsm, k=10, symmetrize=True)
print("Edge rebuilding complete.")


# ==========================================================
# 3. Dataset definition with safe metadata prefixing
# ==========================================================

class GraphDataset(Dataset):
    def __init__(self, graphs, metadata):
        self.graphs = graphs
        self.meta = metadata.reset_index(drop=True)
        self.duplicate_keys = set()

        # Only keep numeric columns
        self.numeric_cols = [
            col for col in self.meta.columns
            if pd.api.types.is_numeric_dtype(self.meta[col])
        ]
        print(f"Attaching {len(self.numeric_cols)} numeric metadata columns to each graph.")

        # Replace missing values with column means
        na_counts = self.meta[self.numeric_cols].isna().sum()
        if na_counts.sum() > 0:
            print("\n⚠️ Missing values detected. Replacing NaNs with column means:")
            print(na_counts[na_counts > 0].to_string())
            self.meta[self.numeric_cols] = self.meta[self.numeric_cols].apply(
                lambda x: x.fillna(x.mean())
            )
        else:
            print("\nNo missing metadata values detected.")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx].clone()
        for col in self.numeric_cols:
            val = self.meta.loc[idx, col]
            safe_key = f"meta_{col}"
            if safe_key in g.keys():
                new_key = f"{safe_key}_meta"
                self.duplicate_keys.add((safe_key, new_key))
                safe_key = new_key
            g[safe_key] = torch.tensor(float(val), dtype=torch.float32)

        # Reconstruction targets for GAUDI
        g.y = [g.x, g.edge_attr]
        return g


# ==========================================================
# 4. Metadata logging utilities
# ==========================================================

def log_graph_metadata(graphs, n=3):
    """Print out metadata keys and example values for the first n graphs."""
    print(f"\n===== METADATA CHECK (showing {n} examples) =====")
    for i, g in enumerate(graphs[:n]):
        meta_keys = [k for k in g.keys() if k.startswith("meta_")]
        print(f"\nGraph {i} ({len(meta_keys)} metadata fields):")
        for key in meta_keys[:10]:
            val = g[key].item() if isinstance(g[key], torch.Tensor) and g[key].numel() == 1 else g[key]
            print(f"  {key}: {val}")
    print("=============================================\n")


def export_graph_metadata_summary(dataset, out_path):
    """Save a CSV summary of all attached metadata for verification."""
    summary = []
    for i in range(len(dataset)):
        g = dataset[i]
        row = {
            k: g[k].item() if isinstance(g[k], torch.Tensor) and g[k].numel() == 1 else None
            for k in g.keys() if k.startswith("meta_")
        }
        row["graph_idx"] = i
        summary.append(row)
    pd.DataFrame(summary).to_csv(out_path, index=False)
    print(f"Metadata summary saved to {out_path}")


# ==========================================================
# 5. GAUDI training function
# ==========================================================

def train_gaudi(graphs, meta, label="modality", epochs=20, save_prefix="latent"):
    dataset = GraphDataset(graphs, meta)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    if len(dataset.duplicate_keys) > 0:
        print("\n⚠️ Duplicate metadata keys detected and renamed:")
        for old_key, new_key in list(dataset.duplicate_keys)[:10]:
            print(f"   {old_key} → {new_key}")

    log_graph_metadata([dataset[i] for i in range(min(3, len(dataset)))])
    export_graph_metadata_summary(dataset, os.path.join(BASE_PATH, f"metadata_summary_{label}.csv"))

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

    trainer = dl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(VGAE, loader)
    trainer.history.plot(title=f"Training loss ({label})")

    VGAE.eval()
    embeddings = []
    for g in graphs:
        g_batch = Batch.from_data_list([g])  # ✅ ensures batch attribute exists
        out = VGAE(g_batch)
        embeddings.append(out["mu"].detach().numpy())
    embeddings = np.vstack(embeddings)

    save_path = os.path.join(BASE_PATH, f"{save_prefix}_{label}.npy")
    np.save(save_path, embeddings)
    print(f"[{label}] Latent embeddings saved to: {save_path}")
    return embeddings


# ==========================================================
# 6. Train separately on MD and QSM
# ==========================================================

z_md = train_gaudi(graphs_md, meta, label="MD", epochs=20, save_prefix="gaudi_latent")
z_qsm = train_gaudi(graphs_qsm, meta, label="QSM", epochs=20, save_prefix="gaudi_latent")


# ==========================================================
# 7. Multimodal fusion & clustering
# ==========================================================

z_joint = np.concatenate([z_md, z_qsm], axis=1)
np.save(os.path.join(BASE_PATH, "gaudi_latent_joint.npy"), z_joint)

z_2d = PCA(n_components=2).fit_transform(z_joint)
kmeans = KMeans(n_clusters=5, random_state=42).fit(z_joint)
labels = kmeans.labels_

plt.figure(figsize=(7, 6))
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", s=40)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("GAUDI Latent Space Clustering (MD + QSM, Top-K Correlation Edges)")
plt.tight_layout()
plt.show()
