#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 17:18:11 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAUDI Latent Space Comparison: Correlation vs Wasserstein
---------------------------------------------------------
Loads latent embeddings from both correlation- and Wasserstein-based GAUDI runs,
computes visualization, cluster-trait correlations, and a side-by-side summary.

Author: Alexandra Badea
Date: Nov 3, 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr, f_oneway, kruskal, shapiro
from statsmodels.stats.multitest import fdrcorrection
from matplotlib.backends.backend_pdf import PdfPages
import umap

# ==========================================================
# CONFIG
# ==========================================================
OUT_ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation/processed_graph_data_110325"
META_PATH = os.path.join(OUT_ROOT, "metadata_with_PCs.xlsx")

MODALITIES = ["MD", "QSM", "Joint"]
VERSIONS = ["corr", "WASS"]
N_CLUSTERS = 4
sns.set(style="whitegrid", font_scale=1.1)

# ==========================================================
# UTILITIES
# ==========================================================
def load_latents(label, version):
    path = os.path.join(OUT_ROOT, f"latent_epochs_{label}_{'' if version=='corr' else version}")
    if not os.path.exists(path):
        path = os.path.join(OUT_ROOT, f"latent_epochs_{label}_{version}")
    latent_file = os.path.join(path, f"latent_final_{label}_{'' if version=='corr' else version}.npy")
    if not os.path.exists(latent_file):
        latent_file = os.path.join(path, f"latent_final_{label}.npy")  # fallback
    z = np.load(latent_file)
    return z, path

def fdr_correct(df, pcol="p_value"):
    df = df.copy()
    if pcol in df.columns:
        df["FDR_p"] = fdrcorrection(df[pcol].fillna(1))[1]
        df["Significant(FDR<0.05)"] = df["FDR_p"] < 0.05
    return df

def cluster_and_trait_tests(meta, z, label, outdir):
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    clusters = km.fit_predict(z)
    sil = silhouette_score(z, clusters)
    meta = meta.iloc[:len(z)].copy()
    meta["Cluster"] = clusters

    # ---- Stats ----
    results = []
    num_cols = meta.select_dtypes(include=[np.number]).columns
    for trait in num_cols:
        if trait == "Cluster":
            continue
        groups = [meta.loc[meta["Cluster"] == c, trait].dropna() for c in sorted(meta["Cluster"].unique())]
        if len(groups) < 2:
            continue
        normal = all(shapiro(g)[1] > 0.05 for g in groups if len(g) >= 5)
        try:
            if normal:
                stat, p = f_oneway(*groups)
                test = "ANOVA"
            else:
                stat, p = kruskal(*groups)
                test = "Kruskal–Wallis"
        except Exception:
            continue
        results.append({"Trait": trait, "Test": test, "p_value": p})
    df = pd.DataFrame(results).sort_values("p_value")
    df = fdr_correct(df)
    df.to_csv(os.path.join(outdir, f"{label}_cluster_trait_stats.csv"), index=False)
    return df, sil

def latent_trait_corr(z, meta, label, outdir, topn=20):
    num_cols = meta.select_dtypes(include=[np.number]).columns
    results = []
    for i in range(z.shape[1]):
        for trait in num_cols:
            r, p = spearmanr(z[:, i], meta[trait], nan_policy="omit")
            results.append({"Latent": i+1, "Trait": trait, "Spearman_r": r, "p_value": p})
    df = pd.DataFrame(results)
    df = fdr_correct(df)
    df.to_csv(os.path.join(outdir, f"{label}_latent_trait_corr.csv"), index=False)

    # Top correlations
    df_top = df.sort_values("FDR_p").head(topn)
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=df_top, x="Spearman_r", y="Trait", hue="Latent", palette="tab10", s=60)
    plt.axvline(0, color="k", lw=0.8)
    plt.title(f"{label}: Top {topn} Latent–Trait Correlations (FDR<0.05)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_latent_trait_corr.png"), dpi=300)
    plt.close()
    return df_top

def visualize_umap(z, label, outdir):
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
    z_umap = reducer.fit_transform(z)
    np.save(os.path.join(outdir, f"{label}_UMAP.npy"), z_umap)
    plt.figure(figsize=(5,4))
    plt.scatter(z_umap[:,0], z_umap[:,1], s=25, alpha=0.7)
    plt.title(f"{label} – UMAP Projection")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_UMAP.png"), dpi=300)
    plt.close()
    return z_umap

# ==========================================================
# MAIN COMPARISON
# ==========================================================
if __name__ == "__main__":
    meta = pd.read_excel(META_PATH)
    meta = meta.replace([np.inf, -np.inf], np.nan)
    meta = meta.fillna(meta.mean(numeric_only=True))

    comparison_rows = []

    with PdfPages(os.path.join(OUT_ROOT, "GAUDI_corr_vs_wass_report.pdf")) as pdf:
        for modality in MODALITIES:
            for version in VERSIONS:
                print(f"\n📊 {modality} ({version})")
                z, outdir = load_latents(modality, version)
                os.makedirs(outdir, exist_ok=True)

                # UMAP visualization
                z_umap = visualize_umap(z, f"{modality}_{version}", outdir)

                # Trait correlations
                df_top = latent_trait_corr(z, meta, f"{modality}_{version}", outdir)

                # Cluster–trait analysis
                df_stats, sil = cluster_and_trait_tests(meta, z, f"{modality}_{version}", outdir)
                comparison_rows.append({
                    "Modality": modality,
                    "Version": version,
                    "Silhouette": sil,
                    "TopTrait": df_stats.iloc[0]["Trait"] if not df_stats.empty else "None",
                    "TopTrait_FDRp": df_stats.iloc[0]["FDR_p"] if not df_stats.empty else np.nan
                })

                # Add a summary page to the PDF
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.scatterplot(data=z_umap, x=z_umap[:, 0], y=z_umap[:, 1],
                                color="steelblue", s=20, alpha=0.8)
                ax.set_title(f"{modality} ({version}) – Silhouette={sil:.2f}")
                pdf.savefig(fig)
                plt.close(fig)

        # Combined summary table
        df_comp = pd.DataFrame(comparison_rows)
        df_comp.to_csv(os.path.join(OUT_ROOT, "corr_vs_wass_summary.csv"), index=False)

        plt.figure(figsize=(6,4))
        sns.barplot(data=df_comp, x="Modality", y="Silhouette", hue="Version")
        plt.title("Silhouette Score Comparison (corr vs WASS)")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        print("\n✅ Comparison complete. Report saved → GAUDI_corr_vs_wass_report.pdf")
        print("   Summary table → corr_vs_wass_summary.csv")
